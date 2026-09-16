"""Conditional GAN architecture and losses from the spectral restoration experiment."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


def crop_to_match(src, tgt):
    def _crop(tensors):
        s, t = tensors
        return s[:, : tf.shape(t)[1], :]

    return layers.Lambda(_crop)([src, tgt])


def build_generator(ref_dim, out_dim):
    inp = Input(shape=(ref_dim,), name="interfered_spectrum")
    x = layers.Reshape((ref_dim, 1))(inp)

    # ===== Encoder =====
    e1 = layers.Conv1D(64, 4, 2, padding="same")(x)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.LeakyReLU(0.2)(e1)

    e2 = layers.Conv1D(128, 4, 2, padding="same")(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.LeakyReLU(0.2)(e2)

    e3 = layers.Conv1D(256, 4, 2, padding="same")(e2)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.LeakyReLU(0.2)(e3)

    e4 = layers.Conv1D(512, 4, 2, padding="same")(e3)
    e4 = layers.BatchNormalization()(e4)
    e4 = layers.LeakyReLU(0.2)(e4)

    # ===== Bottleneck =====
    b = layers.Conv1D(512, 4, 2, padding="same")(e4)
    b = layers.BatchNormalization()(b)
    b = layers.LeakyReLU(0.2)(b)

    # ===== Decoder =====
    d4 = layers.Conv1DTranspose(512, 4, 2, padding="same")(b)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.LeakyReLU(0.2)(d4)
    d4 = crop_to_match(d4, e4)
    d4 = layers.Concatenate()([d4, e4])

    d3 = layers.Conv1DTranspose(256, 4, 2, padding="same")(d4)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.LeakyReLU(0.2)(d3)
    d3 = crop_to_match(d3, e3)
    d3 = layers.Concatenate()([d3, e3])

    d2 = layers.Conv1DTranspose(128, 4, 2, padding="same")(d3)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.LeakyReLU(0.2)(d2)
    d2 = crop_to_match(d2, e2)
    d2 = layers.Concatenate()([d2, e2])

    d1 = layers.Conv1DTranspose(64, 4, 2, padding="same")(d2)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.LeakyReLU(0.2)(d1)
    d1 = crop_to_match(d1, e1)
    d1 = layers.Concatenate()([d1, e1])

    # ===== Output =====
    out = layers.Conv1DTranspose(32, 4, 2, padding="same")(d1)
    out = layers.LeakyReLU(0.2)(out)
    out = crop_to_match(out, x)

    out = layers.Conv1D(1, 3, padding="same", activation="sigmoid")(out)
    out = layers.Flatten()(out)

    out = layers.Lambda(lambda t: tf.ensure_shape(t, [None, out_dim]), name="clean_spectrum")(out)

    return Model(inp, out, name="Generator")


def build_discriminator(spec_dim, ref_dim):
    spec_inp = Input(shape=(spec_dim,))
    ref_inp = Input(shape=(ref_dim,))

    sx = layers.Reshape((spec_dim, 1))(spec_inp)
    rx = layers.Reshape((ref_dim, 1))(ref_inp)
    x = layers.Concatenate(axis=1)([sx, rx])

    for i, f in enumerate([64, 128, 128, 256]):
        x = layers.Conv1D(f, kernel_size=5, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    feat = layers.GlobalAveragePooling1D()(x)
    dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(feat)
    dense = layers.Dropout(0.3)(dense)
    out = layers.Dense(1, activation="sigmoid")(dense)

    D = Model([spec_inp, ref_inp], out, name="Discriminator")
    Fe = Model([spec_inp, ref_inp], feat, name="FeatureExtractor")
    return D, Fe


class ConditionalGAN(Model):
    def __init__(self, spec_dim, ref_dim, Zn_idx, Ni_idx, Cu_idx, lr=1e-4):
        super().__init__()
        self.generator, self.discriminator, self.feature_extractor = (
            build_generator(ref_dim, spec_dim),
            *build_discriminator(spec_dim, ref_dim),
        )
        self.g_opt = Adam(0.5 * lr)
        self.d_opt = Adam(lr)
        self.bce = BinaryCrossentropy(from_logits=False)

        self.Zn, self.Ni, self.Cu = Zn_idx, Ni_idx, Cu_idx
        self.l_adv_log = tf.Variable(tf.math.log(5.0), True)
        self.l_cos_log = tf.Variable(tf.math.log(1.0), True)
        self.l_feat_log = tf.Variable(tf.math.log(10.0), True)
        self.l_metal_log = tf.Variable(tf.math.log(50.0), True)

        self.g_optimizer = self.g_opt
        self.d_optimizer = self.d_opt

    def train_step(self, data):
        clean, x = data
        # ---- D ----
        with tf.GradientTape() as td:
            fake = self.generator(x, training=True)
            r, f = self.discriminator([clean, x], True), self.discriminator([fake, x], True)
            d_loss = self.bce(tf.ones_like(r) * 0.9, r) + self.bce(tf.zeros_like(f), f)
        self.d_opt.apply_gradients(
            zip(
                td.gradient(d_loss, self.discriminator.trainable_variables),
                self.discriminator.trainable_variables,
            )
        )
        # ---- G (+λ) ----
        with tf.GradientTape() as tg:
            fake = self.generator(x, training=True)
            f_pred = self.discriminator([fake, x], True)

            adv = self.bce(tf.ones_like(f_pred), f_pred)
            cos = tf.reduce_mean(1 + tf.keras.losses.cosine_similarity(fake, clean, axis=-1))
            feat = tf.reduce_mean(
                tf.abs(
                    self.feature_extractor([clean, x], False)
                    - self.feature_extractor([fake, x], False)
                )
            )

            real_m = tf.concat(
                [
                    clean[:, self.Zn[0] : self.Zn[1]],
                    clean[:, self.Ni[0] : self.Ni[1]],
                    clean[:, self.Cu[0] : self.Cu[1]],
                ],
                1,
            )
            fake_m = tf.concat(
                [
                    fake[:, self.Zn[0] : self.Zn[1]],
                    fake[:, self.Ni[0] : self.Ni[1]],
                    fake[:, self.Cu[0] : self.Cu[1]],
                ],
                1,
            )
            metal = tf.reduce_mean(tf.abs(fake_m - real_m))

            l_adv = tf.nn.softplus(self.l_adv_log)
            l_cos = tf.nn.softplus(self.l_cos_log)
            l_feat = tf.nn.softplus(self.l_feat_log)
            l_metal = tf.nn.softplus(self.l_metal_log)

            g_loss = l_adv * adv + l_cos * cos + l_feat * feat + l_metal * metal

        vars_g = self.generator.trainable_variables + [
            self.l_adv_log,
            self.l_cos_log,
            self.l_feat_log,
            self.l_metal_log,
        ]
        self.g_opt.apply_gradients(zip(tg.gradient(g_loss, vars_g), vars_g))

        D_real_mean = tf.reduce_mean(r)
        D_fake_mean = tf.reduce_mean(f)

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "adv": adv,
            "cos": cos,
            "feat": feat,
            "metal": metal,
            "λ_adv": l_adv,
            "λ_cos": l_cos,
            "λ_feat": l_feat,
            "λ_metal": l_metal,
            "D_real": D_real_mean,
            "D_fake": D_fake_mean,
        }
