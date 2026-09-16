"""Check GAN construction, one optimizer step, and a weight round trip on synthetic spectra."""

from pathlib import Path
import argparse
import sys
import tempfile
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_gan import ConditionalGAN
import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Optional trusted generator H5 weights")
    args = parser.parse_args()
    random.seed(666)
    np.random.seed(666)
    tf.random.set_seed(666)
    model = ConditionalGAN(1873, 1873, (65, 90), (160, 190), (705, 725))
    measured = tf.random.uniform((2, 1873), minval=0.1, maxval=0.8)
    reference = tf.random.uniform((2, 1873), minval=0.1, maxval=0.8)
    result = model.generator(measured, training=False).numpy()
    assert result.shape == (2, 1873) and np.isfinite(result).all()
    losses = model.train_step((reference, measured))
    assert all(np.isfinite(value.numpy()).all() for value in losses.values())
    expected = model.generator(measured, training=False).numpy()
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "generator.weights.h5"
        model.generator.save_weights(checkpoint)
        restored = ConditionalGAN(1873, 1873, (65, 90), (160, 190), (705, 725))
        restored.generator.load_weights(checkpoint)
        actual = restored.generator(measured, training=False).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    print(
        "PASS: generator/discriminator construction, finite optimizer step, and weight round trip."
    )
    if args.checkpoint:
        archived = ConditionalGAN(1873, 1873, (65, 90), (160, 190), (705, 725))
        archived.generator.load_weights(args.checkpoint)
        prediction = archived.generator(measured, training=False).numpy()
        assert prediction.shape == (2, 1873) and np.isfinite(prediction).all()
        print("PASS: supplied generator checkpoint loads and produces finite predictions.")
    print("Synthetic checks do not measure restoration accuracy or reproduce research training.")


if __name__ == "__main__":
    main()
