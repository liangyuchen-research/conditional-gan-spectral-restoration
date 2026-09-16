# Reproduction guide

## Environment

The public runtime uses TensorFlow 2.16.2 and legacy `tf-keras` 2.16.0. The model module selects legacy Keras before TensorFlow import. Model smoke checks have passed on Python 3.12.14; full retraining has not been performed. Install dependencies in a separate environment. Optional Tk file dialogs may require a separate OS package on Linux.

Random seeds are present, but repeated seed-setting and backend settings do not establish bit-for-bit reproducibility across hardware and dependency versions. The original notebooks had no complete environment lockfile.

## Execution dependencies

The main notebook is the primary implementation. Training runs for 3,000 epochs in its preserved configuration, so running every cell starts a substantial training job. The privately preserved repeat-measurement variant uses 600 epochs, and the transfer variant continues for 100 epochs after restoration. Full training has not been rerun in the documented environment.

`03_repeat_measurement_experiment.ipynb` references earlier `retake` and `wastewater` HDF5 data/label pairs that are absent from the provided archive. `04_transfer_learning_experiment.ipynb` references a `retake2` data/label pair that is also absent. Later `retake3` and `wastewater2` files exist, but substituting them would change the experiment; no substitution was made. The variants are retained privately for provenance and require their intended missing data before a full run.

The augmentation cell in preprocessing reads repeat-measurement version 3 while its HDF5 output names contain `retake2_with_aug` and its CSV names contain `retake3_with_aug`. These naming inconsistencies are preserved in the normalized artifact names rather than silently reinterpreted.

## Scientific interpretation

- Some training examples are constructed by combining metal-specific spectral windows with condition-dependent backgrounds. Counts of augmented pairs are not counts of independent experimental acquisitions.
- Preprocessing uses both measured sequence tables to construct saturation masks. The experimental split and these shared preprocessing choices must be considered when interpreting generalization.
- The code remaps some test condition identifiers. This is an experimental grouping convention, not a new measured concentration.
- Several plotting scripts contain manually entered summary arrays. They can reproduce those figures but do not recompute a statistically independent model evaluation.
- The wastewater spike-recovery computation divides the before/after difference by **2.5**, although original comments described a 3 ppm spike. The computation is unchanged. Resolve this experimental annotation before presenting the values as validated spike recoveries.
- Calibration rejects nonfinite or zero reference intensities and a noninvertible calibration slope. Response correction checks equal wavelength grids and standard-row counts, while corresponding row order remains an experimental requirement. Corrected CSV values use an object array to avoid silent fixed-width string truncation.
- The supplied implementation does not establish a universal sub-10% error or end-to-end latency result. Such claims require the corresponding dataset, protocol, measurement conditions, and independent evaluation.

## Implementation and validation

Input paths are configurable, and each execution writes to a separate run directory. Area normalization writes a new table rather than replacing the measured spectra. Importing an analysis script does not launch a file dialog or start a calculation.

The numerical arrays, layers, loss equations, spectral windows, sample selection, and training settings are unchanged. The main model definitions have been extracted into `src/spectral_gan.py` and imported by the training notebook, providing one implementation for training and smoke checks. Provider names in wastewater figures use neutral sample identifiers. Notebook outputs are omitted from version control; executed originals remain in the research archive.

Validation covers Python syntax, notebook structure, example-data integrity, and standalone figure generation with a noninteractive backend. It additionally checks generator/discriminator construction, one finite optimizer step, a weight round trip, and analytically known calibration cases. These bounded checks do not reproduce full training or validate research accuracy.
