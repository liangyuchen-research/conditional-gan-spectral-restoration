# Conditional GAN Spectral Restoration for Metal Quantification

Research code for restoring plasma emission spectra affected by matrix interference and evaluating downstream Zn, Ni, and Cu quantification. The workflow combines paired-spectrum construction, a one-dimensional conditional GAN, wavelength-specific calibration, repeat-measurement experiments, and wastewater analysis.

## Method

The generator uses a convolutional encoder-decoder with skip connections to map an interfered spectrum to a restored spectrum. A conditional discriminator compares spectra in the context of the input. Training combines adversarial, cosine, feature-matching, and metal-window reconstruction losses with learned positive weights. The original normalization scale, wavelength windows, architecture, training settings, and numerical calculations have been retained.

```text
Measured spectra and acquisition metadata
    -> construct paired spectra and augment metal windows
    -> train a conditional GAN to restore spectra
    -> inspect restored Zn / Ni / Cu emission regions
    -> fit calibration curves and evaluate repeat measurements / wastewater
```

## Repository guide

| Location | Purpose |
| --- | --- |
| `notebooks/01_prepare_datasets.ipynb` | Pair measured spectra, construct augmentation, and export HDF5 datasets |
| `notebooks/02_train_conditional_gan.ipynb` | Main model definition, training, restoration, and figure export |
| `notebooks/03_repeat_measurement_experiment.ipynb` | Preserved experiment that incorporates repeat measurements |
| `notebooks/04_transfer_learning_experiment.ipynb` | Restore an earlier checkpoint and continue training |
| `scripts/calibrate_metal_concentrations.py` | Reference-wavelength normalization, background subtraction, and linear calibration |
| `scripts/correct_spectral_response.py` | Global spectral alignment and wavelength-dependent response correction |
| `scripts/normalize_by_area.py`, `normalize_by_h_beta.py` | Alternative spectrum normalization procedures |
| `scripts/plot_*.py` | Calibration, interference, repeatability, and wastewater result plots |
| `data/examples/` | Two compact tables containing original numeric observations |
| `data/external_artifacts.json` | Exact hashes and local layout for separately archived data and model artifacts |

## Setup

A Python 3.10 or 3.11 environment is a reasonable starting point for the legacy TensorFlow/Keras APIs used by the source. The dependency ranges are compatibility guidance, not a recovered environment lockfile.

```bash
python -m venv .venv
# Activate the environment using the command appropriate to your shell.
python -m pip install -r requirements.txt
python -m jupyter lab
```

TensorFlow GPU setup depends on the operating system and installed runtime. PyTorch appears only in optional random-seed initialization and is not required for the models. Interactive calibration and response-correction scripts use Tk file dialogs and need a graphical desktop.

## Data and execution

The complete working archive contains 4,689 files and approximately 1.70 GB of data, plots, checkpoints, and original source. It is retained separately from this repository. The public code package includes small numeric examples and a checksum manifest, rather than committing the full working archive.

If the private archive is available locally, restore the documented runtime inputs with:

```bash
python scripts/restore_local_data.py --archive-root "/path/to/private/raw-archive"
```

This copies only the artifacts listed in the manifest into `data/local/`, verifies SHA-256 checksums, and refuses to overwrite files with different content. It does not modify the archive. The manifest is an inventory, not a download endpoint.

Start Jupyter in the repository root. Each notebook creates a fresh directory in `results/` for generated files. Inputs are read from `data/local/` by default. Set `SPECTRAL_DATA_DIR` to use a different prepared input directory and `SPECTRAL_OUTPUT_DIR` to choose another output root. Preprocessing writes newly generated HDF5/CSV files to the run's `processed/` directory. To train from those regenerated files, set `SPECTRAL_DATA_DIR` to that preprocessing run directory, supplying the other required test/repeat input files under its `processed/` folder as needed.

Run analysis scripts explicitly, for example:

```bash
python scripts/calibrate_metal_concentrations.py
python scripts/plot_wastewater_spike_recovery.py
```

The measured-versus-predicted plot additionally needs a calibration result table. Set `SPECTRAL_PREDICTIONS` to the `prediction_results_recover.csv` created by the calibration script.

## Reproducibility status

This is an organized research snapshot. Code syntax, notebook structure, text cleanup, example-data integrity, and archive checksums were checked during preparation. Training, inference, and full scientific result reproduction were not performed. Notebook outputs were cleared in this copy; the original executed notebooks remain in the private archive.

The experimental variants reference some earlier dataset versions that are absent from the supplied archive. Several plots use manually recorded arrays rather than deriving results from model predictions. Consequently, this README makes no new accuracy or latency claims. See [reproducibility notes](docs/REPRODUCIBILITY.md) and [data documentation](docs/DATA.md) before interpreting or executing the experiments.

## Provenance and reuse

This repository was prepared from the author's research working files. File renaming and documentation changes are recorded in [the file mapping](docs/FILE_MAPPING.json). Original files were copied and verified before curation. No research data or original source was deleted. Wastewater-provider labels in one plot were replaced with neutral sample identifiers.

No license has been added because licensing terms were not present in the supplied source. Contact the repository owner about reuse and access to the complete research data.
