# GAN-Based Restoration of Plasma Spectra for Real-Time Heavy Metal Quantification

A conditional generative adversarial network for restoring plasma emission spectra affected by matrix interference. The workflow pairs measured spectra, learns a spectral correction, and evaluates Zn, Ni, and Cu concentrations through wavelength-specific calibration, repeat measurements, and wastewater analysis.

Developed at the Plasma Engineering Laboratory, National Taiwan University, with Prof. Cheng-Che Hsu. This repository contains the spectral-restoration component of the research. The companion [plasma spectroscopy repository](https://github.com/liangyuchen-research/plasma-spectroscopy-quantification) contains the direct concentration-prediction models and occlusion-based spectral interpretation experiments.

## Method

A convolutional encoder-decoder with skip connections maps an interfered spectrum to a restored spectrum. A conditional discriminator evaluates the restoration alongside its input. Training combines adversarial, cosine, feature-matching, and metal-window reconstruction losses with learned positive weights.

```text
Measured spectra and acquisition metadata
    -> paired-spectrum construction and metal-window augmentation
    -> conditional GAN training and spectral restoration
    -> Zn / Ni / Cu emission-region analysis
    -> concentration calibration and wastewater evaluation
```

## Repository guide

| Location | Purpose |
| --- | --- |
| `notebooks/01_prepare_datasets.ipynb` | Construct paired and augmented spectra and export HDF5 datasets |
| `notebooks/02_train_conditional_gan.ipynb` | Train and evaluate the conditional GAN |
| `src/spectral_gan.py` | Generator, discriminator, and original research losses |
| `scripts/check_model.py`, `check_calibration.py` | Bounded model and calibration checks |
| `scripts/calibrate_metal_concentrations.py` | Reference normalization, background subtraction, and linear calibration |
| `scripts/correct_spectral_response.py` | Spectral alignment and wavelength-dependent response correction |
| `scripts/normalize_by_area.py`, `normalize_by_h_beta.py` | Alternative normalization procedures |
| `scripts/plot_*.py` | Calibration, matrix interference, repeatability, and wastewater plots |
| `data/examples/` | Compact measured-spectrum tables for format inspection |
| `data/external_artifacts.json` | Checksums and expected layout of external inputs |

## Setup

Use Python 3.10–3.12. Create a virtual environment and activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux.

```bash
python -m venv .venv
# Activate the environment before continuing.
python -m pip install -r requirements.txt
python -m jupyter lab
```

The model uses TensorFlow 2.16.2 with legacy Keras 2.16. Its module selects the legacy runtime before importing TensorFlow. PyTorch is optional and used only when available for seed initialization. Calibration accepts explicit CSV paths. Response correction accepts paths or opens file dialogs when arguments are omitted.

## Data and execution

The repository includes two small measured-spectrum tables. Full training datasets and model checkpoints are stored separately. The examples support format inspection but are insufficient for model training. See [data formats and artifact requirements](docs/DATA.md).

If the matching research archive is available locally, restore the required files with:

```bash
python scripts/restore_local_data.py --archive-root "/path/to/research-archive"
```

The utility verifies checksums and copies listed inputs into `data/local/`. It refuses to overwrite a file with different content. The artifact manifest documents the expected files; it does not provide a download service.

Start Jupyter from the repository root and run preprocessing before training. Each notebook creates a fresh directory under `results/`. Set `SPECTRAL_DATA_DIR` to select another input directory or `SPECTRAL_OUTPUT_DIR` to change the output root. Preprocessing exports HDF5 and CSV files under its run's `processed/` directory. To train from regenerated files, point `SPECTRAL_DATA_DIR` to that run directory and supply the required test and repeat-measurement inputs under `processed/`.

Analysis scripts run independently:

```bash
python scripts/calibrate_metal_concentrations.py training.csv testing.csv --output results/calibration-run
python scripts/plot_wastewater_spike_recovery.py
```

For the measured-versus-predicted plot, set `SPECTRAL_PREDICTIONS` to the calibration script's `prediction_results_recover.csv` output.

## Reproducibility

The main model, preprocessing, and analysis implementations are included. Earlier repeat-measurement and transfer variants require acquisition versions absent from the supplied archive. Those variants are preserved privately; the public notebooks contain the complete preprocessing and main training workflow. Some plotting scripts use recorded summary arrays rather than recomputing predictions.

Python syntax, notebook structure, example-data integrity, and standalone plotting have been checked. The model passes a finite forward/optimizer step and a weight-save/reload check on synthetic spectra. The archived generator weights also load and produce finite predictions, and the main notebook loads its ten required real input files and builds the model before training. Calibration is checked against known slopes and invalid reference intensities.

```bash
python scripts/check_model.py
python scripts/check_calibration.py
```

Full training and independent scientific performance have not been reproduced. The [reproduction notes](docs/REPRODUCIBILITY.md) describe input requirements, experimental assumptions, and unresolved calibration annotations.

## Data and code use

The [file mapping](docs/FILE_MAPPING.json) records the relationship between the research files and repository paths. Numerical observations, model architecture, losses, and training settings are retained. Original measurements, executed notebooks, figures, and checkpoints remain in the research archive.

No software or dataset license is included. Contact the repository owner regarding reuse or access to the full research data.
