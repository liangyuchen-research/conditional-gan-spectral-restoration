# Data formats and local artifacts

## Measured-spectrum CSV files

The inspected mean-spectrum tables contain 1,979 columns. Columns 1-11 contain the exported row index and acquisition/sample metadata:

```text
index, Voltage, Ontime, Offtime, Cycle, Conductivity, Cu, Ni, Pb, Zn, Cr
```

Numeric wavelength headers follow, starting at 199.820 nm. Derived calibration columns follow the wavelength block. Preserve column order: preprocessing selects spectral columns by position and reads concentration/condition columns by name.

The original scripts use `Cr` as a condition selector and sometimes remap it to experimental group IDs (16, 17, and 18). Do not assume every `Cr` value in intermediate tables is a chromium concentration. The source does not supply a complete independent acquisition metadata dictionary, so timing and conductivity units are not inferred here.

`data/examples/standard_spectra.csv` contains the header and first six data rows of the sequences 1-15 mean table. `data/examples/wastewater_spectra.csv` contains all six data rows of the wastewater mean table. Their numeric and header values are unchanged. These are inspection examples, not sufficient training datasets.

## Paired HDF5 datasets

Input files use datasets `X_with_meta` and `X_columns`; target files use `Y_with_meta` and `Y_columns`. The first four columns are:

```text
Cr, Cu, Ni, Zn
```

Subsequent columns are spectral intensities, with byte-string wavelength headers. The main notebook's archived outputs report 2,376 training pairs with 1,873 spectral points each, 60 combined test samples, and 18 samples each in the repeat and corrected datasets. These counts come from archived output, not a new training run.

The model notebook divides spectral intensities by 60,000. The metal windows are Zn 212.129-216.777 nm, Ni 229.563-234.734 nm, and Cu 326.669-330.016 nm. Preprocessing also interpolates selected Pb regions and applies wavelength-dependent saturation rules. Review the code before applying it to a different instrument grid.

## Separate artifacts

`external_artifacts.json` lists the archived datasets, mean-spectrum CSVs, correction table, and checkpoints used to assemble local inputs. Each entry provides an archive-relative path, normalized destination, byte count, and SHA-256 checksum. Original names appear only as relative provenance identifiers; no personal machine path is included.

Raw acquisition TXT files, thousands of generated plots, screen recordings, and large prediction CSVs are preserved in the private archive. They are not needed to inspect the implementation and have not been committed here. HDF5 weights and TensorFlow checkpoint shards are also kept outside the repository. Model files should be loaded only from a trusted archive.

## Derived prediction tables

`calibrate_metal_concentrations.py` exports:

```text
Cr, Zn, Ni, Cu, Zn_predicted, Ni_predicted, Cu_predicted
```

`plot_measured_vs_predicted.py` reads that generated table through `SPECTRAL_PREDICTIONS`. A matching table was not present in the supplied research folder; the plotting script cannot produce its result until one is generated or supplied.
