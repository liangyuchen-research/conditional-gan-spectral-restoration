"""Fit wavelength-normalized Zn, Cu, and Ni calibration models from CSV tables."""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def is_number_str(s):
    try:
        float(str(s))
        return True
    except Exception:
        return False


def canon_wl(s):
    v = float(str(s))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def build_wl_map(headers):
    m = {}
    for c in headers:
        k = canon_wl(c)
        if k in m and m[k] != c:
            raise ValueError(
                f"Equivalent wavelength columns: {m[k]} and {c} (key={k}). Retain one column."
            )
        m[k] = c
    return m


def require_wl(target_wl_str, wl_map):
    k = canon_wl(target_wl_str)
    if k not in wl_map:
        raise KeyError(f"Wavelength column not found: {target_wl_str} (canonical key {k})")
    return wl_map[k]


metal_peaks = {
    "Zn": "214.547",
    "Cu": "328.079",
    "Ni": "232.704",
}

metal_background = {
    "Zn": ["212.501", "216.777"],
    "Ni": ["230.857", "234.180"],
    "Cu": ["325.610", "328.960"],
}

metal_normalize_candidates = {
    "Zn": ["211.757"],
    "Cu": ["328.960"],
    "Ni": ["235.287"],
}

SPECTRAL_START = 11


def find_first_present_wl(candidates, wl_map):
    last_err = None
    for wl in candidates:
        try:
            return require_wl(wl, wl_map)
        except KeyError as e:
            last_err = e
            continue
    raise last_err if last_err else KeyError("No candidate wavelength is present.")


def process_one_metal_block(df, metal):
    """
    Process one metal:
      1) Normalize by a finite, nonzero reference-wavelength intensity.
      2) Subtract a two-point linear background and return the corrected peak intensity.
    """
    col = list(df.columns)
    spectral_cols = [c for c in col[SPECTRAL_START:] if is_number_str(c)]
    if not spectral_cols:
        raise ValueError("No numeric wavelength columns found from column 12 onward.")
    spec_idx = [col.index(c) for c in spectral_cols]
    mat = df.iloc[:, spec_idx].astype(float).to_numpy()

    wl_map = build_wl_map(spectral_cols)

    norm_col = find_first_present_wl(metal_normalize_candidates[metal], wl_map)
    norm_idx = spectral_cols.index(norm_col)
    norm_vec = mat[:, [norm_idx]]
    if not np.isfinite(mat).all() or np.any(norm_vec == 0):
        raise ValueError(f"{metal}: spectra must be finite and reference intensities nonzero.")
    spectrum = mat / norm_vec

    out = pd.DataFrame(spectrum, columns=spectral_cols)
    wl_map_local = build_wl_map(out.columns)

    bg1 = require_wl(metal_background[metal][0], wl_map_local)
    bg2 = require_wl(metal_background[metal][1], wl_map_local)
    pk = require_wl(metal_peaks[metal], wl_map_local)

    x1, x2 = float(bg1), float(bg2)
    y1 = out[bg1].to_numpy()
    y2 = out[bg2].to_numpy()
    slope_bg = (y1 - y2) / (x1 - x2)
    intercept_bg = y2 - slope_bg * x2
    bg = slope_bg * float(pk) + intercept_bg
    corrected = out[pk].to_numpy() - bg

    return pd.Series(corrected, name=f"{metal}_I_corrected")


def build_model_I_of_C(I_series, C_series):
    """Fit I = m*C + b using finite samples and return (m, b, r2)."""
    I = I_series.to_numpy().reshape(-1, 1)
    C = C_series.to_numpy().reshape(-1, 1)
    mask = np.isfinite(I.ravel()) & np.isfinite(C.ravel())
    if mask.sum() < 2 or np.unique(C[mask]).size < 2:
        raise ValueError("Calibration requires finite samples at two or more concentration levels.")
    I = I[mask].reshape(-1, 1)
    C = C[mask].reshape(-1, 1)
    mdl = LinearRegression().fit(C, I)  # I = m*C + b
    pred = mdl.predict(C)
    m = float(mdl.coef_[0][0])
    b = float(mdl.intercept_[0])
    r2 = r2_score(I, pred)
    return m, b, r2


def predict_C_from_I(I_series, m, b):
    """Compute C_hat = (I - b)/m, replacing nonfinite values with NaN."""
    I = I_series.to_numpy().reshape(-1, 1)
    if not np.isfinite(m) or m == 0:
        raise ValueError("Cannot invert a zero or nonfinite calibration slope.")
    C_hat = (I - b) / m
    C_hat = np.where(np.isfinite(C_hat), C_hat, np.nan)
    return C_hat.ravel()


def calibrate(training, testing):
    models = {}
    result = testing.reindex(columns=["Cr", "Zn", "Ni", "Cu"]).copy()
    for metal in ["Zn", "Ni", "Cu"]:
        m, b, r2 = build_model_I_of_C(process_one_metal_block(training, metal), training[metal])
        result[f"{metal}_predicted"] = predict_C_from_I(
            process_one_metal_block(testing, metal), m, b
        )
        models[metal] = {"slope": m, "intercept": b, "training_r2": float(r2)}
    return result, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_csv", type=Path)
    parser.add_argument("testing_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    result, models = calibrate(pd.read_csv(args.training_csv), pd.read_csv(args.testing_csv))
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output / "prediction_results_recover.csv", index=False)
    (args.output / "calibration.json").write_text(
        json.dumps(models, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(models, indent=2))


if __name__ == "__main__":
    main()
