"""Generate the README figures for the spectral-restoration repository.

Two sources are used, both already in the repository:

* the recorded concentration readouts embedded in ``plot_correction_error.py``
  (archived research results, not re-run here);
* the two example mean-spectrum tables under ``data/examples/``.

    python scripts/make_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
METAL_COLORS = {"Cu": OKABE_ITO[0], "Ni": OKABE_ITO[1], "Zn": OKABE_ITO[2]}
METAL_WINDOWS_NM = {"Zn": (212.129, 216.777), "Ni": (229.563, 234.734), "Cu": (326.669, 330.016)}

# Recorded readouts (ppm) for 0-5 ppm standards before and after restoration.
# Copied from scripts/plot_correction_error.py.
TRUTH = np.array([0, 1, 2, 3, 4, 5], dtype=float)
BEFORE = {
    "Zn": np.array([-0.037, 0.769, 1.365, 1.871, 2.484, 2.801]),
    "Ni": np.array([-0.238, 0.329, 1.096, 1.908, 2.537, 3.126]),
    "Cu": np.array([-1.438, -1.011, -0.762, -0.311, 0.081, 0.393]),
}
AFTER = {
    "Zn": np.array([-0.038, 0.849, 1.986, 3.042, 4.280, 5.035]),
    "Ni": np.array([0.063, 0.820, 1.765, 2.915, 3.913, 5.091]),
    "Cu": np.array([-0.065, 0.959, 2.139, 3.362, 4.425, 5.576]),
}

def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.linewidth": 0.7, "lines.linewidth": 1.1, "lines.markersize": 4,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote docs/figures/{name}.png")


def figure_restoration_results() -> None:
    fig, ax = plt.subplots(figsize=(3.9, 3.9), constrained_layout=True)
    lim = (-1.8, 6.2)
    ax.plot(lim, lim, ls="--", color="0.4", lw=0.8)
    ax.fill_between(lim, [0.9 * v for v in lim], [1.1 * v for v in lim], color="0.88", lw=0, label="±10 %")
    for metal, marker in zip(("Cu", "Ni", "Zn"), ("o", "s", "^")):
        ax.plot(TRUTH, BEFORE[metal], marker, mfc="none", mec=METAL_COLORS[metal], mew=0.9, ms=5)
        ax.plot(TRUTH, AFTER[metal], marker, color=METAL_COLORS[metal], mec="white", mew=0.4, ms=5.5,
                label=f"{metal}")
        for t, b, a in zip(TRUTH, BEFORE[metal], AFTER[metal]):
            ax.annotate("", xy=(t, a), xytext=(t, b),
                        arrowprops={"arrowstyle": "-|>", "color": METAL_COLORS[metal], "lw": 0.5,
                                    "alpha": 0.45, "shrinkA": 3, "shrinkB": 3})
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Reference concentration (ppm)")
    ax.set_ylabel("Quantified concentration (ppm)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.74), fontsize=7, ncol=2, columnspacing=0.9, handletextpad=0.4)
    errors = {m: np.mean(np.abs(AFTER[m][1:] - TRUTH[1:]) / TRUTH[1:]) * 100 for m in ("Cu", "Ni", "Zn")}
    ax.text(0.07, 0.88, "open = before, filled = after restoration\nmean |error| after, 1–5 ppm:\n"
            + ", ".join(f"{m} {e:.0f} %" for m, e in errors.items()),
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    save(fig, "restoration_results")


def figure_example_spectra() -> None:
    std = pd.read_csv(ROOT / "data/examples/standard_spectra.csv")
    ww = pd.read_csv(ROOT / "data/examples/wastewater_spectra.csv")
    spectral = [c for c in std.columns if c.replace(".", "", 1).isdigit()]
    wl = np.array([float(c) for c in spectral])
    fig, ax = plt.subplots(figsize=(7.4, 2.7), constrained_layout=True)
    ax.plot(wl, std[spectral].mean(axis=0) / 1000, color=OKABE_ITO[0], lw=0.8, label="standard solution (mean of 6)")
    ax.plot(wl, ww[spectral].mean(axis=0) / 1000, color=OKABE_ITO[1], lw=0.8, label="wastewater (mean of 6)")
    for metal, (lo, hi) in METAL_WINDOWS_NM.items():
        ax.axvspan(lo, hi, color=METAL_COLORS[metal], alpha=0.18, lw=0)
        ax.text((lo + hi) / 2, 0.97, f"{metal}\nwindow", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7, color=METAL_COLORS[metal])
    ax.set_xlim(wl.min(), wl.max())
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (10$^3$ counts)")
    ax.legend(loc="upper right")
    save(fig, "example_spectra")


def main() -> None:
    style()
    figure_restoration_results()
    figure_example_spectra()


if __name__ == "__main__":
    main()
