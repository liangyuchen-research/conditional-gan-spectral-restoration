"""Align spectral shifts and estimate wavelength-dependent response correction."""


def run():
    from pathlib import Path
    import argparse
    from project_paths import create_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, help="Original standard CSV")
    parser.add_argument("--new", type=Path, help="New standard CSV")
    parser.add_argument("--unknown", type=Path, help="Unknown sample CSV")
    args = parser.parse_args()
    for name in ("old", "new", "unknown"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    DATA_DIR, OUTPUT_DIR = create_run("correct_spectral_response")

    import numpy as np
    from scipy.signal import savgol_filter
    import tkinter as tk
    from tkinter import filedialog
    import matplotlib.pyplot as plt
    import csv

    # ==============================

    # ==============================
    COL_START = 11
    COL_END_INCLUSIVE = 1948 + 11
    INTENSITY_MAX = 60000
    SG_WINDOW = 11
    SG_POLY = 2
    MAX_SHIFT = 10

    # ==============================

    # ==============================
    def ask_csv(title):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title=title, filetypes=[("CSV files", "*.csv")])
        root.destroy()
        if not path:
            raise FileNotFoundError(f"No file selected: {title}")
        print(f"[INFO] Selected {title}: {path}")
        return path

    def load_full_csv_as_str(path, title):
        """Read the CSV as strings to preserve all header values."""
        arr_str = np.genfromtxt(path, delimiter=",", dtype=str)
        if arr_str.ndim != 2 or arr_str.shape[0] < 2:
            raise ValueError(f"{title} must be a 2D table with at least two rows.")
        return arr_str

    def to_float_block(arr_str, title):
        """
        Return (wavelength_1d_float, data_2d_float):
        - Select COL_START through COL_END_INCLUSIVE.
        - Stop the header scan at the first nonnumeric field or 'Pb-368nm_slope'.
        - Clip intensities to INTENSITY_MAX.
        """
        end_exclusive = COL_END_INCLUSIVE + 1
        if arr_str.shape[1] <= COL_END_INCLUSIVE:
            raise ValueError(f"{title} needs at least {COL_END_INCLUSIVE+1} columns.")

        header_slice = arr_str[0, COL_START:end_exclusive]
        valid_end = len(header_slice)

        for j, token in enumerate(header_slice):
            t = (token or "").strip()
            if t.lower() == "pb-368nm_slope".lower():
                valid_end = j
                break
            try:
                float(t)
            except ValueError:
                valid_end = j
                break

        if valid_end == 0:
            raise ValueError(f"{title} has no numeric columns in the selected wavelength block.")

        use_slice = slice(COL_START, COL_START + valid_end)
        wl_tokens = arr_str[0, use_slice]

        wl = np.empty(valid_end, dtype=float)
        for i, w in enumerate(wl_tokens):
            try:
                wl[i] = float((w or "").strip())
            except ValueError:
                wl[i] = np.nan

        data = arr_str[1:, use_slice].astype(float)
        data = np.clip(data, None, INTENSITY_MAX)
        return wl, data

    def ensure_sg_params(L, win, poly):
        win = min(win, L)
        if win % 2 == 0:
            win = max(1, win - 1)
        if win < 3:
            win = 3 if L >= 3 else 1
        poly = min(poly, max(0, win - 1))
        return win, poly

    def write_csv_preserve_first_row(arr_str, out_path):
        """Write all rows with csv.writer, preserving the original header."""
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in arr_str:
                writer.writerow(row)
        print(f"[SAVE] {out_path}")

    # ==============================

    # ==============================
    def cosine_sim(a, b, eps=1e-12):
        """Compute cosine similarity between flattened vectors."""
        a = a.ravel().astype(float)
        b = b.ravel().astype(float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
        return float(np.dot(a, b) / denom)

    def best_global_shift(ref_mean, mov_mean, max_shift):
        """
        Find the global lag in [-max_shift, max_shift] maximizing cosine similarity.
        ref_mean, mov_mean: shape (L,)
        Compute similarity only over overlapping regions.
        A positive returned lag compares later indices in mov to earlier indices in ref.
        """
        L = ref_mean.shape[0]
        best_k = 0
        best_score = -1.0

        for k in range(-max_shift, max_shift + 1):
            if k >= 0:

                a = ref_mean[0 : L - k]
                b = mov_mean[k:L]
            else:

                kk = -k
                a = ref_mean[kk:L]
                b = mov_mean[0 : L - kk]
            if a.size < 5:
                continue
            score = cosine_sim(a, b)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k, best_score

    def shift_block_with_edge_fill(block, k):
        """
        Shift a 2D array (samples x L):
        - k > 0: shift right and fill the left edge using column 0.
        - k < 0: shift left and fill the right edge using the last column.
        """
        if k == 0:
            return block.copy()
        L = block.shape[1]
        B = block.copy()
        if k > 0:
            B[:, k:] = block[:, :-k]
            B[:, :k] = block[:, [0]]
        else:
            kk = -k
            B[:, : L - kk] = block[:, kk:]
            B[:, L - kk :] = block[:, [-1]]
        return B

    # ==============================

    # ==============================
    def main():

        old_path = args.old or ask_csv("Select the original standard CSV (wavelength header).")
        new_path = args.new or ask_csv("Select the new standard CSV (wavelength header).")
        unknown_path = args.unknown or ask_csv("Select the unknown-sample CSV (wavelength header).")

        old_full_str = load_full_csv_as_str(old_path, "original standard")
        new_full_str = load_full_csv_as_str(new_path, "new standard")
        unk_full_str = load_full_csv_as_str(unknown_path, "unknown sample")

        wl_old, old_std = to_float_block(old_full_str, "original standard")
        wl_new, new_std = to_float_block(new_full_str, "new standard")
        wl_unk, unk = to_float_block(unk_full_str, "unknown sample")

        if old_std.shape[1] != new_std.shape[1] or unk.shape[1] != new_std.shape[1]:
            raise ValueError(
                "Original, new, and unknown spectra have inconsistent wavelength counts."
            )
        if not (np.array_equal(wl_old, wl_new) and np.array_equal(wl_old, wl_unk)):
            raise ValueError(
                "Wavelength grids must match before estimating an index-based spectral shift."
            )
        if old_std.shape[0] != new_std.shape[0]:
            raise ValueError(
                "Standard tables must contain the same number of correspondingly ordered rows."
            )
        L = new_std.shape[1]
        num_points = min(old_std.shape[0], new_std.shape[0])
        print(f"[INFO] Regression samples = {num_points}; wavelength count L = {L}")

        ref_mean = np.nan_to_num(np.nanmean(old_std, axis=0))
        mov_mean = np.nan_to_num(np.nanmean(new_std, axis=0))
        k_star, score = best_global_shift(ref_mean, mov_mean, MAX_SHIFT)
        print(f"[INFO] Estimated spectral lag = {k_star}, cosine={score:.6f}")

        new_std_aligned = shift_block_with_edge_fill(new_std, -k_star)
        unk_aligned = shift_block_with_edge_fill(unk, -k_star)

        # -------------------------------------------------------------

        s_raw = np.ones(L)
        for i in range(L):
            x = old_std[:num_points, i]
            y = new_std_aligned[:num_points, i]

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size < 2:
                s_raw[i] = 1.0
                continue
            xx = np.dot(x, x)
            if xx == 0:
                s_raw[i] = 1.0
                continue
            s_raw[i] = np.dot(x, y) / xx

        win, poly = ensure_sg_params(L, SG_WINDOW, SG_POLY)
        s_smooth = savgol_filter(s_raw, win, poly) if win >= 3 else s_raw.copy()
        print(f"[INFO] Smoothed s(lambda) (window={win}, poly={poly}).")

        eps = 1e-12
        cal_unk_block = unk_aligned / (s_smooth + eps)
        cal_new_std_for_plot = new_std_aligned / (s_smooth + eps)

        old_sat_cols = np.any(old_std >= INTENSITY_MAX, axis=0)  # shape (L,)
        cal_unk_block[:, old_sat_cols] = INTENSITY_MAX
        cal_new_std_for_plot[:, old_sat_cols] = INTENSITY_MAX

        cal_unk_block = np.clip(cal_unk_block, None, INTENSITY_MAX)
        cal_new_std_for_plot = np.clip(cal_new_std_for_plot, None, INTENSITY_MAX)

        out_full_str = unk_full_str.astype(object)
        cal_unk_str = np.vectorize(lambda v: f"{v:.6f}")(cal_unk_block)

        used_width = cal_unk_block.shape[1]
        end_idx = COL_START + used_width
        out_full_str[1:, COL_START:end_idx] = cal_unk_str

        out_path = "calibrated_unknowns.csv"
        write_csv_preserve_first_row(out_full_str, out_path)
        print(f"[INFO] Saved corrected unknown samples: {out_path}")

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        idx = num_points - 1
        wl = wl_old
        plt.figure(figsize=(10, 4))
        plt.plot(wl, old_std[idx, :], label="Original standard (5 ppm)", linewidth=2)
        plt.plot(wl, new_std[idx, :], label="New standard (5 ppm, before alignment)", alpha=0.5)
        plt.plot(
            wl,
            new_std_aligned[idx, :],
            "--",
            label="New standard (5 ppm, after alignment)",
            linewidth=2,
        )
        plt.plot(
            wl,
            cal_new_std_for_plot[idx, :],
            ":",
            label="Corrected new standard (5 ppm)",
            linewidth=2,
        )
        plt.title("Calibration comparison on the original wavelength grid")
        plt.xlabel("Wavelength")
        plt.ylabel("Intensity")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    main()


if __name__ == "__main__":
    run()
