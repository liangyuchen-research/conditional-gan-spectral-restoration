"""Normalize spectra using the reference intensity at 486.648 nm."""

def run():
    from pathlib import Path
    import os
    from project_paths import create_run
    DATA_DIR, OUTPUT_DIR = create_run('normalize_by_h_beta')

    import pandas as pd
    import numpy as np


    df = pd.read_csv(str(DATA_DIR / 'measurements/retake2/mean.csv'))


    wavelength_start = 11
    wavelength_end = 11 + 1948  # 1959
    wavelength_cols = df.columns[wavelength_start:wavelength_end]

    print("Step 1: Replace spectral intensities above 60000 with 80000.")

    for col in wavelength_cols:
        df.loc[df[col] > 60000, col] = 80000


    cr_1_rows = df[df['Cr'] == 1]
    mean_486_648 = cr_1_rows['486.648'].mean()
    print(f"\nMean intensity at 486.648 nm for Cr=1: {mean_486_648}")

    print("\nStep 2: Normalize spectra.")

    def normalize_row(row):
        factor = row['486.648']
        return row[wavelength_cols] / factor if factor != 0 else row[wavelength_cols]


    normalized_spectra = df.apply(normalize_row, axis=1)


    normalized_spectra = normalized_spectra * mean_486_648


    for col in wavelength_cols:
        df[col] = normalized_spectra[col]

    print("\nStep 3: Clip normalized intensities above 60000 to 60000.")

    for col in wavelength_cols:
        df.loc[df[col] > 60000, col] = 60000


    final_cr_1 = df[df['Cr'] == 1]
    final_mean = final_cr_1['486.648'].mean()
    print(f"\nMean intensity at 486.648 nm after normalization for Cr=1: {final_mean}")


    output_file = str(OUTPUT_DIR / 'mean_normalized.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run()
