"""Normalize spectra using integrated area up to 276.241 nm."""

def run():
    from pathlib import Path
    import os
    from project_paths import create_run
    DATA_DIR, OUTPUT_DIR = create_run('normalize_by_area')

    import pandas as pd
    import numpy as np


    df = pd.read_csv(str(DATA_DIR / 'measurements/wastewater2/mean.csv'))


    wavelength_start = 11
    wavelength_end = 11 + 1948  # 1959
    wavelength_cols = df.columns[wavelength_start:wavelength_end]

    print("Step 1: Replace spectral intensities above 60000 with 80000.")

    for col in wavelength_cols:
        df.loc[df[col] > 60000, col] = 80000


    wavelength_values = [float(col) for col in wavelength_cols]
    target_wavelength = 276.241


    cols_up_to_276 = [col for col, val in zip(wavelength_cols, wavelength_values) if val <= target_wavelength]

    print(f"\nWavelength points up to 276.241 nm: {len(cols_up_to_276)}")


    cr_1_rows = df[df['Cr'] == 1]


    def calculate_area(row, cols):

        wavelengths = np.array([float(col) for col in cols])
        intensities = row[cols].values

        area = np.trapz(intensities, wavelengths)
        return area


    areas_cr1 = cr_1_rows.apply(lambda row: calculate_area(row, cols_up_to_276), axis=1)
    mean_area = areas_cr1.mean()
    print(f"Mean integrated area up to 276.241 nm for Cr=1: {mean_area}")

    print("\nStep 2: Normalize spectra.")

    def normalize_row(row):
        area = calculate_area(row, cols_up_to_276)
        return row[wavelength_cols] / area if area != 0 else row[wavelength_cols]


    normalized_spectra = df.apply(normalize_row, axis=1)


    normalized_spectra = normalized_spectra * mean_area


    for col in wavelength_cols:
        df[col] = normalized_spectra[col]

    print("\nStep 3: Clip normalized intensities above 60000 to 60000.")

    for col in wavelength_cols:
        df.loc[df[col] > 60000, col] = 60000


    final_areas = df[df['Cr'] == 1].apply(lambda row: calculate_area(row, cols_up_to_276), axis=1)
    final_mean_area = final_areas.mean()
    print(f"\nMean integrated area after normalization for Cr=1: {final_mean_area}")


    output_file = str(OUTPUT_DIR / 'area_normalized_spectra.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run()
