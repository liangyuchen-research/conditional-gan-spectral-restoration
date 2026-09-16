"""Verify known calibration slopes and reject invalid reference intensities."""

import unittest

import numpy as np
import pandas as pd

from calibrate_metal_concentrations import calibrate, process_one_metal_block, predict_C_from_I


class CalibrationChecks(unittest.TestCase):
    def test_known_calibration(self):
        concentration = np.arange(4, dtype=float)
        metadata = {"Cr": np.ones(4), "Zn": concentration, "Ni": concentration, "Cu": concentration}
        metadata.update({f"meta_{i}": np.zeros(4) for i in range(7)})
        wavelengths = [
            211.757,
            212.501,
            214.547,
            216.777,
            230.857,
            232.704,
            234.180,
            235.287,
            325.610,
            328.079,
            328.960,
        ]
        metadata.update({f"{value:.3f}": np.ones(4) for value in wavelengths})
        frame = pd.DataFrame(metadata)
        for peak in ["214.547", "232.704", "328.079"]:
            frame[peak] = 1 + 3 * concentration
        predictions, models = calibrate(frame, frame)
        for metal in ["Zn", "Ni", "Cu"]:
            self.assertAlmostEqual(models[metal]["slope"], 3)
            np.testing.assert_allclose(predictions[f"{metal}_predicted"], concentration, atol=1e-12)
        frame["211.757"] = 0
        with self.assertRaises(ValueError):
            process_one_metal_block(frame, "Zn")
        with self.assertRaises(ValueError):
            predict_C_from_I(pd.Series([1.0]), 0, 1)


if __name__ == "__main__":
    unittest.main()
