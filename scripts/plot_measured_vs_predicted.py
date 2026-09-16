"""Compare measured concentrations with calibration predictions."""


def run():
    from pathlib import Path
    import os
    from project_paths import create_run

    configured_predictions = os.environ.get("SPECTRAL_PREDICTIONS")
    configured_predictions = (
        Path(configured_predictions).expanduser().resolve() if configured_predictions else None
    )
    DATA_DIR, OUTPUT_DIR = create_run("plot_measured_vs_predicted")

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    file_path = str(
        configured_predictions or DATA_DIR / "predictions/prediction_results_recover.csv"
    )
    df = pd.read_csv(file_path)

    conditions = [{"Cr": 14, "Zn": i, "Ni": i, "Cu": i} for i in range(6)]

    mask = pd.concat(
        [
            (df["Cr"] == cond["Cr"])
            & (df["Zn"] == cond["Zn"])
            & (df["Ni"] == cond["Ni"])
            & (df["Cu"] == cond["Cu"])
            for cond in conditions
        ],
        axis=1,
    ).any(axis=1)
    filtered_df = df[mask]
    if filtered_df.empty:
        raise ValueError("No prediction rows match the Cr=14 concentration comparison.")

    metals = ["Cu", "Ni", "Zn"]
    predicted_suffixes = {"Cu": "Cu_predicted", "Ni": "Ni_predicted", "Zn": "Zn_predicted"}

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"Cu": "o", "Ni": "s", "Zn": "D"}
    colors = {"Cu": "tab:red", "Ni": "tab:green", "Zn": "tab:blue"}

    for metal in metals:
        ax.scatter(
            filtered_df[metal],
            filtered_df[predicted_suffixes[metal]],
            label=metal,
            marker=markers[metal],
            color=colors[metal],
            s=80,
        )

    all_real = pd.concat([filtered_df[m] for m in metals])
    all_pred = pd.concat([filtered_df[predicted_suffixes[m]] for m in metals])
    min_val = min(min(all_real.min(), all_pred.min()), -0.3)
    max_val = max(all_real.max(), all_pred.max()) * 1.05

    x_vals = np.linspace(0, max_val, 100)
    ax.plot(x_vals, x_vals, "k-", linewidth=1.5)
    ax.plot(x_vals, x_vals * 1.2, "r--", linewidth=1.5)
    ax.plot(x_vals, x_vals * 0.8, "r--", linewidth=1.5)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Real Concentration (ppm)", fontsize=16)
    ax.set_ylabel("Predicted Concentration (ppm)", fontsize=16)
    ax.legend(fontsize=16)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
