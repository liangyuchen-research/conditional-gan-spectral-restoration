"""Plot the archived sodium/calcium interference error summary."""


def run():
    from pathlib import Path
    import os
    from project_paths import create_run

    DATA_DIR, OUTPUT_DIR = create_run("plot_matrix_interference_error")

    import matplotlib.pyplot as plt
    import numpy as np

    conditions = [
        "Without interference",
        "250ppm Na",
        "500ppm Na",
        "250ppm Ca",
        "500ppm Ca",
        "250ppm Na+Ca",
        "500ppm Na+Ca",
    ]

    zn_values = [
        4.860627495,
        4.064246934,
        5.029553635,
        4.450048656,
        4.504845348,
        5.900828616,
        6.939106947,
    ]
    ni_values = [
        5.076000503,
        3.353995498,
        2.897807263,
        4.10843136,
        3.749820279,
        2.835476918,
        2.77637045,
    ]
    cu_values = [
        4.952120002,
        3.711425686,
        3.439542702,
        4.44220168,
        4.189729481,
        4.012834062,
        4.297668772,
    ]

    cu_error = [(v - 5) / 5 * 100 for v in cu_values]
    ni_error = [(v - 5) / 5 * 100 for v in ni_values]
    zn_error = [(v - 5) / 5 * 100 for v in zn_values]

    x = np.arange(len(conditions))
    bar_width = 0.3

    fig, ax = plt.subplots(figsize=(12, 8))

    bars1 = ax.bar(x - bar_width, cu_error, bar_width, color="skyblue", label="Cu")
    bars2 = ax.bar(
        x, ni_error, bar_width, color="orange", edgecolor="gray", hatch="////", label="Ni"
    )
    bars3 = ax.bar(
        x + bar_width,
        zn_error,
        bar_width,
        color="thistle",
        edgecolor="black",
        hatch="\\\\\\\\",
        label="Zn",
    )

    ax.set_ylabel("Relative Error (%)", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20)

    offset = 0.2
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, fontsize=16)

    ax.set_ylim(-50, 50)
    ax.tick_params(axis="y", labelsize=16)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(fontsize=18)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
