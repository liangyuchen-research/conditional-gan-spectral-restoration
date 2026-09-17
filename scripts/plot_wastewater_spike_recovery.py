"""Plot archived wastewater spike recovery values with anonymized sample labels."""


def run():
    from project_paths import create_run

    DATA_DIR, OUTPUT_DIR = create_run("plot_wastewater_spike_recovery")

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    from matplotlib.ticker import FuncFormatter

    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    raw = {
        "Zn": [(0.3116, 1.5518), (0.3095, 1.5475), (0.3791, 1.4195)],
        "Ni": [(0.3879, 2.4182), (0.7230, 2.0059), (0.3483, 2.0695)],
        "Cu": [(0.5301, 1.5790), (0.4617, 1.4303), (0.6240, 1.5289)],
    }

    labels = ["Wastewater A", "Wastewater B", "Wastewater C"]
    x = np.arange(len(labels))
    bar_width = 0.25

    def compute_spike_recovery(pairs):
        return [(pair[1] - pair[0]) / 2.5 for pair in pairs]

    vals_Zn = compute_spike_recovery(raw["Zn"])
    vals_Ni = compute_spike_recovery(raw["Ni"])
    vals_Cu = compute_spike_recovery(raw["Cu"])

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - bar_width, vals_Zn, bar_width, color="skyblue", label="Zn")

    bars2 = ax.bar(
        x, vals_Ni, bar_width, facecolor="orange", hatch="xx", edgecolor="gray", label="Ni"
    )

    bars3 = ax.bar(
        x + bar_width,
        vals_Cu,
        bar_width,
        facecolor="plum",
        hatch="///",
        edgecolor="gray",
        label="Cu",
    )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)

    ax.set_ylabel("Spike Recovery (%)", fontsize=14)
    ax.set_xlabel("", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.set_ylim(0, 1.3)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    ax.legend(loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
