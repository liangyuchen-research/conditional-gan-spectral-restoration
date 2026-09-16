"""Plot the archived correction-method relative-error comparison."""

def run():
    from pathlib import Path
    import os
    from project_paths import create_run
    DATA_DIR, OUTPUT_DIR = create_run('plot_correction_error')

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches


    truth = np.array([1, 2, 3, 4, 5], dtype=float)


    zn_before = np.array([-0.037, 0.769, 1.365, 1.871, 2.484, 2.801])
    zn_after  = np.array([-0.038, 0.849, 1.986, 3.042, 4.280, 5.035])

    ni_before = np.array([-0.238, 0.329, 1.096, 1.908, 2.537, 3.126])
    ni_after  = np.array([ 0.063, 0.820, 1.765, 2.915, 3.913, 5.091])

    cu_before = np.array([-1.438, -1.011, -0.762, -0.311, 0.081, 0.393])
    cu_after  = np.array([-0.065,  0.959,  2.139,  3.362, 4.425, 5.576])


    def relative_errors(pred_vals, truth_vals):
        pred = pred_vals[1:6]
        return (pred - truth_vals) / truth_vals * 100


    cu_err_b = relative_errors(cu_before, truth)
    cu_err_a = relative_errors(cu_after, truth)
    ni_err_b = relative_errors(ni_before, truth)
    ni_err_a = relative_errors(ni_after, truth)
    zn_err_b = relative_errors(zn_before, truth)
    zn_err_a = relative_errors(zn_after, truth)

    results = [cu_err_b.mean(), cu_err_a.mean(),
               ni_err_b.mean(), ni_err_a.mean(),
               zn_err_b.mean(), zn_err_a.mean()]

    labels = ["Cu Before", "Cu After",
              "Ni Before", "Ni After",
              "Zn Before", "Zn After"]


    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x[0::2], results[0::2], color='skyblue', label='Before')
    bars2 = ax.bar(x[1::2], results[1::2],
                   color='orange', edgecolor='gray', hatch='////', label='After')

    ax.axhline(0, color='black', linestyle='--', linewidth=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, fontsize=14)
    ax.set_ylabel("Mean Relative Error (%)", fontsize=16)


    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(top=40)


    ax.xaxis.set_label_coords(1.0, -0.1)


    legend_patches = [
        mpatches.Patch(color='skyblue', label='Before'),
        mpatches.Patch(facecolor='orange', edgecolor='gray', hatch='////', label='After')
    ]
    ax.legend(handles=legend_patches, fontsize=14, loc="upper right")

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
