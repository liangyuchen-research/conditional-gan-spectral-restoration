"""Plot archived repeat-measurement and training error summaries."""

def run():
    from pathlib import Path
    import os
    from project_paths import create_run
    DATA_DIR, OUTPUT_DIR = create_run('plot_repeatability')

    import matplotlib.pyplot as plt
    import numpy as np

    elements = ['Zn', 'Ni', 'Cu']
    retake_means   = [2.10, -27.27, -14.63]
    retake_stds    = [2.79, 2.64, 7.77]
    training_means = [8.51, -30.13, -10.29]
    training_stds  = [3.63, 4.88, 4.24]

    bar_width = 0.35
    x = np.arange(len(elements))

    fig, ax = plt.subplots(figsize=(8, 6))


    ax.bar(x - bar_width/2, retake_means, bar_width, yerr=retake_stds,
           label='Retake', color='skyblue', capsize=8, edgecolor='black', alpha=0.9)


    ax.bar(x + bar_width/2, training_means, bar_width, yerr=training_stds,
           label='Training', color='orange', edgecolor='black', alpha=0.9, hatch='//', capsize=8)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)

    ax.set_xlabel('Element', fontsize=20)
    ax.set_ylabel('Mean Relative Deviation (%)', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(elements, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.legend(fontsize=18, loc='best')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
