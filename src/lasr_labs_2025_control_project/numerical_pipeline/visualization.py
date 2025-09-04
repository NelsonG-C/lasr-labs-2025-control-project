### Visualization utils for numerical pipeline

from typing import Dict

import matplotlib.pyplot as plt

from lasr_labs_2025_control_project.numerical_pipeline.utils import (
    OptimizationResult,
)


def plot_safety_usefulness_curves(
    data_dict: Dict[str, OptimizationResult],
    save_path: str = "results/safety_usefulness_curves.png",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, result in data_dict.items():
        blue_team_win_prob = result.best_score
        safety_data = [0, blue_team_win_prob, 1]
        usefulness_data = [1, 1, 0]
        ax.plot(safety_data, usefulness_data, marker="o", label=label)

    ax.set_xlabel("Safety")
    ax.set_ylabel("Usefulness")
    ax.set_title("Safety vs Usefulness Curves")
    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()
