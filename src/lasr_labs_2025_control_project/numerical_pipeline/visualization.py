### Visualization utils for numerical pipeline

import re
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lasr_labs_2025_control_project.numerical_pipeline.utils import (
    OptimizationResult,
)


def plot_data(
    data_dict: Dict[str, OptimizationResult],
) -> None:
    plot_roc_curves(data_dict)
    plot_safety_usefulness_curves(data_dict)


def plot_roc_curves(
    data_dict: Dict[str, OptimizationResult],
    save_path: str = "src/lasr_labs_2025_control_project/apps/results/roc_curve.png",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # build a single dataframe containing all ROC curves, then plot with seaborn
    rows = []
    scatter_rows = []
    for label, result in data_dict.items():
        x = np.linspace(0, 1, 100)
        tpr, fpr = result.config.red_team_roc_fn(x)
        # ensure we have iterable arrays (function may return scalars for scalar inputs)
        tpr = np.atleast_1d(tpr)
        fpr = np.atleast_1d(fpr)
        auc_match = re.search(
            r"<auc_roc>(.*?)</auc_roc>", result.config.red_team_roc_fn.__doc__ or ""
        )
        auc_roc = float(auc_match.group(1)) if auc_match else "N/A"
        if type(auc_roc) is float:
            auc_roc = f"{auc_roc:.3g}"
        display_label = f"{label} (AUC: {auc_roc})"
        rows.extend(
            [{"fpr": fp, "tpr": tp, "label": display_label} for fp, tp in zip(fpr, tpr)]
        )

        # collect any explicit TPR/FPR score points from config
        # support a few possible field names used across the codebase
        positive_rates = getattr(result.config, "red_team_positive_rates", None)
        if positive_rates is None:
            positive_rates = getattr(
                result.config, "red_team_positive_rate_values", None
            )
        # legacy fields: red_team_tpr_values & red_team_fpr_values
        if positive_rates is None:
            tpr_vals = getattr(result.config, "red_team_tpr_values", None)
            fpr_vals = getattr(result.config, "red_team_fpr_values", None)
            if tpr_vals is not None and fpr_vals is not None:
                positive_rates = {"tpr": tpr_vals, "fpr": fpr_vals}

        if (
            isinstance(positive_rates, dict)
            and "tpr" in positive_rates
            and "fpr" in positive_rates
        ):
            tpr_points = np.atleast_1d(positive_rates["tpr"])
            fpr_points = np.atleast_1d(positive_rates["fpr"])
            for fp, tp in zip(fpr_points, tpr_points):
                scatter_rows.append(
                    {"fpr": float(fp), "tpr": float(tp), "label": display_label}
                )

    if rows:
        df = pd.DataFrame(rows)
        sns.lineplot(data=df, x="fpr", y="tpr", hue="label", ax=ax)

    # plot any scatter points on top (suppress duplicate legend entries)
    if scatter_rows:
        scatter_df = pd.DataFrame(scatter_rows)
        sns.scatterplot(
            data=scatter_df,
            x="fpr",
            y="tpr",
            hue="label",
            ax=ax,
            legend=False,
            s=60,
            marker="X",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    return


def plot_safety_usefulness_curves(
    data_dict: Dict[str, OptimizationResult],
    save_path: str = "src/lasr_labs_2025_control_project/apps/results/safety_usefulness_curves.png",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # build dataframe containing all safety-usefulness curves, then plot via seaborn
    rows = []
    for label, result in data_dict.items():
        blue_team_win_prob = result.best_score
        safety_data = [0, blue_team_win_prob, 1]
        usefulness_data = [1, 1, 0]
        rows.extend(
            [
                {"safety": s, "usefulness": u, "label": label}
                for s, u in zip(safety_data, usefulness_data)
            ]
        )

    if rows:
        df = pd.DataFrame(rows)
        sns.lineplot(
            data=df, x="safety", y="usefulness", hue="label", marker="o", ax=ax
        )

    ax.set_xlabel("Safety")
    ax.set_ylabel("Usefulness")
    ax.set_title("Safety vs Usefulness Curves")
    ax.legend()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    return
