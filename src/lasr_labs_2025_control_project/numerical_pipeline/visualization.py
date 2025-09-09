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
    plot_histograms(data_dict)


def plot_histograms(
    data_dict: Dict[str, OptimizationResult],
    save_path: str = "src/lasr_labs_2025_control_project/apps/results/histograms.png",
) -> None:
    # Collect only those results that have red_team_confidence_scores available
    items = [
        (label, result)
        for label, result in data_dict.items()
        if getattr(result.config, "red_team_confidence_scores", None)
    ]

    if not items:
        print(
            "No red_team_confidence_scores found in any OptimizationResult; skipping histograms."
        )
        return

    n = len(items)
    # Arrange subplots in a near-square grid
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))

    # Normalize axes into a flat list for easy indexing
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for ax, (title, result) in zip(axes_flat, items):
        scores_dict = getattr(result.config, "red_team_confidence_scores", {}) or {}

        # Plot each score list in this subplot. Use consistent colors/alpha for readability.
        any_plotted = False
        # Use consistent integer bins from 1..10 across all histograms
        # edges centered so integer values map cleanly to bins
        bins = np.arange(0.5, 10.6, 1.0)
        for key in sorted(scores_dict.keys()):
            vals = np.atleast_1d(np.array(scores_dict[key], dtype=float))
            if vals.size == 0:
                continue
            any_plotted = True
            ax.hist(vals, bins=bins, alpha=0.6, label=key, rwidth=0.8)

        ax.set_title(title)
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        if any_plotted and len(scores_dict) > 1:
            ax.legend()

    # Turn off any unused axes
    for ax in axes_flat[len(items) :]:
        try:
            ax.axis("off")
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    return


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
        # Plot raw points (no aggregation) and show markers for every datapoint.
        # `estimator=None` disables aggregation which otherwise produces error bars
        # when multiple y values share the same x; `marker` makes each point visible.
        sns.lineplot(
            data=df,
            x="fpr",
            y="tpr",
            hue="label",
            ax=ax,
            estimator=None,
            sort=False,
        )

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
