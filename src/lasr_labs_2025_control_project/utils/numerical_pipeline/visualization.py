"""Visualization utilities for numerical pipeline."""

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr

from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    OptimizationResult,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (
    derive_untrusted_threshold_bounds,
)

OI_YELLOW = "#F0E442"
OI_TEAL = "#009E73"
OI_SKY_BLUE = "#56B4E9"
OI_BLUE = "#0072B2"
OI_VERMILION = "#D55E00"
OI_ORANGE = "#E69F00"
OI_PURPLE = "#CC79A7"

OI_PALETTE = [
    OI_ORANGE,
    OI_SKY_BLUE,
    OI_TEAL,
    OI_YELLOW,
    OI_BLUE,
    OI_VERMILION,
    OI_PURPLE,
]

CONVERGENCE_MARKERS: dict[bool, str] = {True: "o", False: "X"}
CONVERGED_SCATTER_KWARGS = {"s": 110, "edgecolor": "white", "linewidth": 0.7}

# Global setting for plot file format
PLOT_FILE_FORMAT = "svg"  # Change to "png", "pdf", etc. as needed

sns.set_color_codes("colorblind")


def _apply_standard_axes_style(ax: Axes) -> None:
    """Apply consistent axes styling while keeping left/bottom spines visible."""

    ax.set_facecolor("white")
    for spine_name, spine in ax.spines.items():
        if spine_name in {"top", "right"}:
            spine.set_visible(False)

    ax.grid(False)


def _add_unit_square_outline(
    ax: Axes,
    *,
    max_value: float = 1.0,
    linewidth: float = 0.9,
    color: str = "#b0b0b0",
) -> None:
    """Draw a faint unit square outline on ROC-style plots."""

    outline = Rectangle(
        (0.0, 0.0),
        max_value,
        max_value,
        fill=False,
        linewidth=linewidth,
        linestyle="-",
        edgecolor=color,
        zorder=0.5,
    )
    ax.add_patch(outline)


_SENSITIVITY_PARAM_META: dict[str, dict[str, Any]] = {
    "blue_team_honeypot_fraction": {
        "team": "blue",
        "label": "Blue Team Honeypot Fraction",
        "xlabel": "Blue Team Honeypot Fraction",
        "xlim": (0.0, 1.0),
        "xscale": "linear",
    },
    "red_team_roc_threshold": {
        "team": "red",
        "label": "Red Team ROC Threshold",
        "xlabel": "Red Team ROC Threshold",
        "xlim": None,
        "xscale": "linear",
    },
    "red_team_p_attack": {
        "team": "red",
        "label": "Red Team Attack Probability",
        "xlabel": "Red Team Attack Probability",
        "xlim": (1e-6, 1e-1),
        "xscale": "log",
    },
    "num_tasks": {
        "team": "green",
        "label": "Number of Tasks",
        "xlabel": "Number of Tasks",
        "xlim": None,
        "xscale": "log",
    },
    "blue_team_budget_fraction": {
        "team": "green",
        "label": "Blue Team Budget Fraction",
        "xlabel": "Blue Team Budget Fraction",
        "xlim": None,
        "xscale": "log",
    },
    "slack_honeypots": {
        "team": "green",
        "label": "Slack Honeypots",
        "xlabel": "Slack Honeypots",
        "xlim": None,
        "xscale": "log",
    },
}


def _resolve_untrusted_threshold_support(
    config: Any,
    *,
    max_points: int = 256,
) -> tuple[np.ndarray, float, float]:
    """Return sorted UM threshold samples plus reported min/max bounds."""

    meta = getattr(config, "monitor_roc_fn_metadata", {}) or {}

    thresh_min = getattr(config, "untrusted_threshold_min", None)
    thresh_max = getattr(config, "untrusted_threshold_max", None)
    if not isinstance(thresh_min, (int, float)) or not np.isfinite(thresh_min):
        thresh_min = None
    if not isinstance(thresh_max, (int, float)) or not np.isfinite(thresh_max):
        thresh_max = None

    if thresh_min is None or thresh_max is None or thresh_min >= thresh_max:
        thresh_min_meta, thresh_max_meta = derive_untrusted_threshold_bounds(meta)
        thresh_min, thresh_max = thresh_min_meta, thresh_max_meta

    candidate = meta.get("untrusted_threshold_values")
    threshold_values: np.ndarray | None = None
    if isinstance(candidate, (list, tuple, np.ndarray)):
        arr = np.asarray(candidate, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            threshold_values = np.unique(arr)

    if threshold_values is None or threshold_values.size < 2:
        threshold_values = np.linspace(float(thresh_min), float(thresh_max), max_points)
    else:
        threshold_values = threshold_values[
            (threshold_values >= float(thresh_min))
            & (threshold_values <= float(thresh_max))
        ]
        if threshold_values.size < 2:
            threshold_values = np.linspace(
                float(thresh_min), float(thresh_max), max_points
            )
        elif threshold_values.size > max_points:
            idx = np.linspace(0, threshold_values.size - 1, max_points)
            threshold_values = threshold_values[np.round(idx).astype(int)]

    return threshold_values, float(thresh_min), float(thresh_max)


def _create_subplot_grid(
    n: int,
    *,
    width_per: float = 4.5,
    height_per: float = 3.5,
    suptitle: str | None = None,
    axes_titles: list[str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Utility for creating a roughly square grid of subplots.

    Returns (fig, axes_flat). Any unused axes are turned off.
    """
    if n <= 0:
        raise ValueError("Number of subplots must be positive")
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(width_per * ncols, height_per * nrows)
    )
    axes_arr = np.asarray(axes)
    axes_flat: list[Axes] = axes_arr.flatten().tolist() if axes_arr.size > 1 else [axes]  # type: ignore[list-item]
    for ax in axes_flat[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    # Optionally assign per-axis titles (only for the first n axes actually used)
    if axes_titles:
        for i, ax in enumerate(axes_flat[:n]):
            if i < len(axes_titles) and axes_titles[i]:
                ax.set_title(axes_titles[i])
    return fig, axes_flat


def plot_data(
    data_dict: dict[str, OptimizationResult],
    destination_folder: str | Path,
) -> None:
    """
    Generate and save all plots to the specified destination folder.

    Args:
        data_dict: dict of OptimizationResult for each experiment.
        destination_folder: Directory where figures will be written.
    """
    dest = Path(destination_folder)
    dest.mkdir(parents=True, exist_ok=True)

    # Generate both types of ROC curves
    plot_roc_curves(
        data_dict,
        save_path=str(dest / f"untrusted_monitor_roc_curve.{PLOT_FILE_FORMAT}"),
        plot_mode="untrusted",
    )
    plot_roc_curves(
        data_dict,
        save_path=str(dest / f"trusted_monitor_roc_curve.{PLOT_FILE_FORMAT}"),
        plot_mode="trusted",
    )

    plot_safety_usefulness_curves(
        data_dict, save_path=str(dest / f"safety_usefulness_curves.{PLOT_FILE_FORMAT}")
    )
    # Save the underlying safety/usefulness points to JSONL for downstream analysis
    save_safety_usefulness_jsonl(
        data_dict, save_path=str(dest / "safety_usefulness.jsonl")
    )
    plot_histograms(data_dict, save_path=str(dest / f"histograms.{PLOT_FILE_FORMAT}"))
    plot_attack_score_scatter(
        data_dict, save_path=str(dest / f"attack_scores_scatter.{PLOT_FILE_FORMAT}")
    )
    """
    plot_roc_heatmaps(
        data_dict, save_path=str(dest / f"roc_heatmaps.{PLOT_FILE_FORMAT}")
    )
    """
    plot_outcome_distributions(
        data_dict, save_path=str(dest / f"outcome_distributions.{PLOT_FILE_FORMAT}")
    )
    return


def plot_histograms(
    data_dict: dict[str, OptimizationResult],
    save_path: str = f"src/lasr_labs_2025_control_project/results/histograms.{PLOT_FILE_FORMAT}",
) -> None:
    """Plot histograms of score distributions as grouped bar charts."""

    dest_path = Path(save_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    histogram_specs = [
        {
            "meta_key": "trusted_suspiciousness_scores",
            "title": "TM Suspicion Scores",
            "xlabel": "Suspicion Score",
            "class_map": {"positive": "Attack", "negative": "Benign"},
            "palette": {"Attack": OI_VERMILION, "Benign": OI_BLUE},
            "filename": f"histograms_trusted.{PLOT_FILE_FORMAT}",
            "bins": {"min": 0, "max": 10, "width": 1},
        },
        {
            "meta_key": "untrusted_collusion_scores",
            "title": "UM Collusion Scores",
            "xlabel": "Collusion Score",
            "class_map": {"positive": "Attack", "negative": "Honeypot"},
            "palette": {"Attack": OI_VERMILION, "Honeypot": OI_YELLOW},
            "filename": f"histograms_untrusted_collusion.{PLOT_FILE_FORMAT}",
            "bins": {"min": None, "max": None, "width": 2},
        },
        {
            "meta_key": "untrusted_suspiciousness_scores",
            "title": "UM Suspicion Scores",
            "xlabel": "Suspicion Score",
            "class_map": {"positive": "Attack", "negative": "Benign"},
            "palette": {"Attack": OI_VERMILION, "Benign": OI_BLUE},
            "filename": f"histograms_untrusted_suspiciousness.{PLOT_FILE_FORMAT}",
            "bins": {"min": 0, "max": 10, "width": 1},
        },
    ]

    for spec in histogram_specs:
        rows: list[dict[str, Any]] = []
        for label, result in data_dict.items():
            if result.config is None:
                continue
            meta = result.config.monitor_roc_fn_metadata or {}
            score_dict = meta.get(spec["meta_key"], {})
            if not isinstance(score_dict, dict):
                continue
            for raw_class, pretty_class in spec["class_map"].items():
                values = np.asarray(score_dict.get(raw_class, []), dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                rows.extend(
                    {
                        "label": label,
                        "class": pretty_class,
                        "score": float(v),
                    }
                    for v in values
                )

        if not rows:
            print(f"No {spec['title']} data found; skipping.")
            continue

        df = pd.DataFrame(rows)
        class_order = list(spec["palette"].keys())
        label_order = list(dict.fromkeys(df["label"].tolist()))

        score_min = float(df["score"].min())
        score_max = float(df["score"].max())
        if not np.isfinite(score_min) or not np.isfinite(score_max):
            print(f"Invalid scores encountered for {spec['title']}; skipping.")
            continue
        if np.isclose(score_min, score_max):
            epsilon = max(1e-3, abs(score_min) * 0.05 + 1e-3)
            score_min -= epsilon
            score_max += epsilon

        # Get bin config from spec (new Dict[Dict] structure)
        bins_config = spec["bins"]
        bin_width = bins_config["width"]
        bin_min = bins_config.get("min")
        bin_max = bins_config.get("max")

        # Use bin_min/bin_max for axis limits if specified, otherwise use data min/max
        xlim_min = bin_min - bin_width / 2 if bin_min is not None else score_min
        xlim_max = bin_max + bin_width / 2 if bin_max is not None else score_max

        df["rounded_score"] = (df["score"] / bin_width).round() * bin_width

        fig, axes_flat = _create_subplot_grid(
            len(label_order),
            width_per=4.3,
            height_per=3.4,
            suptitle=spec["title"],
        )
        fig.suptitle(spec["title"], y=0.99)

        legend_handles: tuple[list[Any], list[str]] | None = None

        for ax, label in zip(axes_flat, label_order):
            subset = df[df["label"] == label]
            if subset.empty:
                ax.axis("off")
                continue

            max_count = 0
            for idx, class_name in enumerate(class_order):
                class_subset = subset[subset["class"] == class_name]
                if class_subset.empty:
                    continue

                # Count occurrences of each rounded score
                value_counts = class_subset["rounded_score"].value_counts().sort_index()
                if value_counts.empty:
                    continue

                # Calculate bar positions with offsets for grouped bars
                bar_offset = (
                    (idx - (len(class_order) - 1) / 2.0)
                    * bin_width
                    * 0.8
                    / max(len(class_order), 1)
                )

                ax.bar(
                    value_counts.index + bar_offset,
                    value_counts.values,
                    width=bin_width * 0.8 / max(len(class_order), 1),
                    color=spec["palette"].get(class_name, "#555555"),
                    alpha=0.92,
                    label=class_name,
                )
                max_count = max(max_count, value_counts.values.max())

            ax.set_xlabel(spec["xlabel"])
            ax.set_ylabel("Count")
            ax.set_xlim(xlim_min, xlim_max)
            if max_count > 0:
                ax.set_ylim(0, max_count * 1.1)
            ax.set_title(label, fontsize="medium")
            _apply_standard_axes_style(ax)

            if legend_handles is None:
                handles, labels = ax.get_legend_handles_labels()
                legend_handles = (handles, labels)
            ax.legend().remove()

        if legend_handles is not None:
            handles, labels = legend_handles
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=max(1, len(handles)),
                frameon=False,
            )

        fig.tight_layout(rect=(0, 0, 1, 0.88))

        out_path = dest_path.parent / spec["filename"]
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {out_path}")
        plt.close(fig)


def plot_attack_score_scatter(
    data_dict: dict[str, OptimizationResult],
    save_path: str = f"src/lasr_labs_2025_control_project/results/attack_scores_scatter.{PLOT_FILE_FORMAT}",
) -> None:
    """Plot scatter of trusted vs untrusted scores with per-experiment regression."""
    dest_path = Path(save_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    dataframes: dict[str, pd.DataFrame] = {}

    for label, result in data_dict.items():
        if result.config is None:
            continue
        meta = result.config.monitor_roc_fn_metadata or {}
        attacks_df = meta.get("attacks_df")
        if attacks_df is None:
            continue

        if type(attacks_df) is not pd.DataFrame:
            continue
        if "untrusted_collusion_score" not in attacks_df.columns:
            continue
        if "trusted_suspiciousness_score" not in attacks_df.columns:
            continue

        dataframes[label] = attacks_df

    if not dataframes:
        print("No attack score pairs found; skipping attack score scatter.")
        return

    # Create subplot grid
    labels = list(dataframes.keys())
    fig, axes_flat = _create_subplot_grid(
        len(dataframes),
        width_per=5.0,
        height_per=4.0,
        suptitle="Attack Score Scatter: Trusted vs Untrusted",
        axes_titles=labels,
    )

    # Plot scatter with regression on each axis
    for ax, label in zip(axes_flat[: len(dataframes)], labels):
        df = dataframes[label]

        # Use seaborn's regplot which includes linear regression
        sns.regplot(
            data=df,
            x="trusted_suspiciousness_score",
            y="untrusted_collusion_score",
            ax=ax,
            scatter_kws={"alpha": 0.6, "s": 30},
            line_kws={"color": OI_VERMILION, "linewidth": 2},
            color=OI_VERMILION,
        )

        # Calculate and display regression statistics
        x = df["trusted_suspiciousness_score"].dropna()
        y = df["untrusted_collusion_score"].dropna()
        if len(x) > 1 and len(y) > 1:
            # Calculate Pearson correlation and p-value
            mask = x.notna() & y.notna()
            x_clean = x[mask]
            y_clean = y[mask]
            if len(x_clean) > 1:
                r_value, p_value = pearsonr(x_clean, y_clean)
                r_squared = float(r_value) ** 2  # type: ignore

                # Add text box with stats
                stats_text = f"$R^2 = {r_squared:.3f}$\n$p = {float(p_value):.3g}$"  # type: ignore
                ax.text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        ax.set_xlabel("Trusted Suspiciousness Score")
        ax.set_ylabel("Untrusted Collusion Score")
        _apply_standard_axes_style(ax)

    fig.tight_layout()
    fig.savefig(dest_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {dest_path}")
    plt.close(fig)


def plot_roc_heatmaps(
    data_dict: dict[str, OptimizationResult],
    save_path: str = f"src/lasr_labs_2025_control_project/results/roc_heatmaps.{PLOT_FILE_FORMAT}",
) -> None:
    # Filter to results whose monitor is an ensemble
    items = [(label, result) for label, result in data_dict.items()]

    # Grid resolution; TM FPR still sweeps [0,1], UM axis uses real thresholds
    nx, ny = 100, 100
    fpr_t_axis = np.linspace(0.0, 1.0, ny)

    # One row per ensemble result, two columns (two outputs)
    nrows, ncols = len(items), 2
    figsize = (6 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Normalize axes structure
    if nrows == 1 and ncols == 2:
        axes_grid = np.array([axes])
    else:
        axes_grid = np.asarray(axes)

    for row_idx, (label, result) in enumerate(items):
        if result.config is None:
            continue
        roc_fn = result.config.monitor_roc_fn

        threshold_axis, thresh_min, thresh_max = _resolve_untrusted_threshold_support(
            result.config, max_points=nx
        )
        if threshold_axis.size != nx:
            threshold_axis = np.linspace(thresh_min, thresh_max, nx)

        X, Y = np.meshgrid(threshold_axis, fpr_t_axis)
        XY_stack = np.column_stack([X.ravel(), Y.ravel()])

        # Evaluate both outputs on the grid. The function can broadcast, but we keep it explicit for clarity.
        attack_not_caught, fpr_u = roc_fn(XY_stack[:, 0], XY_stack[:, 1])
        Z_attack = np.asarray(attack_not_caught).reshape(Y.shape)
        Z_fpru = np.asarray(fpr_u).reshape(Y.shape)

        # Plot heatmaps
        ax1 = axes_grid[row_idx, 0]
        sns.heatmap(
            Z_attack,
            ax=ax1,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            cbar=True,
            cbar_kws={"label": "P(attack not caught)"},
        )
        ax1.set_title(f"{label}: attack not caught")
        ax1.set_xlabel("UM Threshold")
        ax1.set_ylabel("TM FPR")

        ax1.invert_yaxis()  # so y=0 at bottom, y=1 at top

        xtick_count = min(5, threshold_axis.size)
        xtick_idx = np.linspace(0, threshold_axis.size - 1, xtick_count)
        xtick_idx = np.unique(np.round(xtick_idx).astype(int))
        xtick_pos = xtick_idx + 0.5
        xtick_labels = [f"{threshold_axis[i]:.3g}" for i in xtick_idx]
        ax1.set_xticks(xtick_pos)
        ax1.set_xticklabels(xtick_labels)

        ytick_count = min(5, fpr_t_axis.size)
        ytick_idx = np.linspace(0, fpr_t_axis.size - 1, ytick_count)
        ytick_idx = np.unique(np.round(ytick_idx).astype(int))
        ytick_pos = ytick_idx + 0.5
        ytick_labels = [f"{fpr_t_axis[i]:.2f}" for i in ytick_idx]
        ax1.set_yticks(ytick_pos)
        ax1.set_yticklabels(ytick_labels)
        _apply_standard_axes_style(ax1)

        ax2 = axes_grid[row_idx, 1]
        sns.heatmap(
            Z_fpru,
            ax=ax2,
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
            cbar=True,
            cbar_kws={"label": "FPR_U"},
        )
        ax2.set_title(f"{label}: untrusted FPR")
        ax2.set_xlabel("UM Threshold")
        ax2.set_ylabel("TM FPR")
        ax2.invert_yaxis()
        ax2.set_xticks(xtick_pos)
        ax2.set_xticklabels(xtick_labels)
        ax2.set_yticks(ytick_pos)
        ax2.set_yticklabels(ytick_labels)
    _apply_standard_axes_style(ax2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    return


def plot_roc_curves(
    data_dict: dict[str, OptimizationResult],
    save_path: str = f"src/lasr_labs_2025_control_project/results/roc_curve.{PLOT_FILE_FORMAT}",
    plot_mode: str = "untrusted",  # "untrusted" or "trusted"
) -> None:
    """
    Plot ROC curves across different axes based on plot_mode.

    Args:
        data_dict: Mapping of label -> OptimizationResult
        save_path: Path to save the plot
        plot_mode: Either "untrusted" or "trusted"
            - "untrusted": Plot UM detection rate vs UM FPR across score thresholds
            - "trusted": Plot 1-attack_pass_rate vs t_tpr (roc_fn(0, t_fpr))
    """
    # Create figure with two square subplots side by side
    original_label_order = list(data_dict.keys())
    label_to_display: dict[str, str] = {}
    fig, (ax, axins) = plt.subplots(1, 2, figsize=(9, 5))

    curve_rows: list[dict[str, float | str]] = []
    point_rows: list[dict[str, float | str]] = []

    for label, result in data_dict.items():
        if result.config is None or result.config.monitor_roc_fn is None:
            continue

        roc_fn = result.config.monitor_roc_fn

        if plot_mode == "untrusted":
            thresholds, _, _ = _resolve_untrusted_threshold_support(
                result.config, max_points=400
            )
            attack_pass_rate, false_positive_rate = roc_fn(thresholds, None)
            # Concat with 0 and 1
            fpr_vals = np.concat(
                [np.asarray(false_positive_rate, dtype=float), np.array([0.0, 1.0])]
            )
            # Adjust attack pass rate BEFORE concatenation
            max_pass_rate = np.max(attack_pass_rate)
            if max_pass_rate > 0:
                attack_pass_rate = np.clip(attack_pass_rate / max_pass_rate, 0.0, 1.0)
            tpr_vals = np.concat(
                [np.asarray(attack_pass_rate, dtype=float), np.array([0.0, 1.0])]
            )
            # If fucked, don't
            if fpr_vals.size == 0 or tpr_vals.size == 0:
                continue
            # Sort
            sort_idx = np.lexsort((tpr_vals, fpr_vals))
            fpr_vals = fpr_vals[sort_idx]
            tpr_vals = tpr_vals[sort_idx]
        elif plot_mode == "trusted":
            fpr_vals = np.linspace(0.0, 1.0, 200)
            fnr_vals, _ = roc_fn(None, fpr_vals)
            tpr_vals = 1.0 - np.asarray(fnr_vals, dtype=float)
        else:
            raise ValueError(
                f"Invalid plot_mode: {plot_mode}. Must be 'untrusted' or 'trusted'."
            )

        mask = np.isfinite(fpr_vals) & np.isfinite(tpr_vals)
        fpr_vals = fpr_vals[mask]
        tpr_vals = tpr_vals[mask]
        if fpr_vals.size < 2:
            continue

        sort_idx = np.lexsort((tpr_vals, fpr_vals))
        fpr_vals = fpr_vals[sort_idx]
        tpr_vals = np.clip(tpr_vals[sort_idx], 0.0, 1.0)

        auc_val = float(np.trapezoid(tpr_vals, fpr_vals))
        display_label = f"{label} (AUC: {auc_val:.3f})"

        label_to_display[label] = display_label

        curve_rows.extend(
            {
                "fpr": float(fp),
                "tpr": float(tp),
                "label": display_label,
            }
            for fp, tp in zip(fpr_vals, tpr_vals)
        )

        positive_rates = getattr(result.config, "red_team_positive_rates", None)
        if isinstance(positive_rates, dict):
            tpr_points = np.asarray(positive_rates.get("tpr", []), dtype=float)
            fpr_points = np.asarray(positive_rates.get("fpr", []), dtype=float)
            mask_points = np.isfinite(tpr_points) & np.isfinite(fpr_points)
            for fp, tp in zip(fpr_points[mask_points], tpr_points[mask_points]):
                point_rows.append(
                    {
                        "fpr": float(fp),
                        "tpr": float(tp),
                        "label": display_label,
                    }
                )

    display_label_order = [
        label_to_display[label]
        for label in original_label_order
        if label in label_to_display
    ]

    if curve_rows:
        curve_df = pd.DataFrame(curve_rows).sort_values(["fpr", "tpr"])
        sns.lineplot(
            data=curve_df,
            x="fpr",
            y="tpr",
            hue="label",
            hue_order=display_label_order,
            estimator=None,
            palette="colorblind",
            style="label",
            style_order=display_label_order,
            ax=ax,
            sort=False,
        )

    if point_rows:
        point_df = pd.DataFrame(point_rows).sort_values(["fpr", "tpr"])
        sns.scatterplot(
            data=point_df,
            x="fpr",
            y="tpr",
            hue="label",
            hue_order=display_label_order,
            ax=ax,
            style_order=display_label_order,
            palette="colorblind",
            legend=False,
            s=60,
            marker="X",
        )

    if plot_mode == "untrusted":
        ax.set_xlabel("FPR (Honeypot Failure Rate)")
        ax.set_ylabel("TPR (Attack Pass Rate)")
        ax.set_title("Untrusted Monitor Collusion ROC")
        sub_square_coords = {
            "bottom": 0.0,
            "top": 1.0,
            "left": 0.0,
            "right": 0.05,
            "xtick_spacing": 0.01,
            "ytick_spacing": 0.2,
        }
    else:
        ax.set_xlabel("FPR (1 - Attack Pass Rate)")
        ax.set_ylabel("TPR (Trusted Monitor Detection Rate)")
        ax.set_title("Trusted Monitor ROC")
        sub_square_coords = {
            "bottom": 0.6,
            "top": 1.0,
            "left": 0.0,
            "right": 0.4,
            "xtick_spacing": 0.1,
            "ytick_spacing": 0.1,
        }

    sub_square_coords["width"] = sub_square_coords["right"] - sub_square_coords["left"]
    sub_square_coords["height"] = sub_square_coords["top"] - sub_square_coords["bottom"]

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    _apply_standard_axes_style(ax)
    _add_unit_square_outline(ax)

    # Add rectangle to main plot showing zoomed region
    rect = Rectangle(
        (sub_square_coords["left"], sub_square_coords["bottom"]),
        sub_square_coords["width"],
        sub_square_coords["height"],
        fill=False,
        edgecolor="gray",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
    )
    ax.add_patch(rect)

    # Plot the same data on the zoomed inset (second subplot)
    if curve_rows:
        sns.lineplot(
            data=curve_df,
            x="fpr",
            y="tpr",
            hue="label",
            hue_order=display_label_order,
            estimator=None,
            palette="colorblind",
            style="label",
            style_order=display_label_order,
            ax=axins,
            sort=False,
            legend=False,
        )

    if point_rows:
        sns.scatterplot(
            data=point_df,
            x="fpr",
            y="tpr",
            hue="label",
            hue_order=display_label_order,
            ax=axins,
            palette="colorblind",
            legend=False,
            s=60,
            marker="X",
        )

    # Set zoom limits to show low FPR, high TPR region (with small padding)
    axins.set_xlim(sub_square_coords["left"], sub_square_coords["right"])
    axins.set_ylim(sub_square_coords["bottom"], sub_square_coords["top"] + 0.01)

    # Adjust the outline to match the zoom region
    zoom_outline = Rectangle(
        (sub_square_coords["left"], sub_square_coords["bottom"]),
        sub_square_coords["width"],
        sub_square_coords["height"],
        fill=False,
        linewidth=0.9,
        linestyle="-",
        edgecolor="#b0b0b0",
        zorder=0.5,
    )
    axins.add_patch(zoom_outline)

    # Set ticks at 0.1 intervals
    axins.set_xticks(
        np.arange(
            sub_square_coords["left"],
            sub_square_coords["right"] + 0.01,
            sub_square_coords["xtick_spacing"],
        )
    )
    axins.set_yticks(
        np.arange(
            sub_square_coords["bottom"],
            sub_square_coords["top"] + 0.01,
            sub_square_coords["ytick_spacing"],
        )
    )
    aspect_ratio = sub_square_coords["width"] / sub_square_coords["height"]
    axins.set_aspect(aspect_ratio, adjustable="box")  # Make zoom panel square

    # Adjust which spines are visible on zoom panel
    for key, spine in axins.spines.items():
        if key in ["left", "bottom"]:
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")
        else:
            spine.set_visible(False)
    axins.set_facecolor("white")
    axins.grid(False)

    # Add labels to zoom panel
    if plot_mode == "untrusted":
        axins.set_title("Zoomed: Low FPR Region")
    else:
        axins.set_title("Zoomed: Low FPR, High TPR Region")

    axins.set_xlabel(ax.get_xlabel())
    axins.set_ylabel(ax.get_ylabel())

    # Move legend to underneath both panels
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        title=None,
        frameon=False,
        ncol=min(len(handles), 3),  # Arrange in columns to save vertical space
    )
    ax.legend().remove()  # Remove the original legend

    fig.tight_layout(rect=(0, 0.08, 1, 1))  # Make room for legend at the bottom
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close(fig)

    return


def plot_safety_usefulness_curves(
    data_dict: dict[str, OptimizationResult],
    save_path: str = f"src/lasr_labs_2025_control_project/results/safety_usefulness_curves.{PLOT_FILE_FORMAT}",
) -> None:
    dest_path = Path(save_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax, axins) = plt.subplots(1, 2, figsize=(9, 5))

    rows: list[dict[str, Any]] = []
    for label, opt_result in data_dict.items():
        print("label:", label)
        for entry in getattr(opt_result, "results", []) or []:
            safety = entry.get("safety")
            usefulness = entry.get("usefulness")
            if not isinstance(safety, (int, float)) or not isinstance(
                usefulness, (int, float)
            ):
                continue
            rows.append(
                {
                    "label": label,
                    "safety": float(safety),
                    "usefulness": float(usefulness),
                    "converged": bool(entry.get("converged", True)),
                }
            )

    if not rows:
        print("No safety/usefulness data to plot.")
        plt.close(fig)
        return

    df = pd.DataFrame(rows).dropna(subset=["safety", "usefulness"])
    if df.empty:
        print("No valid safety/usefulness points found.")
        plt.close(fig)
        return

    label_order = list(data_dict.keys())
    df = df.sort_values(by=["usefulness", "safety"])

    for idx, a in enumerate((ax, axins)):
        sns.scatterplot(
            data=df,
            x="safety",
            y="usefulness",
            hue="label",
            hue_order=label_order,
            palette="colorblind",
            style="converged",
            markers=CONVERGENCE_MARKERS,
            ax=a,
            legend=False,
            **CONVERGED_SCATTER_KWARGS,
        )
        sns.lineplot(
            data=df,
            x="safety",
            y="usefulness",
            hue="label",
            style="label",
            hue_order=label_order,
            style_order=label_order,
            palette="colorblind",
            ax=a,
            legend=idx == 0,  # Create legend on first plot to extract handles later
            estimator=None,
            zorder=0.5,
        )

        a.set_xlabel("Safety (Blue Team Win Rate)")
        a.set_ylabel("Usefulness (1 - Replacement FNR)")

        _apply_standard_axes_style(a)
        _add_unit_square_outline(a)

    ax.set_title("Safety vs Usefulness Tradeoff")
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.set_aspect("equal")

    axins.set_title("Zoomed: High Safety & Usefulness")
    axins.set_xlim(0.6, 1.01)
    axins.set_ylim(0.6, 1.01)
    ax.set_aspect("equal")

    box = Rectangle(
        (0.6, 0.6),
        0.4,
        0.4,
        fill=False,
        edgecolor="gray",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
    )
    ax.add_patch(box)

    handles, _ = ax.get_legend_handles_labels()
    if (df["converged"] == False).any():  # noqa: E712
        handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker=CONVERGENCE_MARKERS[False],
                markersize=8,
                markerfacecolor="none",
                markeredgecolor="#444444",
                markeredgewidth=1.2,
                label="Did Not Converge",
            )
        )

    # Extract labels from handles
    legend_labels = [h.get_label() for h in handles]

    # Move legend to underneath both panels
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        ncol=min(len(handles), 3),  # Arrange in columns to save vertical space
    )
    ax.legend().remove()  # Remove the original legend from ax

    fig.tight_layout(rect=(0, 0.08, 1, 1))  # Make room for legend at the bottom
    fig.savefig(dest_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {dest_path}")
    plt.close(fig)

    return


def save_safety_usefulness_jsonl(
    data_dict: dict[str, OptimizationResult],
    save_path: str = "src/lasr_labs_2025_control_project/results/safety_usefulness.jsonl",
) -> None:
    """Persist safety/usefulness sweep data to JSON Lines.

    Schema (one JSON object per line):
      {"label": <label>, "results": [
          {"safety": <float>, "usefulness": <float>, "best_parameters": {..}},
          ...
      ]}
    """
    dest_path = Path(save_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with dest_path.open("w", encoding="utf-8") as f:
        for label, opt_result in data_dict.items():
            results_list: list[dict[str, Any]] = []
            for entry in getattr(opt_result, "results", []) or []:
                safety = entry.get("safety")
                usefulness = entry.get("usefulness")
                if not isinstance(safety, (int, float)) or not isinstance(
                    usefulness, (int, float)
                ):
                    continue
                best_params = entry.get("best_parameters")
                if not isinstance(best_params, dict):
                    best_params = {}
                results_list.append(
                    {
                        "safety": float(safety),
                        "usefulness": float(usefulness),
                        "best_parameters": best_params,
                    }
                )

            if not results_list:
                continue

            f.write(
                json.dumps(
                    {"label": label, "results": results_list}, ensure_ascii=False
                )
                + "\n"
            )
            written += 1

    print(f"Saved {written} label records to: {dest_path}")
    return


def plot_parameter_sensitivity(
    sensitivity_results: Dict[str, OptimizationResult],
    destination_folder: str | Path,
) -> None:
    """Render sensitivity plots for each optimized parameter in a two-column layout.

    Left column: Red/Blue team parameters (optimization variables)
    Right column: Green team parameters (simulation environment)
    All plots use consistent y-axis (Safety = Blue Team Win Rate)
    """
    dest = Path(destination_folder)
    dest.mkdir(parents=True, exist_ok=True)

    # Extract parameter spaces metadata
    parameter_spaces_meta: Dict[str, Dict[str, Any]] = {}
    for result in sensitivity_results.values():
        if result is None or getattr(result, "config", None) is None:
            continue
        spaces_attr = getattr(result.config, "parameter_spaces", None)
        if isinstance(spaces_attr, dict) and spaces_attr:
            parameter_spaces_meta = dict(spaces_attr)
            break

        meta = result.config.monitor_roc_fn_metadata or {}
        if any(
            isinstance(meta.get(key), (int, float))
            for key in ("untrusted_threshold_min", "untrusted_threshold_max")
        ):
            thresh_min, thresh_max = derive_untrusted_threshold_bounds(meta)
            parameter_spaces_meta = {
                "red_team_roc_threshold": {
                    "min": float(thresh_min),
                    "max": float(thresh_max),
                    "scale": "linear",
                }
            }
            break

    # Build parameter metadata lists for red/blue and green teams
    red_blue_params = []
    green_params = []

    for name, meta in _SENSITIVITY_PARAM_META.items():
        # Check if data exists for this parameter
        has_fixed = f"{name}_sensitivity_fixed" in sensitivity_results
        has_reopt = f"{name}_sensitivity_reoptimized" in sensitivity_results
        if not (has_fixed or has_reopt):
            continue

        meta_copy = dict(meta)
        bounds = parameter_spaces_meta.get(name)
        if isinstance(bounds, dict):
            min_val = bounds.get("min")
            max_val = bounds.get("max")
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                # For red_team_roc_threshold, use actual data range instead of config bounds
                if name == "red_team_roc_threshold":
                    # Compute xlim from actual sensitivity data
                    data_min, data_max = None, None
                    for suffix in ["_sensitivity_fixed", "_sensitivity_reoptimized"]:
                        result_key = f"{name}{suffix}"
                        if result_key in sensitivity_results:
                            result = sensitivity_results[result_key]
                            if result and hasattr(result, "results") and result.results:
                                for entry in result.results:
                                    varied_value = entry.get("varied_value")
                                    if isinstance(
                                        varied_value, (int, float)
                                    ) and np.isfinite(varied_value):
                                        if data_min is None or varied_value < data_min:
                                            data_min = varied_value
                                        if data_max is None or varied_value > data_max:
                                            data_max = varied_value

                    # Use data range if found, otherwise use config bounds
                    if (
                        data_min is not None
                        and data_max is not None
                        and data_min < data_max
                    ):
                        meta_copy["xlim"] = (float(data_min), float(data_max))
                    else:
                        meta_copy["xlim"] = (float(min_val), float(max_val))
                else:
                    # For other parameters, use config bounds
                    meta_copy["xlim"] = (float(min_val), float(max_val))
            scale_override = bounds.get("scale")
            if isinstance(scale_override, str):
                meta_copy["xscale"] = scale_override

        if meta_copy.get("xlim") is None:
            if name == "red_team_p_attack":
                meta_copy["xlim"] = (1e-6, 1e-1)
            elif name in (
                "num_tasks",
                "blue_team_budget_fraction",
                "slack_honeypots",
                "red_team_roc_threshold",
            ):
                # Will be determined from data
                meta_copy["xlim"] = None
            else:
                meta_copy["xlim"] = (0.0, 1.0)

        if meta.get("team") == "green":
            green_params.append((name, meta_copy))
        else:
            red_blue_params.append((name, meta_copy))

    if not red_blue_params and not green_params:
        print("No parameter metadata found; skipping sensitivity plot.")
        return

    # Create two-column layout
    n_rows = max(len(red_blue_params), len(green_params))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(14.0, 3.8 * n_rows),
        sharex=False,
        sharey=True,  # Share y-axis for consistent "Safety" scale
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot red/blue parameters in left column
    for row_idx in range(n_rows):
        if row_idx < len(red_blue_params):
            param_name, meta = red_blue_params[row_idx]
            team = meta.get("team", "blue")
            _plot_single_parameter_sensitivity(
                ax=axes[row_idx, 0],
                sensitivity_results=sensitivity_results,
                param_name=param_name,
                meta=meta,
                use_consistent_ylabel=True,
                team=team,
            )
        else:
            axes[row_idx, 0].axis("off")

    # Plot green parameters in right column
    for row_idx in range(n_rows):
        if row_idx < len(green_params):
            param_name, meta = green_params[row_idx]
            _plot_single_parameter_sensitivity(
                ax=axes[row_idx, 1],
                sensitivity_results=sensitivity_results,
                param_name=param_name,
                meta=meta,
                use_consistent_ylabel=False,  # Only label on left column
                team="green",
            )
        else:
            axes[row_idx, 1].axis("off")

    out_path = dest / f"parameter_sensitivity.{PLOT_FILE_FORMAT}"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Sensitivity plot saved to: {out_path}")
    plt.close(fig)


def _plot_single_parameter_sensitivity(
    ax: Axes,
    sensitivity_results: Dict[str, OptimizationResult],
    param_name: str,
    meta: Dict[str, Any],
    use_consistent_ylabel: bool = True,
    team: str = "blue",
) -> None:
    """Plot sensitivity for a single parameter on the given axis.

    Args:
        ax: Matplotlib axis to plot on
        sensitivity_results: Full results dictionary
        param_name: Name of the parameter to plot
        meta: Metadata dict with xlabel, xlim, xscale, etc.
        use_consistent_ylabel: If True, use "Safety" as ylabel
        team: Team type - "red", "blue", or "green" for color selection
    """
    fixed_result = sensitivity_results.get(f"{param_name}_sensitivity_fixed")
    reopt_result = sensitivity_results.get(f"{param_name}_sensitivity_reoptimized")

    if (fixed_result is None or not fixed_result.results) and (
        reopt_result is None or not reopt_result.results
    ):
        ax.axis("off")
        return

    def _extract_df(opt_result: OptimizationResult | None, mode: str) -> pd.DataFrame:
        """Extract data as DataFrame for plotting."""
        if opt_result is None or not opt_result.results:
            return pd.DataFrame()
        rows = []
        for entry in opt_result.results:
            varied_value = entry.get("varied_value")
            safety = entry.get("safety")
            converged = entry.get("converged", True)
            if isinstance(varied_value, (int, float)) and isinstance(
                safety, (int, float)
            ):
                rows.append(
                    {
                        "varied_value": float(varied_value),
                        "safety": float(safety),
                        "converged": bool(converged),
                        "mode": mode,
                    }
                )
        return (
            pd.DataFrame(rows).sort_values("varied_value") if rows else pd.DataFrame()
        )

    fixed_df = _extract_df(fixed_result, "Hold constant")
    reopt_df = _extract_df(reopt_result, "Re-optimise")
    combined_df = pd.concat([fixed_df, reopt_df], ignore_index=True)

    if combined_df.empty:
        ax.axis("off")
        return

    # Use blue team perspective (safety = blue win rate) for all plots
    combined_df["y_value"] = combined_df["safety"]

    # Set up colors based on team
    if team == "green":
        # Green team: use teal (solid) and yellow (light) from Okabe-Ito
        palette = {
            "Hold constant": OI_TEAL,
            "Re-optimise": OI_YELLOW,
        }
    elif team == "red":
        # Red team: use vermilion (dark) and orange (light) from Okabe-Ito
        palette = {
            "Hold constant": OI_VERMILION,
            "Re-optimise": OI_ORANGE,
        }
    else:
        # Blue team (default): use blue colors
        palette = {
            "Hold constant": OI_BLUE,
            "Re-optimise": OI_SKY_BLUE,
        }

    linestyle_map = {"Hold constant": "-", "Re-optimise": "--"}
    zorder_map = {"Hold constant": 1.0, "Re-optimise": 2.0}
    mode_order = ["Hold constant", "Re-optimise"]

    line_handles: list[Line2D] = []
    for mode in mode_order:
        if mode not in combined_df["mode"].values:
            continue
        mode_data = combined_df[combined_df["mode"] == mode]
        (line_handle,) = ax.plot(
            mode_data["varied_value"],
            mode_data["y_value"],
            color=palette[mode],
            linestyle=linestyle_map.get(mode, "-"),
            linewidth=2.2,
            zorder=zorder_map.get(mode, 1.0),
        )
        line_handles.append(line_handle)

        for converged_val, subset in mode_data.groupby("converged", sort=False):
            if subset.empty:
                continue
            point_kwargs = {
                "s": CONVERGED_SCATTER_KWARGS["s"] * (1.0 if converged_val else 1.1),
                "marker": CONVERGENCE_MARKERS[bool(converged_val)],
                "edgecolors": CONVERGED_SCATTER_KWARGS["edgecolor"],
                "linewidths": CONVERGED_SCATTER_KWARGS["linewidth"],
                "facecolors": palette[mode],
                "zorder": zorder_map.get(mode, 1.0) + (0.05 if converged_val else 0.1),
            }

            ax.scatter(
                subset["varied_value"],
                subset["y_value"],
                **point_kwargs,
            )

    # Use consistent y-label: "Safety"
    if use_consistent_ylabel:
        ax.set_ylabel("Safety")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(meta["xlabel"])

    xlim = meta.get("xlim")
    if xlim is not None:
        ax.set_xlim(xlim)
    if meta.get("xscale") == "log":
        ax.set_xscale("log")

    # Custom legend describing lines and convergence markers
    available_modes = {str(m) for m in combined_df["mode"].unique()}
    line_handles_legend = [
        Line2D(
            [],
            [],
            color=palette[mode],
            linestyle=linestyle_map.get(mode, "-"),
            linewidth=2.2,
            label=mode,
        )
        for mode in ["Hold constant", "Re-optimise"]
        if mode in available_modes
    ]
    marker_handles = [
        Line2D(
            [],
            [],
            marker=CONVERGENCE_MARKERS[False],
            linestyle="None",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="#222222",
            markeredgewidth=1.2,
            label="Did Not Converge",
        ),
    ]
    handles = line_handles_legend + (
        marker_handles if (combined_df["converged"] == False).any() else []  # noqa: E712
    )
    if handles:
        ax.legend(
            handles,
            [str(h.get_label()) for h in handles],
            loc="best",
            frameon=False,
        )

    ax.set_title(meta["label"], fontsize="medium")
    _apply_standard_axes_style(ax)


def plot_outcome_distributions(
    data_dict: dict[str, OptimizationResult],
    save_path: str = f"src/lasr_labs_2025_control_project/results/outcome_distributions.{PLOT_FILE_FORMAT}",
) -> None:
    """Plot stacked outcome probability distributions across usefulness levels.

    For each OptimizationResult we create a subplot showing how the outcome
    probabilities vary as usefulness (1 - replacement FNR) changes. Each horizontal
    "strip" at y = usefulness is split into four colored regions whose widths sum
    to 1.0, corresponding to the four mutually exclusive red team outcomes.

    X-axis: cumulative probability (0..1) labelled "outcome" per spec.
    Y-axis: usefulness values from the optimization sweep.
    Colors (Okabe–Ito palette) and fixed order:
        red_team_fails_honeypots          -> yellow
        red_team_introduces_no_attacks    -> teal
        red_team_caught                   -> sky blue
        red_team_victory                  -> vermilion
    """

    dest_path = Path(save_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Okabe–Ito colors (consistent with rest of file)

    outcome_order = [
        "red_team_fails_honeypots",
        "red_team_introduces_no_attacks",
        "red_team_caught",
        "red_team_victory",
    ]
    outcome_colors = {
        "red_team_fails_honeypots": OI_YELLOW,
        "red_team_introduces_no_attacks": OI_TEAL,
        "red_team_caught": OI_SKY_BLUE,
        "red_team_victory": OI_VERMILION,
    }
    outcome_human = {
        "red_team_fails_honeypots": "Fails Honeypots",
        "red_team_introduces_no_attacks": "No Attacks",
        "red_team_caught": "Caught",
        "red_team_victory": "Red Team Victory",
    }

    rows: list[dict[str, float | str]] = []
    for label, opt_res in data_dict.items():
        for entry in getattr(opt_res, "results", []) or []:
            usefulness = entry.get("usefulness")
            dist = entry.get("outcome_distribution")
            if not isinstance(usefulness, (int, float)) or not isinstance(dist, dict):
                continue
            probs = [max(float(dist.get(k, 0.0)), 0.0) for k in outcome_order]
            total = sum(probs)
            if total <= 0:
                continue
            norm = [p / total for p in probs]
            for outcome_key, prob in zip(outcome_order, norm):
                rows.append(
                    {
                        "label": label,
                        "outcome": outcome_key,
                        "probability": prob,
                        "usefulness": float(usefulness),
                    }
                )

    if not rows:
        print("No outcome distributions found; skipping outcome distribution plot.")
        return

    df = pd.DataFrame(rows)
    label_order = list(dict.fromkeys(df["label"].tolist()))

    fig, axes_flat = _create_subplot_grid(
        len(label_order),
        width_per=4.5,
        height_per=3.5,
        suptitle="Outcome Distributions (stacked by usefulness)",
        axes_titles=label_order,
    )

    legend_handles: tuple[list[Any], list[str]] | None = None

    for ax, label in zip(axes_flat, label_order):
        subset = df[df["label"] == label]
        pivot = (
            subset.pivot_table(
                index="usefulness",
                columns="outcome",
                values="probability",
                aggfunc="mean",
                fill_value=0.0,
            )
            .reindex(columns=outcome_order, fill_value=0.0)
            .sort_index()
        )

        if pivot.empty:
            ax.axis("off")
            continue

        y_vals = pivot.index.to_numpy()
        cumulative = np.zeros_like(y_vals, dtype=float)

        for outcome_key in outcome_order:
            probs = pivot[outcome_key].to_numpy()
            left = cumulative
            right = cumulative + probs
            ax.fill_betweenx(
                y_vals,
                left,
                right,
                color=outcome_colors[outcome_key],
                alpha=0.9,
                label=outcome_human[outcome_key],
            )
            cumulative = right

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Usefulness")
        _apply_standard_axes_style(ax)

        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            legend_handles = (handles, labels)
        ax.legend().remove()

    if legend_handles is not None:
        handles, labels = legend_handles
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(outcome_order),
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

    fig.tight_layout()
    fig.savefig(dest_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {dest_path}")
    plt.close(fig)

    return
