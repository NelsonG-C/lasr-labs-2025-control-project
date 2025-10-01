### Visualization utils for numerical pipeline

import json
import math
import re
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import linregress

from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    OptimizationResult,
)

OI_YELLOW = "#F0E442"
OI_TEAL = "#009E73"
OI_SKY_BLUE = "#56B4E9"
OI_BLUE = "#0072B2"
OI_VERMILION = "#D55E00"


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
        save_path=str(dest / "untrusted_monitor_roc_curve.png"),
        plot_mode="untrusted",
    )
    plot_roc_curves(
        data_dict,
        save_path=str(dest / "trusted_monitor_roc_curve.png"),
        plot_mode="trusted",
    )

    plot_safety_usefulness_curves(
        data_dict, save_path=str(dest / "safety_usefulness_curves.png")
    )
    # Save the underlying safety/usefulness points to JSONL for downstream analysis
    save_safety_usefulness_jsonl(
        data_dict, save_path=str(dest / "safety_usefulness.jsonl")
    )
    plot_histograms(data_dict, save_path=str(dest / "histograms.png"))
    plot_attack_score_scatter(
        data_dict, save_path=str(dest / "attack_scores_scatter.png")
    )
    plot_roc_heatmaps(data_dict, save_path=str(dest / "roc_heatmaps.png"))
    plot_outcome_distributions(
        data_dict, save_path=str(dest / "outcome_distributions.png")
    )
    return


def plot_histograms(
    data_dict: dict[str, OptimizationResult],
    save_path: str = "src/apps/results/histograms.png",
) -> None:
    dest_path = Path(save_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    def gather_items_for_keys(
        key_names: list[str],
    ) -> list[tuple[str, OptimizationResult, str]]:
        items: list[tuple[str, OptimizationResult, str]] = []
        for label, result in data_dict.items():
            if result.config is None:
                continue
            meta = result.config.monitor_roc_fn_metadata or {}
            if not isinstance(meta, dict):
                continue
            found_key = None
            for kn in key_names:
                if kn in meta and isinstance(meta.get(kn), dict):
                    found_key = kn
                    break
            if found_key is not None:
                items.append((label, result, found_key))
        return items

    # Helper: generic grid plotting for a set of (label,result,key) tuples
    def _plot_grid(
        items: list[tuple[str, OptimizationResult, str]],
        x_label: str,
        suptitle: str,
        label_names: tuple[str, str],
        palette: dict[str, str],
        out_fname: str,
    ) -> None:
        if not items:
            print(f"No {suptitle} found in any OptimizationResult; skipping.")
            return

        fig, axes_flat = _create_subplot_grid(
            len(items),
            width_per=4,
            height_per=3,
            suptitle=suptitle,
            axes_titles=[t for t, *_ in items],  # use data_dict keys as subplot titles
        )

        for ax, (title, result, key) in zip(axes_flat, items):
            if result.config is None:
                continue
            meta = result.config.monitor_roc_fn_metadata or {}
            scores_dict = meta.get(key, {}) or {}
            pos = list(scores_dict.get("positive", []))
            neg = list(scores_dict.get("negative", []))

            if len(pos) + len(neg) == 0:
                ax.axis("off")
                continue

            data = pd.DataFrame(
                {
                    x_label: pos + neg,
                    "Label": [label_names[0]] * len(pos) + [label_names[1]] * len(neg),
                }
            )

            values = data[x_label].astype(float).to_list()
            vmin, vmax = min(values), max(values)

            def next_nice(step: int) -> int:
                if step <= 0:
                    return 1
                exp = int(math.floor(math.log10(step)))
                base = step // (10**exp)
                if base <= 1:
                    return 2 * 10**exp
                if base == 2:
                    return 5 * 10**exp
                if base == 5:
                    return 10 * 10**exp
                return 2 * 10 ** (exp + 1)

            def smallest_step_under_max_bins(
                vmin_: float, vmax_: float, max_bins_exclusive: int = 15
            ) -> int:
                if vmax_ - vmin_ <= 0:
                    return 1
                step_candidate = 1
                for _ in range(100):
                    start_local = math.floor(vmin_ / step_candidate) * step_candidate
                    end_local = math.ceil(vmax_ / step_candidate) * step_candidate
                    n_bins = int((end_local - start_local) / step_candidate) + 1
                    if n_bins < max_bins_exclusive:
                        return int(step_candidate)
                    step_candidate = next_nice(step_candidate)
                return int(step_candidate)

            step = smallest_step_under_max_bins(vmin, vmax, max_bins_exclusive=15)

            def compute_bins(current_step: int) -> list[int]:
                start_local = int(math.floor(vmin / current_step) * current_step)
                end_local = int(math.ceil(vmax / current_step) * current_step)
                return list(range(start_local, end_local + current_step, current_step))

            all_bins = compute_bins(step)

            def round_to_step(v: float) -> int:
                return int(round(v / step) * step)

            data["_bin"] = data[x_label].apply(round_to_step)

            counts = (
                data.groupby(["_bin", "Label"], as_index=False)
                .size()
                .rename(columns={"size": "Count"})
            )

            labels_unique = counts["Label"].unique().tolist()
            missing_rows = []
            present = set(zip(counts["_bin"], counts["Label"]))
            for b in all_bins:
                for lbl_val in labels_unique:
                    if (b, lbl_val) not in present:
                        missing_rows.append({"_bin": b, "Label": lbl_val, "Count": 0})
            if missing_rows:
                counts = pd.concat(
                    [counts, pd.DataFrame(missing_rows)], ignore_index=True
                )

            counts = counts.sort_values(["_bin", "Label"]).reset_index(drop=True)
            counts["_bin_str"] = counts["_bin"].astype(str)

            sns.barplot(
                data=counts,
                x="_bin_str",
                y="Count",
                hue="Label",
                ax=ax,
                palette=palette,
                dodge=True,
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel("Count")
            unique_bins = [str(b) for b in all_bins]
            ax.set_xticks(range(len(unique_bins)))
            ax.set_xticklabels(unique_bins)
            if len(unique_bins) > 12:
                ax.tick_params(axis="x", labelrotation=45)

        fig.tight_layout()
        out_path = dest_path.parent / out_fname
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {out_path}")
        plt.close(fig)

    # Build and save the three requested histogram types.
    # 1) trusted_suspiciousness_scores -> Attack vs Benign
    trusted_items = gather_items_for_keys(["trusted_suspiciousness_scores"])
    _plot_grid(
        trusted_items,
        x_label="Suspicion Score",
        suptitle="TM Suspicion Scores",
        label_names=("Attack", "Benign"),
        palette={"Attack": OI_VERMILION, "Benign": OI_BLUE},
        out_fname="histograms_trusted.png",
    )

    # 2) untrusted_collusion_scores -> Attack vs Honeypot
    # Accept some legacy keys for backward compatibility
    untrusted_collusion_items = gather_items_for_keys(
        [
            "untrusted_collusion_scores",
            "untrusted_confidence_scores",
            "confidence_scores",
        ]
    )
    _plot_grid(
        untrusted_collusion_items,
        x_label="Collusion Score",
        suptitle="UM Collusion Scores",
        label_names=("Attack", "Honeypot"),
        palette={"Attack": OI_VERMILION, "Honeypot": OI_YELLOW},
        out_fname="histograms_untrusted_collusion.png",
    )

    # 3) untrusted_suspiciousness_scores -> Attack vs Benign
    untrusted_susp_items = gather_items_for_keys(
        ["untrusted_suspiciousness_scores", "suspiciousness_scores"]
    )
    _plot_grid(
        untrusted_susp_items,
        x_label="Suspicion Score",
        suptitle="UM Suspicion Scores",
        label_names=("Attack", "Benign"),
        palette={"Attack": OI_VERMILION, "Benign": OI_BLUE},
        out_fname="histograms_untrusted_suspiciousness.png",
    )

    return


def plot_attack_score_scatter(
    data_dict: dict[str, OptimizationResult],
    save_path: str = "src/apps/results/attack_scores_scatter.png",
) -> None:
    """Per-experiment scatter plots of attack samples' s_T vs s_U with regression.

    For each OptimizationResult (experiment), we attempt to extract pairs of
    (trusted_monitor_score, untrusted_monitor_score) from metadata.
    A simple linear regression is fit; slope/intercept/R^2 and p-value are annotated.
    """
    dest_path = Path(save_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather per-experiment lists of (s_T, s_U)
    per_experiment: list[tuple[str, list[tuple[float, float]]]] = []

    for label, result in data_dict.items():
        if result.config is None:
            continue
        meta = result.config.monitor_roc_fn_metadata or {}
        if not isinstance(meta, dict):
            continue
        pairs: list[tuple[float, float]] = []

        # Preferred: A_scores matrix (n,2)
        A = meta.get("A_scores")
        if A is not None:
            try:
                A_arr = np.atleast_2d(np.asarray(A, dtype=float))
                if A_arr.ndim == 2 and A_arr.shape[1] == 2:
                    for row in A_arr:
                        pairs.append((float(row[0]), float(row[1])))
            except Exception:
                pass
        if not pairs:
            # Fallback: two separate lists
            def _as_list(maybe: object) -> list[float]:
                out: list[float] = []
                if isinstance(maybe, dict):
                    seq = maybe.get("positive", [])
                elif isinstance(maybe, (list, tuple, np.ndarray)):
                    seq = list(maybe)
                else:
                    seq = []
                for v in seq:
                    try:
                        out.append(float(v))
                    except Exception:
                        continue
                return out

            s_t_list = _as_list(
                meta.get("trusted_suspiciousness_score")
                or meta.get("trusted_suspiciousness_scores")
            )
            s_u_list = _as_list(
                meta.get("untrusted_collusion_score")
                or meta.get("untrusted_confidence_scores")
                or meta.get("untrusted_suspiciousness_scores")
            )
            if s_t_list and s_u_list:
                n = min(len(s_t_list), len(s_u_list))
                for i in range(n):
                    try:
                        pairs.append((float(s_t_list[i]), float(s_u_list[i])))
                    except Exception:
                        continue
        if pairs:
            per_experiment.append((label, pairs))

    if not per_experiment:
        print("No attack score pairs found; skipping attack score scatter.")
        return

    fig, axes_flat = _create_subplot_grid(
        len(per_experiment),
        width_per=4.5,
        height_per=3.8,
        suptitle="Attack Scatter",
        axes_titles=[lbl for lbl, _ in per_experiment],
    )

    for ax, (label, pairs) in zip(axes_flat, per_experiment):
        arr = np.asarray(pairs, dtype=float)
        s_T = arr[:, 0]
        s_U = arr[:, 1]
        sns.scatterplot(
            x=s_T, y=s_U, color=OI_VERMILION, ax=ax, s=36, marker="X", alpha=0.7
        )
        ax.set_xlabel("$T_M$ Suspiciousness Score")
        ax.set_ylabel("$U_M$ Collusion Score")
        if len(s_T) >= 2:
            try:
                lr = cast(Any, linregress(s_T, s_U))
                x_line = np.linspace(float(np.min(s_T)), float(np.max(s_T)), 50)
                y_line = lr.slope * x_line + lr.intercept
                ax.plot(
                    x_line,
                    y_line,
                    color=OI_VERMILION,
                    linewidth=1.5,
                    label=f"$R^2={lr.rvalue**2:.3g}$, $p={lr.pvalue:.2g}$",
                )
            except Exception:
                pass
        ax.set_title(label)
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(fontsize="small")
        else:
            ax.legend().remove()

    fig.tight_layout()
    fig.savefig(dest_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {dest_path}")
    plt.close(fig)
    return


def plot_roc_heatmaps(
    data_dict: dict[str, OptimizationResult],
    save_path: str = "src/apps/results/roc_heatmaps.png",
) -> None:
    # Filter to results whose monitor is an ensemble
    items = [(label, result) for label, result in data_dict.items()]

    # Grid over inputs x,y in [0,1]
    nx, ny = 100, 100
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    XY_stack = np.column_stack([X.ravel(), Y.ravel()])

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
        ax1.set_xlabel("x = (TPR_U + FPR_U)/2")
        ax1.set_ylabel("y = FPR_T")
        ax1.invert_yaxis()  # so y=0 at bottom, y=1 at top
        ax1.set_xticks([0, nx // 2, nx - 1])
        ax1.set_xticklabels(["0.0", "0.5", "1.0"])
        ax1.set_yticks([0, ny // 2, ny - 1])
        ax1.set_yticklabels(["0.0", "0.5", "1.0"])

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
        ax2.set_xlabel("x = (TPR_U + FPR_U)/2")
        ax2.set_ylabel("y = FPR_T")
        ax2.invert_yaxis()
        ax2.set_xticks([0, nx // 2, nx - 1])
        ax2.set_xticklabels(["0.0", "0.5", "1.0"])
        ax2.set_yticks([0, ny // 2, ny - 1])
        ax2.set_yticklabels(["0.0", "0.5", "1.0"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    return


def plot_roc_curves(
    data_dict: dict[str, OptimizationResult],
    save_path: str = "src/apps/results/roc_curve.png",
    plot_mode: str = "untrusted",  # "untrusted" or "trusted"
) -> None:
    """
    Plot ROC curves across different axes based on plot_mode.

    Args:
        data_dict: Mapping of label -> OptimizationResult
        save_path: Path to save the plot
        plot_mode: Either "untrusted" or "trusted"
            - "untrusted": Plot u_tpr vs attack_pass_rate (roc_fn(x, 0))
            - "trusted": Plot 1-attack_pass_rate vs t_tpr (roc_fn(0, t_fpr))
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # build a single dataframe containing all ROC curves, then plot with seaborn
    rows = []
    scatter_rows = []

    for label, result in data_dict.items():
        x = np.linspace(0, 1, 100)
        if result.config is None:
            continue
        roc_fn = result.config.monitor_roc_fn

        if plot_mode == "untrusted":
            # Plot untrusted monitor collusion ROC: u_tpr vs attack_pass_rate
            # Call roc_fn(x, 0) where x is attack_pass_rate, get (u_tpr, u_fpr)
            u_tpr, u_fpr = roc_fn(x, 0)

            # For untrusted monitor: FPR = attack_pass_rate, TPR = u_tpr
            fpr_vals = np.atleast_1d(u_fpr)
            tpr_vals = np.atleast_1d(u_tpr)

            # Normalize TPR values to account for non-zero T_M FPR at y=0
            # This ensures curve ends at (1, 1)
            tpr_vals /= np.max(tpr_vals)  # avoid exact zero TPR values

        elif plot_mode == "trusted":
            t_fnr, _ = roc_fn(1, x)
            t_tpr = np.atleast_1d(1 - t_fnr)
            t_fpr = np.atleast_1d(x)
            fpr_vals = t_fpr
            tpr_vals = t_tpr

        else:
            raise ValueError(
                f"Invalid plot_mode: {plot_mode}. Must be 'untrusted' or 'trusted'"
            )
        auc_doc = ""
        if result.config and getattr(result.config, "monitor_roc_fn", None):
            auc_doc = result.config.monitor_roc_fn.__doc__ or ""
        auc_match = re.search(r"<auc_roc>(.*?)</auc_roc>", auc_doc)
        auc_roc = float(auc_match.group(1)) if auc_match else "N/A"

        if auc_roc == "N/A":
            auc_roc = f"{np.trapezoid(tpr_vals, fpr_vals):.3g}"

        if isinstance(auc_roc, float):
            auc_display = f"{auc_roc:.3g}"
        else:
            auc_display = str(auc_roc)
        display_label = f"{label} (AUC: {auc_display})"

        rows.extend(
            [
                {"fpr": fp, "tpr": tp, "label": display_label}
                for fp, tp in zip(fpr_vals, tpr_vals)
            ]
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
            tpr_vals_config = getattr(result.config, "red_team_tpr_values", None)
            fpr_vals_config = getattr(result.config, "red_team_fpr_values", None)
            if tpr_vals_config is not None and fpr_vals_config is not None:
                positive_rates = {"tpr": tpr_vals_config, "fpr": fpr_vals_config}

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

    # Set labels and title based on plot mode
    if plot_mode == "untrusted":
        ax.set_xlabel("FPR (Attack Pass Rate)")
        ax.set_ylabel("TPR (Untrusted Monitor Detection Rate)")
        ax.set_title("Untrusted Monitor Collusion ROC")
    elif plot_mode == "trusted":
        ax.set_xlabel("FPR (1 - Attack Pass Rate)")
        ax.set_ylabel("TPR (Trusted Monitor Detection Rate)")
        ax.set_title("Trusted Monitor ROC")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    return


def plot_safety_usefulness_curves(
    data_dict: dict[str, OptimizationResult],
    save_path: str = "src/apps/results/safety_usefulness_curves.png",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    rows = []
    for label, opt_result in data_dict.items():
        # If results list exists, plot actual sweep; else fallback to legacy behavior
        if getattr(opt_result, "results", None):
            for entry in opt_result.results:
                rows.append(
                    {
                        "safety": entry.get("safety"),
                        "usefulness": entry.get("usefulness"),
                        "label": label,
                        "blue_team_replace_fnr": entry.get("blue_team_replace_fnr"),
                    }
                )
        else:
            # Legacy single-point style curve
            blue_team_win_prob = opt_result.best_score
            safety_data = [0, blue_team_win_prob, 1]
            usefulness_data = [1, 1, 0]
            for s, u in zip(safety_data, usefulness_data):
                rows.append({"safety": s, "usefulness": u, "label": label})

    if not rows:
        print("No safety/usefulness data to plot.")
        return

    df = pd.DataFrame(rows)

    # If multiple points per label, connect them in order of increasing usefulness (descending replace FNR)
    # Sorting by usefulness descending ensures typical tradeoff shape left->right if needed
    df_sorted = df.sort_values(by=["label", "usefulness"], ascending=[True, False])

    # Plot lines per label if >1 point else just scatter
    for label, sub in df_sorted.groupby("label"):
        if len(sub) > 1:
            ax.plot(
                sub["safety"],
                sub["usefulness"],
                marker="o",
                label=label,
            )
        else:
            ax.scatter(sub["safety"], sub["usefulness"], marker="o", label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Safety (Blue Team Win Rate)")
    ax.set_ylabel("Usefulness (1 - Replacement FNR)")
    ax.set_title("Safety vs Usefulness Tradeoff")
    ax.legend()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()

    return


def save_safety_usefulness_jsonl(
    data_dict: dict[str, OptimizationResult],
    save_path: str = "src/apps/results/safety_usefulness.jsonl",
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

    n_labels = 0
    with dest_path.open("w", encoding="utf-8") as f:
        for label, opt_result in data_dict.items():
            results_list: list[dict[str, Any]] = []
            if getattr(opt_result, "results", None):
                for entry in opt_result.results:
                    best_params = entry.get("best_parameters")
                    if not isinstance(best_params, dict):
                        best_params = {}
                    results_list.append(
                        {
                            "safety": entry.get("safety"),
                            "usefulness": entry.get("usefulness"),
                            "best_parameters": best_params,
                        }
                    )
            else:
                # Legacy fallback: three points without parameters
                blue_team_win_prob = opt_result.best_score
                safety_data = [0, blue_team_win_prob, 1]
                usefulness_data = [1, 1, 0]
                for s, u in zip(safety_data, usefulness_data):
                    results_list.append(
                        {"safety": s, "usefulness": u, "best_parameters": {}}
                    )

            rec = {"label": label, "results": results_list}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_labels += 1

    print(f"Saved {n_labels} label records to: {dest_path}")
    return


def plot_outcome_distributions(
    data_dict: dict[str, OptimizationResult],
    save_path: str = "src/apps/results/outcome_distributions.png",
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

    # Collect only results that actually have outcome_distribution entries
    items: list[tuple[str, OptimizationResult]] = []
    for label, opt_res in data_dict.items():
        if any(
            isinstance(r.get("outcome_distribution"), dict) for r in opt_res.results
        ):
            items.append((label, opt_res))

    if not items:
        print("No outcome distributions found; skipping outcome distribution plot.")
        return

    n = len(items)
    fig, axes_flat = _create_subplot_grid(
        n,
        width_per=4.5,
        height_per=3.5,
        suptitle="Outcome Distributions (stacked by usefulness)",
        axes_titles=[lbl for lbl, _ in items],
    )

    legend_handles = None  # capture from first populated axis

    for ax, (label, opt_res) in zip(axes_flat, items):
        # Extract usefulness values and per-outcome probabilities
        y_vals: list[float] = []
        probs_per_outcome: dict[str, list[float]] = {k: [] for k in outcome_order}

        for idx, r in enumerate(opt_res.results):
            od = r.get("outcome_distribution")
            if not isinstance(od, dict):
                continue
            usefulness = r.get("usefulness")
            if usefulness is None or not isinstance(usefulness, (float, int)):
                continue
            # Get probabilities in fixed order, normalize defensively
            raw = [float(od.get(k, 0.0)) for k in outcome_order]
            total = sum(raw)
            if total <= 0:
                continue
            norm = [v / total for v in raw]
            y_vals.append(float(usefulness))
            for k, v in zip(outcome_order, norm):
                probs_per_outcome[k].append(v)

        if len(y_vals) == 0:
            ax.set_title(f"{label}\n(no data)")
            ax.axis("off")
            continue

        # Sort by y (usefulness) so curves are monotonic visually
        order_idx = np.argsort(y_vals)
        y_sorted = np.array(y_vals)[order_idx]
        # Ensure strictly increasing for fill_betweenx aesthetics (add tiny eps if equal)
        if np.any(np.diff(y_sorted) == 0):
            # add an epsilon proportional to index to break ties visually
            eps = 1e-9
            for i in range(1, len(y_sorted)):
                if y_sorted[i] <= y_sorted[i - 1]:
                    y_sorted[i] = y_sorted[i - 1] + eps

        cum_left = np.zeros_like(y_sorted, dtype=float)

        # Plot each outcome as a horizontal band between cumulative boundaries
        for outcome in outcome_order:
            probs = np.array(probs_per_outcome[outcome])[order_idx]
            x_left = cum_left
            x_right = cum_left + probs
            ax.fill_betweenx(
                y_sorted,
                x_left,
                x_right,
                color=outcome_colors[outcome],
                label=outcome_human[outcome],
                alpha=0.9,
            )
            cum_left = x_right  # advance boundary

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Outcome")  # per spec
        ax.set_ylabel("Usefulness")
        # Title already set by axes_titles argument above.

        # capture legend handles once
        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            legend_handles = (handles, labels)
        ax.legend().remove()

    if legend_handles is not None:
        handles, labels = legend_handles
        # Place a single centered legend below plots
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )

    fig.tight_layout()
    fig.savefig(dest_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {dest_path}")
    plt.close(fig)

    return
