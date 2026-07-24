"""Safety-usefulness analysis: joint (dependence-aware) vs independence model.

Runs the full optimization pipeline on synthetic monitor-score data at several
dependence levels, since the paper's real APPS/BigCodeBench eval logs are not
available on this machine. Swap `make_score_dataframes` for the real
dataframes when the data lands; everything downstream is unchanged.

Outputs (timestamped folder next to this script):
  - safety_usefulness.pdf   main comparison figure
  - pass_rate_deviation.pdf joint-vs-product attack pass rate heatmap
  - results.json            all curves + equilibrium parameters + run metadata

Run from repo root:
  PYTHONPATH=src:tests uv run --no-project \
    --with numpy,pandas,scipy,matplotlib,seaborn \
    python docs/dependent_monitors_stats/run_synthetic_analysis.py
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from conftest import make_score_dataframes
from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (
    build_optimization_config,
    optimize_parameters,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.roc_function_builders import (
    build_roc_function,
)

RHOS = [0.0, 0.6]
# Honest-UM quality regimes: "noisy" gives epsilon ~ 0.16 (audit queue
# swamped); "sharp" gives epsilon ~ 0.01, closer to the paper's near-perfect
# UM suspiciousness ROC.
VARIANTS = {
    "noisy honest UM": dict(susp_attack=(7.0, 2.0), susp_benign=(2.5, 2.0)),
    "sharp honest UM": dict(susp_attack=(8.5, 1.2), susp_benign=(1.5, 1.2)),
}
REPLACE_FNR_VALUES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
MAX_ITERATIONS = 30
NUM_TASKS = 2000
BUDGET_FRACTION = 0.02
SLACK_HONEYPOTS = 2.0

sns.set_theme(style="whitegrid")


def run_curve(dfs, assume_independent: bool) -> dict:
    attacks_df, honeypots_df, benign_df = dfs
    config = build_optimization_config(
        num_tasks=NUM_TASKS,
        blue_team_budget_fraction=BUDGET_FRACTION,
        slack_honeypots=SLACK_HONEYPOTS,
        attacks_df=attacks_df,
        honeypots_df=honeypots_df,
        benign_df=benign_df,
        assume_independent=assume_independent,
    )
    result = optimize_parameters(
        base_config=config,
        max_iterations=MAX_ITERATIONS,
        replace_fnr_values=REPLACE_FNR_VALUES,
    )
    return {
        "model_type": config.monitor_roc_fn_metadata.get("model_type"),
        "untrusted_monitor_epsilon": config.monitor_roc_fn_metadata.get(
            "untrusted_monitor_epsilon"
        ),
        "untrusted_honest_fnr": config.monitor_roc_fn_metadata.get(
            "untrusted_honest_fnr"
        ),
        "points": [
            {
                "safety": r["safety"],
                "usefulness": r["usefulness"],
                "blue_team_replace_fnr": r["blue_team_replace_fnr"],
                "best_parameters": r["best_parameters"],
                "converged": r["converged"],
            }
            for r in result.results
        ],
    }


def deviation_surface(dfs) -> dict:
    """Max |joint - product| attack pass rate over the (t, fpr_t) grid."""
    attacks_df, honeypots_df, benign_df = dfs
    fn_joint, meta = build_roc_function(
        attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
    )
    fn_ind, _ = build_roc_function(
        attacks_df=attacks_df,
        honeypots_df=honeypots_df,
        benign_df=benign_df,
        assume_independent=True,
    )
    t_axis = np.linspace(0, 10, 41)
    f_axis = np.linspace(0, 1, 41)
    T, F = np.meshgrid(t_axis, f_axis)
    joint, _ = fn_joint(T.ravel(), F.ravel())
    ind, _ = fn_ind(T.ravel(), F.ravel())
    dev = (np.asarray(joint) - np.asarray(ind)).reshape(T.shape)
    return {"t_axis": t_axis, "f_axis": f_axis, "deviation": dev}


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = Path(__file__).parent / f"results_synthetic_{stamp}"
    out_dir.mkdir(parents=True)

    results = {
        "metadata": {
            "commit": commit,
            "dirty_worktree": dirty,
            "timestamp": stamp,
            "data": "synthetic (Gaussian copula, integer 0-10 scores); "
            "real APPS/BCB eval logs unavailable on this machine",
            "params": {
                "num_tasks": NUM_TASKS,
                "budget_fraction": BUDGET_FRACTION,
                "slack_honeypots": SLACK_HONEYPOTS,
                "replace_fnr_values": REPLACE_FNR_VALUES,
                "max_iterations": MAX_ITERATIONS,
                "rhos": RHOS,
            },
        },
        "curves": {},
    }

    n_rows, n_cols = len(VARIANTS), len(RHOS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), sharey=True, sharex=True
    )
    axes = np.atleast_2d(axes)
    palette = sns.color_palette("colorblind")

    for row, (variant, susp_kwargs) in enumerate(VARIANTS.items()):
        for col, rho in enumerate(RHOS):
            ax = axes[row, col]
            dfs = make_score_dataframes(rho=rho, seed=0, **susp_kwargs)
            # Ablation: permute the TM column on attacks to remove
            # cross-monitor dependence while keeping the joint model's
            # honest-error treatment. (legacy vs shuffled) isolates the
            # epsilon/asymmetry effect; (shuffled vs joint) isolates the
            # dependence effect.
            shuffled_attacks = dfs[0].copy()
            shuffled_attacks["trusted_suspiciousness_score"] = np.random.default_rng(
                1
            ).permutation(shuffled_attacks["trusted_suspiciousness_score"].to_numpy())
            dfs_shuffled = (shuffled_attacks, dfs[1], dfs[2])

            entry = {}
            for label, run_dfs, flag, color in [
                ("joint (dependence-aware)", dfs, False, palette[0]),
                (
                    "joint, TM shuffled (no dependence)",
                    dfs_shuffled,
                    False,
                    palette[2],
                ),
                ("independence assumption (legacy)", dfs, True, palette[3]),
            ]:
                print(f"{variant} rho={rho} model={label} ...", flush=True)
                curve = run_curve(run_dfs, assume_independent=flag)
                entry[label] = curve
                pts = sorted(curve["points"], key=lambda p: p["usefulness"])
                ax.plot(
                    [p["safety"] for p in pts],
                    [p["usefulness"] for p in pts],
                    marker="o",
                    markersize=3,
                    color=color,
                    label=label,
                )
            results["curves"][f"{variant}|rho={rho}"] = entry
            eps = entry["joint (dependence-aware)"]["untrusted_monitor_epsilon"]
            ax.set_title(f"{variant}, ρ = {rho} (ε = {eps:.3f})", fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel("Safety (blue team win rate)")
            if col == 0:
                ax.set_ylabel("Usefulness")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(
        "Safety-usefulness: joint empirical model vs independence assumption "
        "(synthetic data)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "safety_usefulness.pdf")
    plt.close(fig)

    # Deviation heatmap at the strongest dependence
    dev = deviation_surface(make_score_dataframes(rho=RHOS[-1], seed=0))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        dev["deviation"],
        cmap="vlag",
        center=0.0,
        ax=ax,
        cbar_kws={"label": "joint − independent attack pass rate"},
    )
    ax.set_xticks(np.linspace(0, len(dev["t_axis"]) - 1, 6))
    ax.set_xticklabels([f"{v:.0f}" for v in np.linspace(0, 10, 6)])
    ax.set_yticks(np.linspace(0, len(dev["f_axis"]) - 1, 6))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("red team collusion threshold t")
    ax.set_ylabel("trusted monitor FPR budget")
    ax.set_title(f"Model disagreement, ρ = {RHOS[-1]} (synthetic)")
    fig.tight_layout()
    fig.savefig(out_dir / "pass_rate_deviation.pdf")
    plt.close(fig)

    results["deviation_max_abs"] = float(np.abs(dev["deviation"]).max())
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nWrote {out_dir}")
    for rho, entry in results["curves"].items():
        for label, curve in entry.items():
            s = [p["safety"] for p in curve["points"]]
            print(
                f"{rho:10s} {label:28s} safety@u=1.0: {s[0]:.4f}  "
                f"epsilon: {curve['untrusted_monitor_epsilon']}"
            )


if __name__ == "__main__":
    sys.exit(main())
