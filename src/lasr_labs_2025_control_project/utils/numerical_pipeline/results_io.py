"""Serialize optimization results to a compact, self-contained data file.

One `analysis_data.json.gz` per condition holds everything needed to redraw
the paper's figures (and run diagnostics) without re-running the optimizer:

* the safety/usefulness sweep, with the Nash mixed strategies of both teams
  at every sweep point;
* the empirical ROC curves of each monitor (raw survival curves -- no convex
  hulling, matching the joint model's use of the raw empirical marginals);
* a sampled grid of the joint attack-pass surface f(t, fpr_t);
* the honest-UM operating point (FPR/FNR, threshold, binary-or-numeric) and
  the game parameters.

The same payload builder is reused by the bootstrap runner, which stores one
payload per resample (with `include_history=False` and no per-sample score
lists, so a 15-condition x 1000-replicate run stays a few GB).
"""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .optimization import OptimizationResult


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and tuples to JSON types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None  # JSON has no NaN/inf; None round-trips unambiguously
    return obj


def _raw_roc(positive: np.ndarray, negative: np.ndarray) -> Dict[str, list]:
    """Raw empirical ROC: survival rates at every unique threshold, unhulled."""
    thresholds = np.unique(np.concatenate([positive, negative]))
    span = max(float(thresholds[-1] - thresholds[0]), 1e-8)
    pad = span * 0.01
    thresholds = np.concatenate(
        ([thresholds[0] - pad], thresholds, [thresholds[-1] + pad])
    )
    tpr = [float(np.mean(positive >= t)) for t in thresholds]
    fpr = [float(np.mean(negative >= t)) for t in thresholds]
    return {"thresholds": thresholds.tolist(), "tpr": tpr, "fpr": fpr}


def _clean_scores(df: Optional[pd.DataFrame], col: str) -> Optional[np.ndarray]:
    if df is None or col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    return values if len(values) else None


def build_analysis_payload(
    result: OptimizationResult,
    *,
    include_history: bool = False,
    joint_t_points: int = 61,
    joint_f_points: int = 41,
) -> Dict[str, Any]:
    """Build the serializable analysis payload from an OptimizationResult."""
    config = result.config
    meta = config.monitor_roc_fn_metadata or {}
    attacks_df = meta.get("attacks_df")
    honeypots_df = meta.get("honeypots_df")
    benign_df = meta.get("benign_df")

    payload: Dict[str, Any] = {
        "game_parameters": {
            "num_tasks": config.num_tasks,
            "blue_team_budget_fraction": config.blue_team_budget_fraction,
            "slack_honeypots": config.slack_honeypots,
        },
        "monitor_metadata": {
            key: _to_jsonable(meta.get(key))
            for key in (
                "model_type",
                "untrusted_honest_fpr",
                "untrusted_honest_fnr",
                "untrusted_honest_threshold",
                "untrusted_honest_is_binary",
                "joint_num_attack_rows",
            )
            if key in meta
        },
        "sample_counts": {
            name: (len(df) if df is not None else None)
            for name, df in (
                ("attacks", attacks_df),
                ("honeypots", honeypots_df),
                ("benign", benign_df),
            )
        },
        "converged": bool(result.converged),
    }

    # --- Marginal ROC curves (raw empirical survival curves) ---
    roc_curves: Dict[str, Any] = {}
    tm_pos = _clean_scores(attacks_df, "trusted_suspiciousness_score")
    tm_neg = _clean_scores(benign_df, "trusted_suspiciousness_score")
    if tm_pos is not None and tm_neg is not None:
        roc_curves["trusted_suspiciousness"] = _raw_roc(tm_pos, tm_neg)
    # Collusion monitor: positives = attacks it should wave through for red,
    # negatives = honeypots. tpr here is P(coll >= t | attack).
    coll_pos = _clean_scores(attacks_df, "untrusted_collusion_score")
    coll_neg = _clean_scores(honeypots_df, "untrusted_collusion_score")
    if coll_pos is not None and coll_neg is not None:
        roc_curves["untrusted_collusion"] = _raw_roc(coll_pos, coll_neg)
    susp_pos = _clean_scores(attacks_df, "untrusted_suspiciousness_score")
    susp_neg = _clean_scores(benign_df, "untrusted_suspiciousness_score")
    if susp_pos is not None and susp_neg is not None:
        roc_curves["untrusted_suspiciousness"] = _raw_roc(susp_pos, susp_neg)
    payload["roc_curves"] = roc_curves

    # --- Joint attack-pass surface, sampled on a regular grid ---
    if coll_pos is not None and coll_neg is not None:
        coll_all = np.concatenate([coll_pos, coll_neg])
        span = max(float(coll_all.max() - coll_all.min()), 1e-8)
        t_grid = np.linspace(
            float(coll_all.min()) - 0.01 * span,
            float(coll_all.max()) + 0.01 * span,
            joint_t_points,
        )
        f_grid = np.linspace(0.0, 1.0, joint_f_points)
        surface = np.empty((joint_t_points, joint_f_points))
        hp_fpr = np.empty(joint_t_points)
        for i, t in enumerate(t_grid):
            hp_fpr[i] = float(config.monitor_roc_fn(float(t), None)[1])
            for j, f in enumerate(f_grid):
                surface[i, j] = float(config.monitor_roc_fn(float(t), float(f))[0])
        payload["joint_surface"] = {
            "collusion_threshold_grid": t_grid.tolist(),
            "trusted_fpr_grid": f_grid.tolist(),
            "attack_pass_rate": surface.tolist(),
            "honeypot_fpr": hp_fpr.tolist(),
        }

    # --- Safety/usefulness sweep with Nash strategies ---
    sweep = []
    for entry in result.results:
        record = {
            key: _to_jsonable(entry.get(key))
            for key in (
                "safety",
                "usefulness",
                "blue_team_replace_fnr",
                "best_parameters",
                "blue_mixed_strategy",
                "red_mixed_strategy",
                "converged",
            )
        }
        if include_history:
            record["history"] = _to_jsonable(entry.get("history"))
        sweep.append(record)
    payload["sweep"] = sweep

    return payload


def save_analysis_payload(payload: Dict[str, Any], path: "str | Path") -> Path:
    """Write a payload as gzipped JSON, creating parent directories."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return dest


def load_analysis_payload(path: "str | Path") -> Dict[str, Any]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        return json.load(f)
