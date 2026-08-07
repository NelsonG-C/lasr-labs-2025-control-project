"""Event-count audit: which point-estimate cells rest on (near-)zero counts?

The bootstrap (and the point estimate itself) is only trustworthy where the
empirical rates entering the game value are supported by enough samples. This
audits, per condition x usefulness point, the effective event count behind
each channel of the win calculator, evaluated at the saved Nash equilibrium
strategies:

* ``pass``      -- n_attacks x P(attack passes all monitors). THE red-team win
                   channel: zero here pins safety at 1.0 and makes the
                   bootstrap degenerate (all replicates identical).
* ``flagged``   -- n_attacks x P(attack flagged for audit).
* ``hp_trigger``-- n_honeypots x P(collusion monitor stays silent on a
                   honeypot) -- the honeypot kill channel; 0 or n are both
                   boundaries.
* ``honest_fnr``/``honest_fpr`` -- honest-UM error numerators (fixed per
                   condition, threshold-independent).
* ``tm_benign`` -- n_benign x TM benign FPR at the audit budget (how many
                   benign samples pin down the audit threshold c(f)).

Dependence is handled by construction: rates are taken from the fitted ROC
function itself, so for the joint empirical model ``pass`` is the count of
attack samples in the JOINT cell (which can be zero while every marginal
count is healthy -- the marginal audit that suffices under independence is
NOT sufficient here). Interpolated/randomised thresholds make counts
fractional; a fractional count in (0, 1) means the rate is supported by less
than one sample.

Verdicts per cell are driven by the binding ``pass`` channel; a red mixture
may contain cautious components with ~0 passes, so degeneracy is judged on
the max over support pairs and precision on the mixture-weighted count:
  SATURATED  max-over-pairs pass count == 0 (bootstrap provably degenerate)
  CRITICAL   max-over-pairs pass count < 1 (sub-sample support)
  LOW        mixture-weighted pass count < 10
  OK         otherwise
Other channels (honest errors, honeypot boundary, TM-benign support) are
reported as ``low_count_caveats`` when under 10 events, without driving the
verdict.

Usage (from repo root, experiments/ symlinks in place):
  PYTHONPATH=src python .../event_count_audit.py --trees apps apps_open bigcodebench \
      --out docs/dependent_monitors_stats/results_real_paper_data/event_count_audit.json
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict


from lasr_labs_2025_control_project.scripts.helpers.list_runnable_monitor_configs import (
    runnable_configs,
)
from lasr_labs_2025_control_project.scripts.plot_data import (
    build_jsonl_dataframe_from_eval_log,
)
from lasr_labs_2025_control_project.utils.config_loader import (
    RunLabel,
    load_and_normalize_config,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.roc_function_builders import (
    build_roc_function,
)

# Paper game parameters (plot_data.run_single_config).
NUM_TASKS = 5000
BUDGET_FRACTION = 0.02
MIN_SUPPORT_WEIGHT = 0.05  # audit equilibrium support points above this prob


def audit_cell(
    roc_fn,
    honest_fpr: float,
    honest_fnr: float,
    n_att: int,
    n_hp: int,
    n_ben: int,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Effective event counts for one sweep entry, over the Nash supports."""
    replace_fnr = float(entry["blue_team_replace_fnr"])
    channels: Dict[str, float] = {}

    def merge(name: str, count: float) -> None:
        channels[name] = min(channels.get(name, float("inf")), count)

    blue_support = [
        (s["blue_team_honeypot_fraction"], p)
        for s, p in entry["blue_mixed_strategy"]
        if p >= MIN_SUPPORT_WEIGHT
    ] or [(entry["best_parameters"].get("blue_team_honeypot_fraction", 0.0), 1.0)]
    red_support = [
        (s["red_team_roc_threshold"], p)
        for s, p in entry["red_mixed_strategy"]
        if p >= MIN_SUPPORT_WEIGHT
    ] or [(entry["best_parameters"].get("red_team_roc_threshold", 0.0), 1.0)]

    pass_counts = []
    pass_weights = []
    for hp_frac, w_b in blue_support:
        audit_fraction = BUDGET_FRACTION * (1.0 - float(hp_frac))
        tm_fnr = max(audit_fraction - honest_fpr, 0.0)
        for t, w_r in red_support:
            not_flagged, _ = roc_fn(float(t), tm_fnr)
            pass_rate, _ = roc_fn(float(t), min(replace_fnr + tm_fnr, 1.0))
            _, u_fpr = roc_fn(float(t), None)
            pass_counts.append(n_att * float(pass_rate))
            pass_weights.append(w_b * w_r)
            merge("flagged", n_att * float(1.0 - not_flagged))
            # boundary on either side matters for the honeypot channel
            merge("hp_trigger", min(n_hp * float(u_fpr), n_hp * float(1.0 - u_fpr)))
            merge("tm_benign", n_ben * tm_fnr)

    # The pass channel is what saturates safety at 1.0, and it is a mixture:
    # the bootstrap only degenerates when the count is ~0 under EVERY support
    # pair (max), while precision is governed by the mixture-weighted count.
    total_w = sum(pass_weights)
    channels["pass_max"] = max(pass_counts)
    channels["pass_expected"] = (
        sum(c * w for c, w in zip(pass_counts, pass_weights)) / total_w
        if total_w
        else 0.0
    )
    channels["honest_fnr"] = n_att * honest_fnr
    channels["honest_fpr"] = n_ben * honest_fpr

    # Verdict is driven by the binding (pass) channel; the other channels are
    # reported and surfaced as caveats but do not saturate the game value.
    if channels["pass_max"] <= 1e-9:
        verdict = "SATURATED"
    elif channels["pass_max"] < 1.0:
        verdict = "CRITICAL"
    elif channels["pass_expected"] < 10.0:
        verdict = "LOW"
    else:
        verdict = "OK"
    caveats = sorted(
        name for name, v in channels.items() if not name.startswith("pass") and v < 10.0
    )
    return {
        "usefulness": entry["usefulness"],
        "safety": entry["safety"],
        "channels": {k: round(v, 3) for k, v in channels.items()},
        "low_count_caveats": caveats,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report: Dict[str, Any] = {}
    for tree in args.trees:
        for key, cfg_path in runnable_configs(tree):
            out_key = key + ("__tm_only" if "tm_only" in cfg_path.stem else "")
            normalized = load_and_normalize_config(cfg_path, config_type="monitor")
            runs = {run.label: run for run in normalized.runs}
            attacks = build_jsonl_dataframe_from_eval_log(
                runs[RunLabel.ATTACKS].save_path
            )
            honeypots = build_jsonl_dataframe_from_eval_log(
                runs[RunLabel.HONEYPOTS].save_path
            )
            benign = build_jsonl_dataframe_from_eval_log(
                runs[RunLabel.HONEST].save_path
            )
            roc_fn, meta = build_roc_function(
                attacks_df=attacks,
                honeypots_df=honeypots,
                benign_df=benign,
                disable_untrusted_monitor=normalized.defaults.get(
                    "disable_untrusted_monitor", False
                ),
            )
            payload_path = Path(normalized.results_folder) / "analysis_data.json.gz"
            payload = json.load(gzip.open(payload_path, "rt"))
            cells = [
                audit_cell(
                    roc_fn,
                    float(meta.get("untrusted_honest_fpr", 0.0)),
                    float(meta.get("untrusted_honest_fnr", 0.0)),
                    len(attacks),
                    len(honeypots),
                    len(benign),
                    entry,
                )
                for entry in payload["sweep"]
            ]
            class_name = Path(runs[RunLabel.ATTACKS].save_path).parts[1]
            report[f"{class_name}/{out_key}"] = {
                "model_type": meta.get("model_type"),
                "sample_counts": {
                    "attacks": len(attacks),
                    "honeypots": len(honeypots),
                    "benign": len(benign),
                },
                "cells": cells,
            }
            worst = min(
                cells,
                key=lambda c: ["SATURATED", "CRITICAL", "LOW", "OK"].index(
                    c["verdict"]
                ),
            )
            n_bad = sum(1 for c in cells if c["verdict"] != "OK")
            print(
                f"{class_name}/{out_key:58s} worst={worst['verdict']:9s} "
                f"non-OK cells={n_bad}/10"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
