"""Bootstrap the full analysis per condition, saving per-replicate data.

Supersedes the prototype on the `bootstrapping` branch. For every replicate we
resample the attack / honeypot / benign dataframes with replacement (rows,
i.e. whole samples -- preserving the joint dependence structure the stats
model exists to capture), rebuild the joint ROC model, re-solve the game, and
persist the complete analysis payload (safety/usefulness sweep, blue/red Nash
mixed strategies, empirical ROC curves, joint-surface samples) as one JSON
line, so CI bands and strategy diagnostics can be computed later without
re-running anything.

Conditions are enumerated the same way as the analysis driver: from monitor
configs whose output triples exist on disk (see
helpers/list_runnable_monitor_configs.py), so the bootstrap can never silently
run on a different set of conditions than the point estimates.

    PYTHONPATH=src python src/lasr_labs_2025_control_project/scripts/bootstrap.py \
        --trees apps_open --replicates 1000 --out-root bootstrap_results/open-source-model

Output layout (one directory per condition):

    <out-root>/<setting>/<experiment>/<arm>[__<config-stem>]/
        replicates.jsonl.gz   # one analysis payload per line, gzip members appended per chunk
        manifest.json         # commit, params, seeds, timing, replicate count

Resumable: on restart, completed replicates (lines already in replicates.jsonl.gz)
are skipped. Seeds are deterministic per (condition, replicate index).
"""

import argparse
import gzip
import json
import logging
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    build_analysis_payload,
    build_optimization_config,
    optimize_parameters,
)

logger = logging.getLogger(__name__)

# Paper game parameters -- keep in lockstep with plot_data.run_single_config.
NUM_TASKS = 5000
SLACK_HONEYPOTS = 5.0
BLUE_TEAM_BUDGET_FRACTION = 0.02

# Per-replicate payload size knobs: no optimizer history, coarser surface grid
# than the point estimate (31x21 instead of 61x41).
SURFACE_T_POINTS = 31
SURFACE_F_POINTS = 21


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_single_replicate(
    replicate: int,
    *,
    condition_seed: int,
    attacks_df: pd.DataFrame,
    honeypots_df: pd.DataFrame,
    benign_df: pd.DataFrame,
    disable_trusted_monitor: bool,
    disable_untrusted_monitor: bool,
) -> Dict[str, Any]:
    """Resample, re-fit, re-solve; return the analysis payload for one replicate."""
    seeds = np.random.SeedSequence([condition_seed, replicate]).generate_state(3)
    config = build_optimization_config(
        num_tasks=NUM_TASKS,
        blue_team_budget_fraction=BLUE_TEAM_BUDGET_FRACTION,
        blue_team_replace_fnr=0.0,
        attacks_df=attacks_df.sample(
            len(attacks_df), replace=True, random_state=int(seeds[0])
        ),
        honeypots_df=honeypots_df.sample(
            len(honeypots_df), replace=True, random_state=int(seeds[1])
        ),
        benign_df=benign_df.sample(
            len(benign_df), replace=True, random_state=int(seeds[2])
        ),
        slack_honeypots=SLACK_HONEYPOTS,
        disable_trusted_monitor=disable_trusted_monitor,
        disable_untrusted_monitor=disable_untrusted_monitor,
    )
    result = optimize_parameters(base_config=config, max_iterations=100)
    payload = build_analysis_payload(
        result,
        include_history=False,
        joint_t_points=SURFACE_T_POINTS,
        joint_f_points=SURFACE_F_POINTS,
    )
    payload["bootstrap_replicate"] = replicate
    return payload


def _existing_replicates(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        logger.warning("Could not read %s; starting this condition over", path)
        path.unlink()
        return 0


def bootstrap_condition(
    key: str,
    cfg_path: Path,
    *,
    out_dir: Path,
    replicates: int,
    n_jobs: int,
    chunk_size: int,
) -> None:
    normalized = load_and_normalize_config(cfg_path, config_type="monitor")
    runs = {run.label: run for run in normalized.runs}
    attacks_df = build_jsonl_dataframe_from_eval_log(runs[RunLabel.ATTACKS].save_path)
    honeypots_df = build_jsonl_dataframe_from_eval_log(
        runs[RunLabel.HONEYPOTS].save_path
    )
    benign_df = build_jsonl_dataframe_from_eval_log(runs[RunLabel.HONEST].save_path)
    disable_tm = normalized.defaults.get("disable_trusted_monitor", False)
    disable_um = normalized.defaults.get("disable_untrusted_monitor", False)

    out_dir.mkdir(parents=True, exist_ok=True)
    lines_path = out_dir / "replicates.jsonl.gz"
    done = _existing_replicates(lines_path)
    if done >= replicates:
        logger.info("[%s] already complete (%d replicates)", key, done)
        return
    if done:
        logger.info("[%s] resuming at replicate %d", key, done)

    # Stable across processes and restarts (unlike builtin hash()).
    condition_seed = zlib.crc32(key.encode("utf-8"))
    manifest = {
        "condition": key,
        "config": str(cfg_path),
        "commit": _git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "replicates_target": replicates,
        "condition_seed": condition_seed,
        "game_parameters": {
            "num_tasks": NUM_TASKS,
            "slack_honeypots": SLACK_HONEYPOTS,
            "blue_team_budget_fraction": BLUE_TEAM_BUDGET_FRACTION,
        },
        "sample_counts": {
            "attacks": len(attacks_df),
            "honeypots": len(honeypots_df),
            "benign": len(benign_df),
        },
        "disable_trusted_monitor": disable_tm,
        "disable_untrusted_monitor": disable_um,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    start = time.time()
    for chunk_start in range(done, replicates, chunk_size):
        chunk = range(chunk_start, min(chunk_start + chunk_size, replicates))
        payloads = Parallel(n_jobs=n_jobs)(
            delayed(run_single_replicate)(
                r,
                condition_seed=condition_seed,
                attacks_df=attacks_df,
                honeypots_df=honeypots_df,
                benign_df=benign_df,
                disable_trusted_monitor=disable_tm,
                disable_untrusted_monitor=disable_um,
            )
            for r in chunk
        )
        # Appending opens a new gzip member; readers see one continuous stream.
        with gzip.open(lines_path, "at", encoding="utf-8") as f:
            for payload in payloads:
                f.write(json.dumps(payload) + "\n")
        elapsed = time.time() - start
        completed = chunk.stop - done
        logger.info(
            "[%s] %d/%d replicates (%.1fs/replicate wall)",
            key,
            chunk.stop,
            replicates,
            elapsed / max(completed, 1),
        )

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["replicates_done"] = replicates
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", nargs="+", required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional condition-key substrings to filter to",
    )
    args = parser.parse_args()

    todo: list[tuple[str, Path]] = []
    for tree in args.trees:
        for key, cfg_path in runnable_configs(tree):
            # tm_only shares a condition key with none_base; disambiguate by
            # config stem so both keep separate output directories.
            out_key = key
            if "tm_only" in cfg_path.stem:
                out_key = f"{key}__tm_only"
            if args.conditions and not any(s in out_key for s in args.conditions):
                continue
            todo.append((out_key, cfg_path))

    logger.info(
        "Bootstrapping %d conditions x %d replicates", len(todo), args.replicates
    )
    for out_key, cfg_path in todo:
        bootstrap_condition(
            out_key,
            cfg_path,
            out_dir=args.out_root / out_key,
            replicates=args.replicates,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
        )
    logger.info("ALL_BOOTSTRAPS_DONE")


if __name__ == "__main__":
    main()
