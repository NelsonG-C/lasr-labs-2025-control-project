"""
README
To use the script, add FINAL_RESULTS data (e.g. `apps/self_rec` folder with `outputs` subfolder with monitor results) to the project.

Specify the path to this data folder in `dataset_folder` variable in `main()` (line 44), e.g. `experiments/apps` for a folder containing apps data (with subfolders `self_rec`, `causal`, ...).

Run the script with `uv run bootstrap.py`, it will create a `bootstrap_results` folder in each setup folder (e.g. `experiments/apps/self_rec/bootstrap_results) with all the resulting .jsonl files containing all the S/U values.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from lasr_labs_2025_control_project.scripts.plot_data import (
    build_jsonl_dataframe_from_eval_log,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    build_optimization_config,
    optimize_parameters,
)


def main():
    """Main execution function with parallelization."""

    # Configuration parameters
    disable_trusted_monitor = False
    disable_untrusted_monitor = False
    num_tasks = 5000
    slack_honeypots = 5.0
    blue_team_budget_fraction = 0.02
    blue_team_replace_fnr = 0.0

    # Determine number of parallel jobs
    n_jobs = -2
    bootstrap_size = 1000

    result_dir = Path("bootstrap_results")
    dataset_folder = Path("experiments/apps")

    # full path is like "experiments/apps/self_rec/outputs/none_base/attack_monitor_results.eval"
    eval_files = list(dataset_folder.glob("*/outputs/*/*.eval"))
    triplets = [sorted(eval_files[i : i + 3]) for i in range(0, len(eval_files), 3)]

    start_time = time.time()
    for i, (attacks_path, benign_path, honeypots_path) in enumerate(triplets):
        parts = attacks_path.parts
        # e.g. apps-self_rec-none_base or apps-additive-minified_sft
        experiment_slug = f"{parts[1]}-{parts[2]}-{parts[4]}"

        print(f"Processing triplet {i + 1}/{len(triplets)} ({experiment_slug})")

        # full path is like "experiments/apps/self_rec/bootstrap_results/none_sft-bootstrap_1000.jsonl"
        result_path = f"{dataset_folder}/{parts[2]}/{result_dir}/{parts[4]}-bootstrap_{bootstrap_size}.jsonl"
        result_path = Path(result_path)
        if result_path.exists():
            print(f"Result path {result_path} already exists, skipping")
            continue

        attacks_df = build_jsonl_dataframe_from_eval_log(attacks_path)
        benign_df = build_jsonl_dataframe_from_eval_log(benign_path)
        honeypots_df = build_jsonl_dataframe_from_eval_log(honeypots_path)

        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Parallel execution with progress tracking
        results: list[dict[str, Any]] = Parallel(n_jobs=n_jobs, verbose=10)(  # pyright: ignore[reportAssignmentType]
            delayed(run_single_bootstrap)(
                iteration=i,
                num_tasks=num_tasks,
                blue_team_budget_fraction=blue_team_budget_fraction,
                blue_team_replace_fnr=blue_team_replace_fnr,
                attacks_df=attacks_df,
                honeypots_df=honeypots_df,
                slack_honeypots=slack_honeypots,
                benign_df=benign_df,
                disable_trusted_monitor=disable_trusted_monitor,
                disable_untrusted_monitor=disable_untrusted_monitor,
            )
            for i in range(bootstrap_size)
        )

        # Save to JSON file
        with open(result_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"Saved {experiment_slug} results to {result_path}")
        print(f"elapsed: {time.time() - start_time:.2f}s", end=" | ")
        print(
            f"remaining: {((time.time() - start_time) / (i + 1)) * (len(triplets) - (i + 1)):.2f}s"
        )


def run_single_bootstrap(
    iteration: int,
    num_tasks: int,
    blue_team_budget_fraction: float,
    blue_team_replace_fnr: float,
    attacks_df: pd.DataFrame,
    honeypots_df: pd.DataFrame,
    slack_honeypots: float,
    benign_df: pd.DataFrame,
    disable_trusted_monitor: bool,
    disable_untrusted_monitor: bool,
) -> dict[str, Any]:
    """
    Run a single bootstrap iteration.

    Args:
        iteration: Bootstrap iteration number (used for random seed)
        ... other parameters for configuration

    Returns:
        Optimization result dictionary
    """
    random_state = 42 + iteration

    config = build_optimization_config(
        num_tasks=num_tasks,
        blue_team_budget_fraction=blue_team_budget_fraction,
        blue_team_replace_fnr=blue_team_replace_fnr,
        attacks_df=attacks_df.sample(
            len(attacks_df), replace=True, random_state=random_state
        ),
        honeypots_df=honeypots_df.sample(
            len(honeypots_df), replace=True, random_state=random_state
        ),
        slack_honeypots=slack_honeypots,
        benign_df=benign_df.sample(
            len(benign_df), replace=True, random_state=random_state
        ),
        disable_trusted_monitor=disable_trusted_monitor,
        disable_untrusted_monitor=disable_untrusted_monitor,
    )

    result = asdict(optimize_parameters(base_config=config, max_iterations=100))
    result["bootstrap_iteration"] = iteration
    del result["config"]
    result["metadata"] = {
        "num_tasks": num_tasks,
        "blue_team_budget_fraction": blue_team_budget_fraction,
        "blue_team_replace_fnr": blue_team_replace_fnr,
        "slack_honeypots": slack_honeypots,
        "disable_trusted_monitor": disable_trusted_monitor,
        "disable_untrusted_monitor": disable_untrusted_monitor,
    }
    return result


if __name__ == "__main__":
    start_time = time.time()
    results = main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal running time: {elapsed_time:.2f} seconds")
