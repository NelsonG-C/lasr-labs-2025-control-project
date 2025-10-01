import argparse
import logging
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

from lasr_labs_2025_control_project.utils.config_loader import (
    ExperimentConfig,
    load_and_normalize_config,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    OptimizationResult,
    build_optimization_config,
    optimize_parameters,
    visualization,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plot_data",
        description=(
            "Load processed EVAL logs outputs (from eval_analysis) via YAML and generate "
            "plots/optimization results. Can pass multiple configs with -cfg"
        ),
    )
    parser.add_argument(
        "--config",
        "-cfg",
        nargs="+",
        required=True,
        help="Path(s) to YAML config file(s) produced for eval_analysis",
    )
    parser.add_argument(
        "--experiment-default",
        default="exp",
        help="Fallback experiment name prefix if a config lacks defaults.experiment_name",
    )
    args = parser.parse_args()

    # Aggregate results keyed by experiment_name
    aggregated: dict[str, OptimizationResult] = {}

    for idx, cfg in enumerate(args.config, start=1):
        cfg_path = Path(cfg)
        normalized = load_and_normalize_config(cfg_path)
        # Re-run optimization for this config and store under experiment_name
        result = run_single_config(normalized)
        aggregated[normalized.experiment_name] = result

    logger.info("Generating visualizations across %d experiments", len(aggregated))
    visualization.plot_data(aggregated, destination_folder=normalized.results_folder)
    logger.info("Visualizations saved to %s", normalized.results_folder)


def build_jsonl_dataframe_from_eval_log(eval_log_path: Path) -> pd.DataFrame:
    """Build a DataFrame from a .eval log file.

    Args:
        eval_log_path: Path to the .eval log file.

    Returns:
        DataFrame with the eval log data. DataFrame has columns:
        - sample_id
        - problem
        - solution
        - untrusted_collusion_score
        - trusted_suspiciousness_score
        - collusion_monitor_system_prompt
        - collusion_monitor_user_prompt
    """
    log = read_eval_log(eval_log_path)
    assert log.samples is not None, "No samples found in the eval log"
    final_samples = []
    for sample in log.samples:
        metadata = {**sample.metadata, **sample.store}
        final_samples.append(
            {
                "sample_id": sample.id,
                "problem": metadata["main_task_description"],
                "solution": metadata["base_solution"],
                "untrusted_collusion_score": metadata["untrusted_collusion_score"],
                "trusted_suspiciousness_score": metadata[
                    "trusted_suspiciousness_score"
                ],
                "collusion_monitor_system_prompt": metadata[
                    "untrusted_collusion_monitor"
                ]["model_info"]["system_prompt"],
                "collusion_monitor_user_prompt": metadata[
                    "untrusted_collusion_monitor"
                ]["model_info"]["user_prompt"],
            }
        )
    return pd.DataFrame(final_samples)


def plot_ensemble_monitoring(
    *,
    attacks_result_eval: Path,
    honeypots_result_eval: Path,
    benign_result_eval: Path,
    num_tasks: int,
    blue_team_budget_fraction: float,
    blue_team_replace_fnr: float,
) -> OptimizationResult:
    """Build optimization inputs from both monitors using DataFrames.

    We treat:
      - attacks: positive examples for both monitors
      - honeypots: negative examples for the untrusted (collusion) monitor
      - benign (formerly 'honest'): negative examples for the trusted monitor

    Returns a dict suitable for visualization.plot_data.
    """
    logger.info("Loading attacks / honeypots / benign eval logs for monitoring plots")
    attacks_df = build_jsonl_dataframe_from_eval_log(attacks_result_eval)
    honeypots_df = build_jsonl_dataframe_from_eval_log(honeypots_result_eval)
    benign_df = build_jsonl_dataframe_from_eval_log(benign_result_eval)

    config = build_optimization_config(
        num_tasks=num_tasks,
        blue_team_budget_fraction=blue_team_budget_fraction,
        blue_team_replace_fnr=blue_team_replace_fnr,
        attacks_df=attacks_df,
        honeypots_df=honeypots_df,
        benign_df=benign_df,
    )

    result = optimize_parameters(config)

    # Attach extra metadata (diagnostic) for visualization
    meta = result.config.monitor_roc_fn_metadata or {}
    # Expose untrusted suspiciousness distributions if available
    if (
        "untrusted_suspiciousness_score" in attacks_df.columns
        and "untrusted_suspiciousness_score" in benign_df.columns
    ):
        meta.setdefault(
            "untrusted_suspiciousness_scores",
            {
                "positive": attacks_df["untrusted_suspiciousness_score"]
                .astype(float)
                .tolist(),
                "negative": benign_df["untrusted_suspiciousness_score"]
                .astype(float)
                .tolist(),
            },
        )
    result.config.monitor_roc_fn_metadata = meta
    return result


def run_single_config(normalized: ExperimentConfig) -> OptimizationResult:
    """Execute optimization for a single normalized config dict.

    Args:
        normalized: The normalized config dict from load_and_normalize_config.
        It contains the runs and the defaults.

    Returns:
        The OptimizationResult (ensemble monitoring currently assumed).
    """
    runs = {run.label: run for run in normalized.runs}

    attacks_result_eval = runs["attacks"].save_path
    honeypots_result_eval = runs["honeypots"].save_path
    benign_result_eval = runs["honest"].save_path

    return plot_ensemble_monitoring(
        attacks_result_eval=attacks_result_eval,
        honeypots_result_eval=honeypots_result_eval,
        benign_result_eval=benign_result_eval,
        num_tasks=5000,
        blue_team_budget_fraction=0.02,
        blue_team_replace_fnr=0.0,
    )


if __name__ == "__main__":
    main()
