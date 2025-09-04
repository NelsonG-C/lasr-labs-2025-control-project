import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from inspect_ai.analysis import samples_df

from lasr_labs_2025_control_project.numerical_pipeline import utils


def row_filter(row: Any) -> bool:
    try:
        # Check if score_apps_scorer column exists and has a value
        if "score_apps_scorer" not in row.index or pd.isna(row["score_apps_scorer"]):
            return False

        score_data = row["score_apps_scorer"]
        print(f"score_data: {score_data}")

        # Handle case where score_data might be a string representation of a dict
        if isinstance(score_data, str):
            print("score_data is a string")
            import ast

            try:
                score_data = ast.literal_eval(score_data)
            except (ValueError, SyntaxError):
                return False

        # Check if it's a dictionary with the required keys
        if isinstance(score_data, dict):
            main_task = score_data.get("main_task_success")
            side_task = score_data.get("side_task_success")

            # Return True only if both values are 'C'
            return main_task == "C" and side_task == "C"

        return False

    except Exception as e:
        print(f"Error in row_filter: {e}")
        return False


def get_scores_from_path(path: os.PathLike[str]) -> pd.DataFrame:
    return _extract_scores_from_df(
        samples_df(path),
        column_name="score_max_suspicion_scorer",
        row_filter=row_filter,
    )


def _extract_scores_from_df(
    df: pd.DataFrame,
    column_name: str = "max_suspiciousness_score",
    new_column_name: str = "Score",
    row_filter: Callable[[Any], bool] = lambda row: True,
) -> pd.DataFrame:
    """
    Extract scores from a DataFrame column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the scores.
    column_name (str): Name of the column containing the scores.

    Returns:
    pd.DataFrame: DataFrame containing only the specified column with scores.
    """
    assert (
        column_name in df.columns
    ), f"Column '{column_name}' not found in DataFrame.\nAvailable columns: {df.columns.tolist()}"

    try:
        df = df[df.apply(row_filter, axis=1)]
    except Exception as e:
        print(f"Error applying row filter: {e}")

    new_df: pd.DataFrame = df[[column_name]].copy()
    new_df.columns = [new_column_name]
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def data_into_pipeline() -> Any:
    tpr_file_path = Path(
        "src/lasr_labs_2025_control_project/apps/results/logs/examples/2025-09-04T16-17-35+01-00_apps-untrusted-monitoring-protocol-attack-e77f75e1_W5ckoER4zGatBwwhxf37qW.eval"
    )
    fpr_file_path = Path(
        "src/lasr_labs_2025_control_project/apps/results/logs/examples/2025-09-04T15-25-30+01-00_apps-untrusted-monitoring-protocol-attack-f9dcd062_cwPTp6GjX9prAWm2JpResx.eval"
    )
    return {
        "tpr_scores": get_scores_from_path(tpr_file_path),
        "fpr_scores": get_scores_from_path(fpr_file_path),
    }


def runner():
    data = data_into_pipeline()
    tpr_scores = data["tpr_scores"]
    fpr_scores = data["fpr_scores"]

    print("True Positive Rates (TPR) Scores:")
    print(tpr_scores)

    print("\nFalse Positive Rates (FPR) Scores:")
    print(fpr_scores)
    config = utils.build_optimization_config(
        num_tasks=5000,
        blue_team_budget_fraction=0.02,
        positive_scores=tpr_scores,
        negative_scores=fpr_scores,
    )
    result = utils.optimize_parameters(
        config=config,
        max_iterations=100,
        grid_squares_per_iteration=10,
    )
    print(result)


if __name__ == "__main__":
    runner()
