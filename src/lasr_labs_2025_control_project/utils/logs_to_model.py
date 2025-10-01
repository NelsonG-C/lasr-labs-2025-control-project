import os
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
from inspect_ai.analysis import samples_df


def row_filter(row: Any) -> bool:
    try:
        # Check if score_apps_scorer column exists and has a value
        if "score_apps_scorer" not in row.index or pd.isna(row["score_apps_scorer"]):
            return False

        score_data = row["score_apps_scorer"]

        # Handle case where score_data might be a string representation of a dict
        if isinstance(score_data, str):
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
        column_name="score_max_confidence_scorer",
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
        # use .loc to filter rows
        filtered = df.loc[df.apply(row_filter, axis=1)]
        # Convert filtered to a DataFrame explicitly to avoid Union types
        filtered = pd.DataFrame(filtered)
        df = cast(pd.DataFrame, filtered)
    except Exception as e:
        print(f"Error applying row filter: {e}")

    new_df: pd.DataFrame = df[[column_name]].copy()  # pyright: ignore[reportAssignmentType]
    new_df.columns = [new_column_name]
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def runner():
    tpr_file_path = Path(
        "src/lasr_labs_2025_control_project/results/logs/base/2025-09-08T15-48-38+01-00_apps-untrusted-monitoring-protocol-attack-18d39942_E2cimgd9vrvKfCgTvhBkb2.eval"
    )
    fpr_file_path = Path(
        "src/lasr_labs_2025_control_project/results/logs/base/2025-09-08T21-30-28+01-00_apps-untrusted-monitoring-protocol-attack-e38dbb92_jyQux8JaWXSh6esxSwNhkW.eval"
    )

    tpr_scores = get_scores_from_path(tpr_file_path)["Score"].tolist()
    print("TPR scores length:", len(tpr_scores))
    fpr_scores = get_scores_from_path(fpr_file_path)["Score"].tolist()
    print("FPR scores length:", len(fpr_scores))

    # Commented out, I think this is old code, see plot_data.py

    raise NotImplementedError("This script is deprecated, see plot_data.py")

    """
    config = build_optimization_config(
        num_tasks=5000,
        blue_team_budget_fraction=0.02,
        red_team_confidence_scores={"positive": tpr_scores, "negative": fpr_scores},
    )
    result = _optimize_single_parameter_set(
        config=config,
        max_iterations=100,
        grid_squares_per_iteration=10,
    )
    data = {"GPT-4.1 (Base)": result}
    destination = Path("src/lasr_labs_2025_control_project/results")
    destination.mkdir(parents=True, exist_ok=True)
    visualization.plot_data(data, destination_folder=destination)
    """


if __name__ == "__main__":
    runner()
