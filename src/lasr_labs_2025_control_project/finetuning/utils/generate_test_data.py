"""
Build test JSONL for OpenAI Eval API.

Uses Dataset class from lasr_labs_2025_control_project.finetuning.scripts.data_models.

Expected input: Dataset containing ProblemSolutions with Solution objects.

Outputs:
 - test_data.jsonl  (simple format with columns: problem_statement, solution, model, answer)

Usage:
  uv run generate-test-data --help
"""

import json
import logging
from pathlib import Path

import click

from lasr_labs_2025_control_project.self_rec_baselines.utils.scripts.data_models import (
    Dataset,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_data(path: str | Path) -> Dataset:
    """
    Load dataset using the Dataset class.
    Validates that the path exists and is readable.
    """
    path = Path(path)
    logger.info(f"Attempting to load dataset from: {path}")

    if not path.exists():
        logger.error(f"Dataset path does not exist: {path}")
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if not path.is_dir() and not path.is_file():
        logger.error(f"Dataset path is neither a file nor a directory: {path}")
        raise ValueError(f"Dataset path is neither a file nor a directory: {path}")

    try:
        dataset = Dataset.read(path)
        logger.info(
            f"Successfully loaded dataset from {path} with {len(dataset)} problems."
        )
        return dataset
    except Exception as e:
        logger.exception(f"Failed to load dataset from {path}: {e}")
        raise RuntimeError(f"Failed to load dataset from {path}: {e}") from e


def prepare_test_data(dataset: Dataset, target_model_name: str) -> list[dict[str, str]]:
    """
    Prepare test data with columns: problem_statement, solution, model, answer
    """
    logger.info(f"Preparing test data for target model: {target_model_name}")

    # See generate_train_data.prepare_rows: a target model matching nothing
    # silently labels every example NO.
    models_present = {
        solution.model for problem in dataset for solution in problem.solutions
    }
    if target_model_name not in models_present:
        raise ValueError(
            f"--target-model {target_model_name!r} matches no solution in this "
            f"dataset; every example would be labelled NO. Models present: "
            f"{sorted(models_present)}"
        )

    test_rows = []
    num_pos = 0
    num_neg = 0

    for idx, problem in enumerate(dataset):
        problem_statement = problem.problem_statement
        logger.debug(
            f"Processing problem {idx + 1}/{len(dataset)}: {getattr(problem, 'apps_problem_id', 'N/A')}"
        )

        for sol_idx, solution in enumerate(problem.solutions):
            is_target_model = solution.model == target_model_name
            answer = (
                "<answer>YES</answer>" if is_target_model else "<answer>NO</answer>"
            )
            if is_target_model:
                num_pos += 1
            else:
                num_neg += 1

            logger.debug(
                f"  Solution {sol_idx + 1}: model={solution.model}, is_target={is_target_model}, answer={answer}"
            )

            test_rows.append(
                {
                    "problem_statement": problem_statement,
                    "solution": solution.solution,
                    "model": solution.model,
                    "answer": answer,
                }
            )

    logger.info(
        f"Prepared {len(test_rows)} test rows ({num_pos} positives, {num_neg} negatives)."
    )
    return test_rows


def write_test_jsonl(rows: list[dict[str, str]], filepath: str | Path):
    """Write test data to JSONL file with robust error handling."""
    filepath = Path(filepath)
    logger.info(f"Writing {len(rows)} rows to JSONL file: {filepath}")

    # Ensure parent directory exists
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured parent directory exists: {filepath.parent}")
    except Exception as e:
        logger.error(f"Failed to create parent directory {filepath.parent}: {e}")
        raise

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if (i + 1) % 1000 == 0:
                    logger.info(f"Wrote {i + 1} rows so far...")
        logger.info(f"Successfully wrote test JSONL file to {filepath}")
    except Exception as e:
        logger.exception(f"Failed to write test JSONL file to {filepath}: {e}")
        raise RuntimeError(f"Failed to write test JSONL file to {filepath}: {e}") from e


def build_and_save_jsonl(
    test_dataset: Dataset,
    target_model_name: str,
    seed: int,
    output_folder: str | Path,
):
    """Build and save test JSONL file."""
    output_folder = Path(output_folder)
    logger.info(f"Building and saving test JSONL to output folder: {output_folder}")

    # Ensure output folder exists
    if not output_folder.parent.exists():
        logger.error(f"Parent directory does not exist: {output_folder.parent}")
        raise FileNotFoundError(
            f"Parent directory does not exist: {output_folder.parent}"
        )
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured output folder exists: {output_folder}")

    # Generate test data in simple format
    test_data = prepare_test_data(test_dataset, target_model_name)

    # Write file
    test_data_file = output_folder / "test.jsonl"
    write_test_jsonl(test_data, test_data_file)

    # Print summary
    click.echo(f"Test problems: {len(test_dataset)}")
    click.echo(f"Test data rows: {len(test_data)}")
    click.echo(f"Wrote: {test_data_file}")
    logger.info(f"Test data generation complete. Wrote file: {test_data_file}")


@click.command()
@click.option(
    "--dataset-folder",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the dataset folder containing the test data",
)
@click.option(
    "--output-folder",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the output folder where JSONL files will be written",
)
@click.option(
    "--target-model",
    required=True,
    help=(
        "Name of the model to treat as positive examples. Required: a wrong "
        "or defaulted value silently labels every example NO, producing a "
        "dataset that trains the monitor to always answer NO."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible shuffling (default: 42)",
)
@click.option(
    "--max-problems",
    type=int,
    help="Maximum number of problems to process (default: all)",
)
def main(
    dataset_folder: Path,
    output_folder: Path,
    target_model: str,
    seed: int,
    max_problems: int | None,
):
    """Build test JSONL for binary fine-tuning."""

    click.echo("Starting test data generation script.")
    click.echo(
        f"Parameters: dataset_folder={dataset_folder}, output_folder={output_folder}, target_model={target_model}, seed={seed}, max_problems={max_problems}"
    )

    # Load dataset
    try:
        dataset = load_data(dataset_folder)
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        raise click.Abort()

    # Limit problems if specified
    if max_problems is not None:
        click.echo(f"Limiting dataset to first {max_problems} problems.")
        dataset = dataset[:max_problems]

    # Sanity checks
    if len(dataset) == 0:
        click.echo(f"No problems found in dataset at {dataset_folder}.", err=True)
        raise click.Abort()

    try:
        build_and_save_jsonl(
            dataset,
            target_model_name=target_model,
            seed=seed,
            output_folder=output_folder,
        )
        click.echo("Successfully generated test data!")
    except Exception as e:
        click.echo(f"Error generating data: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
