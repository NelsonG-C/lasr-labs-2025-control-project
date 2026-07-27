"""
Build train/test JSONL for binary fine-tuning (OpenAI) with upsampling.

Uses Dataset class from lasr_labs_2025_control_project.finetuning.scripts.data_models.

Expected input: Dataset containing ProblemSolutions with Solution objects.

Outputs:
 - train.jsonl      (upsampled if needed)

Usage:
  uv run generate-data --help
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import click

from lasr_labs_2025_control_project.self_rec_baselines.utils.scripts.data_models import (
    Dataset,
)
from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Prompt variables
ANSWER_YES = "<answer>YES</answer>"
ANSWER_NO = "<answer>NO</answer>"


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


def build_messages(
    problem_statement: str,
    solution: str,
    is_target_model: bool,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    """
    Create the messages format used for binary examples.
    Based on the format used in dspy_exps.ipynb.
    """
    user_prompt = user_prompt.format(
        problem_statement=problem_statement,
        solution=solution,
    )

    answer = ANSWER_YES if is_target_model else ANSWER_NO

    logger.debug(
        f"Building messages for problem_statement (truncated): {problem_statement[:60]!r}, "
        f"solution (truncated): {solution[:60]!r}, is_target_model: {is_target_model}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer},
    ]


def prepare_rows(
    dataset: Dataset,
    target_model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, Any]]:
    """
    Convert the dataset into flat positive and negative rows.
    Returns a list of dicts where each dict is:
      {"problem_id":..., "messages": [...]}
    """
    logger.info("Preparing rows for dataset")

    # A target model that matches nothing labels every example NO, which
    # trains the monitor to always answer NO -- and the resulting dataset
    # looks superficially fine. Fail loudly instead.
    models_present = {
        solution.model for problem in dataset for solution in problem.solutions
    }
    if target_model_name not in models_present:
        raise ValueError(
            f"--target-model {target_model_name!r} matches no solution in this "
            f"dataset; every example would be labelled NO. Models present: "
            f"{sorted(models_present)}"
        )

    rows = []

    for idx, problem in enumerate(dataset):
        problem_id = problem.apps_problem_id
        problem_statement = problem.problem_statement
        logger.debug(
            f"Processing problem {idx + 1}/{len(dataset)}: problem_id={problem_id}"
        )

        for sol_idx, solution in enumerate(problem.solutions):
            messages = build_messages(
                problem_statement,
                solution.solution,
                solution.model == target_model_name,
                system_prompt,
                user_prompt,
            )
            rows.append(
                {
                    "problem_id": problem_id,
                    "messages": messages,
                    "model": solution.model,
                }
            )
            logger.debug(
                f"Added example for problem_id={problem_id}, solution_index={sol_idx}, model={solution.model}"
            )
    logger.info(f"Prepared {len(rows)} rows.")
    return rows


def write_jsonl(rows: list[dict[str, Any]], filepath: str | Path):
    """Write rows to JSONL file with robust error handling."""
    filepath = Path(filepath)
    logger.info(f"Writing {len(rows)} rows to JSONL file: {filepath}")

    # Ensure parent directory exists
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured parent directory exists: {filepath.parent}")
    except Exception as e:
        logger.exception(f"Failed to create parent directory {filepath.parent}: {e}")
        raise

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                # Write the messages format directly
                entry = {"messages": r["messages"]}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if (i + 1) % 1000 == 0:
                    logger.info(f"Wrote {i + 1} rows so far to {filepath}")
        logger.info(f"Successfully wrote JSONL file: {filepath}")
    except Exception as e:
        logger.exception(f"Failed to write JSONL file to {filepath}: {e}")
        raise RuntimeError(f"Failed to write JSONL file to {filepath}: {e}") from e


def build_and_save_jsonl(
    train_dataset: Dataset,
    target_model_name: str,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    output_folder: str | Path,
    upsample_pos_factor: int = 1,
    upsample_neg_factor: int = 1,
):
    """Build and save train JSONL file.

    Args:
        train_dataset: Dataset containing ProblemSolutions with Solution objects.
        target_model_name: Name of the model to treat as positive examples.
        seed: Random seed for reproducible shuffling.
        output_folder: Path to the output folder where JSONL files will be written.
        upsample_pos_factor: Factor to upsample positives. Must be an integer greater than 0. E.g. if 2, there will be 2x more positives than in the original dataset.
        upsample_neg_factor: Factor to upsample negatives. Must be an integer greater than 0. E.g. if 2, there will be 2x more negatives than in the original dataset.
    """
    output_folder = Path(output_folder)
    logger.info(f"Building and saving JSONL to {output_folder}")
    logger.info(
        f"Parameters: target_model_name={target_model_name}, seed={seed}, "
        f"upsample_pos_factor={upsample_pos_factor}, upsample_neg_factor={upsample_neg_factor}"
    )

    # Ensure output folder exists
    if not output_folder.parent.exists():
        logger.error(f"Parent directory does not exist: {output_folder.parent}")
        raise FileNotFoundError(
            f"Parent directory does not exist: {output_folder.parent}"
        )
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured output folder exists: {output_folder}")

    # Build flattened rows
    train_rows = prepare_rows(
        train_dataset, target_model_name, system_prompt, user_prompt
    )
    logger.info(f"Original train rows: {len(train_rows)}")

    train_pos = [
        sample
        for _ in range(upsample_pos_factor)
        for sample in train_rows
        if sample["model"] == target_model_name
    ]
    train_neg = [
        sample
        for _ in range(upsample_neg_factor)
        for sample in train_rows
        if sample["model"] != target_model_name
    ]
    logger.info(
        f"After upsampling: positives={len(train_pos)}, negatives={len(train_neg)}"
    )

    # Train set rows: include all negatives and the upsampled positives
    train_rows = []
    train_rows.extend(train_neg)
    train_rows.extend(train_pos)
    logger.info(f"Total train rows before shuffling: {len(train_rows)}")

    # Shuffle training rows for randomness
    random.Random(seed).shuffle(train_rows)
    logger.info(f"Shuffled train rows with seed {seed}")

    # Write files
    train_file = output_folder / "train_messages.jsonl"
    logger.info(f"Writing train rows to file: {train_file}")

    write_jsonl(train_rows, train_file)

    # Print summary
    click.echo(f"Train problems: {len(train_dataset)}")
    click.echo(f"Train negatives: {len(train_neg)}")
    click.echo(f"Train positives: {len(train_pos)}")
    click.echo(f"Train final rows: {len(train_rows)}")
    click.echo(f"Wrote: {train_file}")
    logger.info(
        f"Summary: Train problems={len(train_dataset)}, negatives={len(train_neg)}, positives={len(train_pos)}, final rows={len(train_rows)}"
    )
    logger.info(f"Finished writing train data to {train_file}")


@click.command()
@click.option(
    "--dataset-folder",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the dataset folder containing the training data",
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
    "--system-prompt",
    type=str,
    required=True,
    default="SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT",
    help="Name of the system prompt. The prompt must exist among the prompts in the lasr_labs_2025_control_project.prompts folder.",
)
@click.option(
    "--user-prompt",
    type=str,
    required=True,
    default="MONITOR_USER_PROMPT",
    help="Name of the user prompt. The prompt must exist among the prompts in the lasr_labs_2025_control_project.prompts folder.",
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
@click.option(
    "--upsample-pos-factor",
    type=int,
    default=1,
    help="Factor to upsample positives (default: 1)",
)
@click.option(
    "--upsample-neg-factor",
    type=int,
    default=1,
    help="Factor to upsample negatives (default: 1)",
)
def main(
    dataset_folder: Path,
    output_folder: Path,
    target_model: str,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    max_problems: int | None,
    upsample_pos_factor: int,
    upsample_neg_factor: int,
):
    """Build train JSONL for binary fine-tuning with positive upsampling."""

    logger.info("Starting training data generation process.")
    logger.info(
        f"Arguments: dataset_folder={dataset_folder}, output_folder={output_folder}, "
        f"target_model={target_model}, seed={seed}, max_problems={max_problems}, "
        f"upsample_pos_factor={upsample_pos_factor}, upsample_neg_factor={upsample_neg_factor}"
    )

    # Load dataset
    try:
        dataset = load_data(dataset_folder)
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        logger.error(f"Error loading dataset: {e}")
        raise click.Abort()

    # Limit problems if specified
    if max_problems is not None:
        logger.info(f"Limiting dataset to first {max_problems} problems.")
        dataset = dataset[:max_problems]

    # Sanity checks
    if len(dataset) == 0:
        click.echo(f"No problems found in dataset at {dataset_folder}.", err=True)
        logger.error(f"No problems found in dataset at {dataset_folder}.")
        raise click.Abort()

    # Load system prompt
    system_prompt = get_prompt_by_name(system_prompt)
    user_prompt = get_prompt_by_name(user_prompt)

    build_and_save_jsonl(
        dataset,
        target_model_name=target_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        seed=seed,
        output_folder=output_folder,
        upsample_pos_factor=upsample_pos_factor,
        upsample_neg_factor=upsample_neg_factor,
    )
    click.echo("Successfully generated training data!")
    logger.info("Successfully generated training data!")


if __name__ == "__main__":
    main()
