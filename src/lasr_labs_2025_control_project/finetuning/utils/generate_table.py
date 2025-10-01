#!/usr/bin/env python3
"""
Script to generate a JSONL table with columns [system_message, user_message, model, ground_truth, prediction]
for a specific sampled model from OpenAI evaluation results .jsonl file.

The evaluation results must follow a specific schema format, see `eval_schema.json` for more details.
"""

import json
import re
from pathlib import Path
from typing import Any

import click


def extract_answer_from_prediction(prediction: str) -> str:
    """
    Extract YES/NO answer from prediction text, handling various formats.

    Args:
        prediction: Raw prediction text

    Returns:
        Extracted answer ("YES" or "NO") or empty string if not found
    """
    if not prediction:
        return ""

    # First try to find <answer> tags
    answer_match = re.search(r"<answer>(YES|NO)</answer>", prediction, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Fallback: look for YES/NO in the text
    yes_match = re.search(r"\b(YES)\b", prediction, re.IGNORECASE)
    no_match = re.search(r"\b(NO)\b", prediction, re.IGNORECASE)

    if yes_match and no_match:
        # If both found, return the one that appears first
        if yes_match.start() < no_match.start():
            return "YES"
        else:
            return "NO"
    elif yes_match:
        return "YES"
    elif no_match:
        return "NO"

    # Default to empty string if no clear answer found
    return ""


def load_eval_data(file_path: Path) -> list[dict[str, Any]]:
    """
    Load evaluation data from JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of evaluation records
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    return data


def extract_table_data(record: dict[str, Any]) -> dict[str, str]:
    """
    Extract table data from a single evaluation record.

    Args:
        record: Single evaluation record

    Returns:
        Dictionary with table columns
    """
    # Extract system message (developer role content)
    system_message = ""
    try:
        trajectory = record.get("sample", {}).get("trajectory", [])
        for msg in trajectory:
            if msg.get("role") == "developer":
                system_message = msg.get("content", "").strip()
                break
    except (KeyError, IndexError, AttributeError):
        system_message = ""

    # Extract user message (user role content)
    user_message = ""
    try:
        trajectory = record.get("sample", {}).get("trajectory", [])
        for msg in trajectory:
            if msg.get("role") == "user":
                user_message = msg.get("content", "").strip()
                break
    except (KeyError, IndexError, AttributeError):
        user_message = ""

    # Extract model (the model that generated the solution)
    model = record.get("item", {}).get("model", "").strip()

    # Extract ground truth from item.answer
    raw_ground_truth = record.get("item", {}).get("answer", "").strip()
    ground_truth = extract_answer_from_prediction(raw_ground_truth)

    # Extract prediction from model's response
    raw_prediction = ""
    try:
        outputs = record.get("sample", {}).get("outputs", [])
        if outputs and len(outputs) > 0:
            raw_prediction = outputs[0].get("content", "").strip()
    except (KeyError, IndexError, AttributeError):
        raw_prediction = ""

    prediction = extract_answer_from_prediction(raw_prediction)

    return {
        "system_message": system_message,
        "user_message": user_message,
        "model": model,
        "ground_truth": ground_truth,
        "prediction": prediction,
    }


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--sampled-model",
    required=True,
    help="Name of the sampled model to filter for (e.g., 'ft:gpt-4.1-2025-04-14:lasr-labs::CC5IeIFO')",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output JSONL file path (default: prints to stdout)",
)
@click.option("--verbose", is_flag=True, help="Show verbose output")
def main(input_file: Path, sampled_model: str, output: Path | None, verbose: bool):
    """
    Generate a JSONL table with columns [system_message, user_message, model, ground_truth, prediction]
    for a specific sampled model from evaluation results.
    """
    if verbose:
        print(f"Loading data from: {input_file}")

    # Load evaluation data
    data = load_eval_data(input_file)
    if verbose:
        print(f"Loaded {len(data)} evaluation records")

    # Filter for the specified sampled model
    filtered_data = []
    for record in data:
        record_sampled_model = (
            record.get("sample", {}).get("sampled_model_name", "").strip()
        )
        if record_sampled_model == sampled_model:
            filtered_data.append(record)

    if verbose:
        print(f"Found {len(filtered_data)} records for sampled model: {sampled_model}")

    if not filtered_data:
        print(f"Error: No records found for sampled model '{sampled_model}'")
        return

    # Extract table data
    table_data = []
    for record in filtered_data:
        table_row = extract_table_data(record)
        table_data.append(table_row)

    # Output results
    if output:
        with open(output, "w", encoding="utf-8") as f:
            for row in table_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if verbose:
            print(f"Table saved to: {output}")
    else:
        # Print to stdout
        for row in table_data:
            print(json.dumps(row, ensure_ascii=False))

    if verbose:
        print(f"Generated {len(table_data)} table rows")


if __name__ == "__main__":
    main()
