"""
Transform data to new format script.

This script transforms JSON data files from the old format to the new format
using Pydantic models. It supports both single file processing and recursive
directory processing.

Usage Examples:
    # Process a single file
    python transform_data_to_new_format.py input.json output_folder/

    # Process all JSON files recursively
    python transform_data_to_new_format.py data/ output_folder/ --recursive

    # Process recursively and keep original filenames
    python transform_data_to_new_format.py data/ output_folder/ --recursive --keep-name

The input data should be in JSON format with either:
1. Single model format: {problem_id, problem, model, solution}
2. Comparison format: {problem_id, problem, model_A, solution_A, model_B, solution_B}

The output will be in the new format using Dataset, ProblemSolutions, and Solution models.
"""

import argparse
import json
from json import JSONEncoder
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd  # type: ignore

from lasr_labs_2025_control_project.self_rec.utils.scripts.data_models import (
    Dataset,
    ProblemSolutions,
    Solution,
)

# Some monkey-patching to make UUIDs JSON serializable
old_default = JSONEncoder.default


def new_default(self: JSONEncoder, o: Any) -> Any:
    """
    Custom JSON encoder that handles UUID objects.

    Args:
        self: The JSONEncoder instance
        o: The object to encode

    Returns:
        String representation of UUID or falls back to default encoder
    """
    if isinstance(o, UUID):
        return str(o)
    return old_default(self, o)


JSONEncoder.default = new_default


def process_data(input_file: str, output_file: str):
    """
    Process the JSON data and convert it to the desired format.

    Reads a JSON file containing problem data and converts it to the new format
    using Pydantic models. Handles both single model and comparison formats.

    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file

    Example:
        process_data("old_format.json", "new_format.json")
    """
    df = pd.read_json(input_file)

    problem_solutions = []
    for index, row in df.iterrows():
        row = row.to_dict()
        if "model" in row:
            problem_solutions.append(
                ProblemSolutions(
                    apps_problem_id=row["problem_id"],  # pyright: ignore[reportArgumentType]
                    problem_statement=row["problem"],  # pyright: ignore[reportArgumentType]
                    solutions=[
                        Solution(
                            model=row["model"],
                            solution=row["solution"],
                        )
                    ],
                )
            )
        else:
            problem_solutions.append(
                ProblemSolutions(
                    apps_problem_id=row["problem_id"],
                    problem_statement=row["problem"],
                    solutions=[
                        Solution(
                            model=row[f"model_{_}"],
                            solution=row[f"solution_{_}"],
                        )
                        for _ in ["A", "B"]
                    ],
                )
            )

    with open(output_file, "w") as f:
        json.dump(
            Dataset(problems=problem_solutions).model_dump(),
            f,
        )


def main():
    """
    Main CLI function for processing JSON data files.

    Provides command-line interface for transforming data files from old format
    to new format. Supports both single file processing and recursive directory
    processing with various options.

    Command-line Arguments:
        input_path: Path to input JSON file or directory
        output_folder: Path to output folder where processed files will be saved
        -r, --recursive: Process all JSON files in input directory recursively
        --keep-name: Keep original filenames instead of generating UUIDs

    Examples:
        # Process single file
        python transform_data_to_new_format.py data.json output/

        # Process directory recursively
        python transform_data_to_new_format.py data/ output/ --recursive

        # Process directory recursively keeping original names
        python transform_data_to_new_format.py data/ output/ --recursive --keep-name
    """
    parser = argparse.ArgumentParser(description="Process JSON data files")
    parser.add_argument("input_path", help="Input JSON file path")
    parser.add_argument("output_folder", help="Output folder path")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Process files recursively"
    )
    parser.add_argument(
        "--keep-name", action="store_true", help="Keep the name of the input file"
    )

    args = parser.parse_args()

    if args.recursive:
        print(f"Processing recursively from {args.input_path}")
        input_path = Path(args.input_path)
        output_folder = Path(args.output_folder)
        dirs_to_process = [input_path]
        while dirs_to_process:
            current_dir = dirs_to_process.pop(0)
            for file in current_dir.iterdir():
                if file.suffix == ".json":
                    print(f"Processing {file}")
                    output_filename = file.name if args.keep_name else f"{uuid4()}.json"
                    process_data(str(file), str(output_folder / output_filename))
                elif file.is_dir():
                    if output_folder == file:
                        print(f"Skipping {file} because it is the output folder")
                    else:
                        dirs_to_process.append(file)
        print(
            f"Successfully processed {args.input_path} and saved to {args.output_folder}"
        )

    else:
        print(f"Processing {args.input_path} to {args.output_path}")
        process_data(args.input_path, args.output_path)
        print(
            f"Successfully processed {args.input_path} and saved to {args.output_folder}"
        )


if __name__ == "__main__":
    main()
