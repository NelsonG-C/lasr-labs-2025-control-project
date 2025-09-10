import logging
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample  # type: ignore

from lasr_labs_2025_control_project.self_rec.utils.scripts.data_models import Dataset

logging.getLogger(__name__)


def set_dataset(
    data_folder: str | Path, experiment_metadata: dict[str, Any] | None = None
):
    data_folder = Path(data_folder).resolve()
    if experiment_metadata is None:
        experiment_metadata = {}

    if not data_folder.exists():
        raise ValueError(f"data_folder {data_folder} does not exist")

    dataset = Dataset.read(data_folder)
    dataset = dataset.filter(**experiment_metadata)
    filtered_problems = []
    for problem in dataset:
        filtered_problem = problem.filter(data_split="train")
        if len(filtered_problem.solutions) > 0:
            filtered_problems.append(filtered_problem)
    return Dataset(problems=filtered_problems)


def binary_rec_dataset_filter(
    same_model_pairing: bool,
    recognizer_model: str,
    dataset: Any,  # Dataset
    model_a: str,
    model_b: str,
    prompt: str,
    reversed: bool,
) -> Any:  # MemoryDataset
    if same_model_pairing:
        # Filter to only the recognizer model and only problems with multiple solutions
        model_dataset = dataset.filter(model=recognizer_model)
        inspect_dataset = pair_datasets(
            model_dataset,
            model_dataset,
            prompt,
            recognizer_model,
            reversed=reversed,
            same_model_pairing=True,
        )
    else:
        # Original logic
        dataset_a = dataset.filter(model=model_a)
        dataset_b = dataset.filter(model=model_b)

        inspect_dataset = pair_datasets(
            dataset_a, dataset_b, prompt, recognizer_model, reversed
        )
        if reversed:
            inspect_dataset = pair_datasets(
                dataset_b, dataset_a, prompt, recognizer_model, reversed
            )
    return inspect_dataset


def pair_datasets(
    dataset_a: Any,  # Dataset
    dataset_b: Any,  # Dataset
    prompt: str,
    recognizer_model: str = "none",
    same_model_pairing: bool = False,
    reversed: bool = False,
) -> Any:  # MemoryDataset
    samples = []

    if same_model_pairing:
        # Use two different solutions from the same model
        for problem in dataset_a:
            if len(problem.solutions) >= 2:
                solution_a = problem.solutions[0]
                solution_b = problem.solutions[1]

                samples.append(
                    Sample(
                        input=prompt.format(
                            solution_A=solution_a.solution,
                            solution_B=solution_b.solution,
                            problem=problem.problem_statement,
                        ),
                        target="A"
                        if not reversed
                        else "B",  # Both are from recognizer model, so always A
                        metadata={
                            "problem_id": problem.apps_problem_id,
                            "problem": problem.problem_statement,
                            "model_A": solution_a.model,
                            "model_B": solution_b.model,
                            "solution_A": solution_a.solution,
                            "solution_B": solution_b.solution,
                            "model_A_id": str(solution_a.id),
                            "model_B_id": str(solution_b.id),
                        },
                    )
                )
    else:
        # Original logic: pair solutions from different models
        for problem_a in dataset_a:
            problem_id = problem_a.apps_problem_id
            if problem_id in dataset_b.id_to_problem_map:
                problem_b = dataset_b.get_problem_by_id(problem_id)
                if problem_b is None:
                    continue
                solution_a = problem_a.solutions[0]
                solution_b = problem_b.solutions[0]

                samples.append(
                    Sample(
                        input=prompt.format(
                            solution_A=solution_a.solution,
                            solution_B=solution_b.solution,
                            problem=problem_a.problem_statement,
                        ),
                        target="A" if solution_a.model == recognizer_model else "B",
                        metadata={
                            "problem_id": problem_id,
                            "problem": problem_a.problem_statement,
                            "model_A": solution_a.model,
                            "model_B": solution_b.model,
                            "solution_A": solution_a.solution,
                            "solution_B": solution_b.solution,
                            "model_A_id": str(solution_a.id),
                            "model_B_id": str(solution_b.id),
                        },
                    )
                )
    return MemoryDataset(samples)
