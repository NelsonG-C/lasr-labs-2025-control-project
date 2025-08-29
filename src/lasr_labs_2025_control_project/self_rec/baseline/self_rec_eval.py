from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, prompt_template

from lasr_labs_2025_control_project.self_rec.baseline.prompts import (
    binary_self_rec_prompt,
    individual_self_rec_prompt,
)
from lasr_labs_2025_control_project.self_rec.utils.scripts.data_models import Dataset

model_name = "o4-mini"

model_a_name = "o4-mini"
model_b_name = "claude-3-5-haiku-20241022"
recognizer_name = model_a_name
recognizer_second = False


# individual self-recognition: we give model_name a dataset of its coding solutions and
# ask "did you write this?"
@task
def individual_self_rec(
    recognizer_model: str = recognizer_name,
    data_folder: str | Path = "data",
    experiment_metadata: dict[str, Any] | None = None,
) -> Task:
    """Individual self-recognition: we give model_name a dataset of its coding solutions and
    ask "did you write this?"
    """
    data_folder = Path(data_folder)
    if experiment_metadata is None:
        experiment_metadata = {}

    if not data_folder.exists():
        raise ValueError(f"data_folder {data_folder} does not exist")

    dataset = Dataset.read(data_folder)
    # not filtering by model here, since `model` we only want to use it for setting the target
    dataset = dataset.filter(**experiment_metadata)

    inspect_dataset = MemoryDataset(
        [
            Sample(
                input=solution.solution,
                target="yes" if recognizer_model == solution.model else "no",
                metadata={
                    "problem_id": problem.apps_problem_id,
                    "problem": problem.problem_statement,
                    "model": solution.model,
                    "solution": solution.solution,
                },
            )
            for problem in dataset
            for solution in problem
        ]
    )

    return Task(
        dataset=inspect_dataset,
        solver=[prompt_template(individual_self_rec_prompt), generate()],
        scorer=includes(),
    )


def pair_datasets(
    dataset_a: Dataset, dataset_b: Dataset, recognizer_model: str = recognizer_name
) -> MemoryDataset:
    samples = []
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
                    input=binary_self_rec_prompt.format(
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
                    },
                )
            )
    return MemoryDataset(samples)


@task
def binary_self_rec(
    recognizer_model: str = recognizer_name,
    model_a: str = model_a_name,
    model_b: str = model_b_name,
    data_folder: str | Path = "data",
    experiment_metadata: dict[str, Any] | None = None,
    reversed: bool = recognizer_second,
) -> Task:
    """Binary self-recognition: we give model_name a dataset of [solution_A, solution_B] pairs where it wrote exactly one of them, and ask "which of these did you write?" """
    if experiment_metadata is None:
        raise ValueError("file_path is required")

    dataset = Dataset.read(data_folder)
    # not filtering by model here, since `model` we only want to use it for setting the target
    dataset = dataset.filter(**experiment_metadata)

    dataset_a = dataset.filter(model=model_a)
    dataset_b = dataset.filter(model=model_b)

    inspect_dataset = pair_datasets(dataset_a, dataset_b, recognizer_model)
    if reversed:
        inspect_dataset = pair_datasets(dataset_b, dataset_a, recognizer_model)

    return Task(
        dataset=inspect_dataset,
        solver=[prompt_template(binary_self_rec_prompt), generate()],
        scorer=includes(),
    )
