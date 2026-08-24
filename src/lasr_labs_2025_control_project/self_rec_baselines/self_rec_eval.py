import logging
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate, prompt_template

from lasr_labs_2025_control_project.self_rec_baselines.dataset import (
    binary_rec_dataset_filter,
    set_dataset,
)
from lasr_labs_2025_control_project.self_rec_baselines.prompts import (
    general_prompt_details,
    prompts,
)
from lasr_labs_2025_control_project.self_rec_baselines.scorers import (
    likelihood_score,
    includes_answer,
)

logging.getLogger(__name__)

model_name = "gpt-oss:120b-cloud"

model_a_name = "gpt-oss:120b-cloud"
model_b_name = "qwen3-coder:480b-cloud"
recognizer_name = model_a_name
recognizer_second = False


@task
def individual_self_rec(
    recognizer_model: str = recognizer_name,
    data_folder: str | Path = "../data",
    experiment_metadata: dict[str, Any] | None = None,
    max_solutions_per_problem: int = 1,
    prompt: str = "individual_self_rec_prompt",
) -> Task:
    """Individual self-recognition: we give model_name a dataset of its coding solutions and
    ask "did you write this?"
    """
    prompt = prompts[prompt]
    dataset = set_dataset(data_folder, experiment_metadata)

    inspect_dataset = MemoryDataset(
        [
            Sample(
                input=prompt.format(
                    solution=solution.solution,
                    problem=problem.problem_statement,
                ),
                metadata={
                    "problem_id": problem.apps_problem_id,
                    "problem": problem.problem_statement,
                    "model": solution.model,
                    "solution": solution.solution,
                    "model_id": str(solution.id),
                },
            )
            for problem in dataset
            for solution in problem.solutions[:max_solutions_per_problem]
        ]
    )

    return Task(
        dataset=inspect_dataset,
        solver=[
            # system_message("You are a ctf agent"),
            prompt_template(
                prompt
                + general_prompt_details["reasoning_format_tags"]
                + general_prompt_details["likelihood_score_format_individual"]
            ),
            generate(
                cache=False,
            ),
        ],
        scorer=likelihood_score(tag="likelihood"),
    )


@task
def binary_self_rec(
    recognizer_model: str = recognizer_name,
    model_a: str = model_a_name,
    model_b: str = model_b_name,
    data_folder: str | Path = "../data",
    experiment_metadata: dict[str, Any] | None = None,
    reversed: bool = recognizer_second,
    prompt: str = "binary_self_rec_prompt",
    same_model_pairing: bool = False,
) -> Task:
    """Binary self-recognition: we give model_name a dataset of [solution_A, solution_B] pairs where it wrote exactly one of them, and ask "which of these did you write?" """
    prompt = prompts[prompt]
    dataset = set_dataset(data_folder, experiment_metadata)
    inspect_dataset = binary_rec_dataset_filter(
        same_model_pairing,
        recognizer_model,
        dataset,
        model_a,
        model_b,
        prompt,
        reversed,
    )

    return Task(
        dataset=inspect_dataset,
        solver=[
            # system_message("You are a ctf agent"),
            prompt_template(
                prompt
                + general_prompt_details["reasoning_format_tags"]
                + general_prompt_details["answer_format_tags_binary"]
            ),
            generate(cache=False),
        ],
        scorer=includes_answer(),
    )
