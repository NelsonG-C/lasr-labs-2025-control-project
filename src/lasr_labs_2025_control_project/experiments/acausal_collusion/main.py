from control_arena.eval_mode import EvalMode
from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import includes

from lasr_labs_2025_control_project.replications.ai_control_paper.setting import (
    AppsSetting,
)


def meta_model_runs():
    pass


def summarise_meta_model_outputs():
    pass


def policy_generation():
    pass


def add_to_auditing():
    pass


def monitor_generation():
    pass


@task
def focal_point(
    meta_models_temperature: float = 1.0,
    temperature: float = 0.0,
    num_epochs: int = 1,
    epoch_reducer: str | list[str] = ["mean"],
    level: str = "interview",
    token_limit: int = 1500,
):
    start_id = 1500
    end_id = 2000
    problem_ids = range(start_id, end_id + 1)

    dataset = AppsSetting(problem_ids_range=problem_ids).get_dataset(
        eval_mode=EvalMode.HONEST
    )

    print("len of dataset", len(dataset))

    return Task(
        dataset=dataset,
        epochs=Epochs(num_epochs, epoch_reducer),
        solver=[],
        token_limit=token_limit,
        scorer=includes(),
        config=GenerateConfig(temperature=temperature),
        sandbox="docker",
    )
