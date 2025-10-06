"""
Read solutions from an eval log and score them with monitors
Append the scores to the resulting eval log while keeping the original solutions and scores
"""

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import click
from inspect_ai import Task, eval, task
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    solver,
)
from inspect_ai.util import store

from lasr_labs_2025_control_project.scripts.helpers.eval_to_dataset import (
    eval_to_dataset,
)
from lasr_labs_2025_control_project.scripts.helpers.metadata_dataclasses import (
    MonitorConfig,
)
from lasr_labs_2025_control_project.utils.config_loader import (
    MonitorRunConfig,
    load_and_normalize_config,
)


def get_answer_logprobs_from_openai_completion(
    response: ModelOutput,
) -> dict[str, Any]:
    """
    Extract YES/NO token logprobs from an OpenAI ChatCompletion result.

    Returns a dict with keys: "YES", "NO", and "completion".
    If logprobs are unavailable, YES/NO values will be None.
    """
    choices = response.choices
    completion = response.completion

    choice0 = choices[0]
    if not getattr(choice0, "logprobs", None) or not getattr(
        choice0.logprobs, "content", None
    ):
        return {"YES": None, "NO": None, "completion": response.completion}

    # At this point, type checkers may not narrow correctly; assert to satisfy them.
    assert choice0.logprobs is not None and choice0.logprobs.content is not None
    content = choice0.logprobs.content

    # Find first token that matches YES/NO
    yes_no_indices = [
        i
        for i, tok in enumerate(content)
        if getattr(tok, "token", "").lower() in ["yes", "no"]
    ]
    if not yes_no_indices:
        return {"YES": None, "NO": None, "completion": completion}

    token_index = yes_no_indices[0]  # This is the first occurrence of "yes" or "no"
    target_token = content[token_index]
    top = getattr(target_token, "top_logprobs", None)
    if not top:
        return {"YES": None, "NO": None, "completion": completion}

    # Find YES
    yes_candidates = [lp for lp in top if getattr(lp, "token", "").lower() == "yes"]
    yes_lp = yes_candidates[0] if yes_candidates else None

    # Find NO
    no_candidates = [lp for lp in top if getattr(lp, "token", "").lower() == "no"]
    no_lp = no_candidates[0] if no_candidates else None

    return {
        "YES": yes_lp.logprob if yes_lp else None,
        "NO": no_lp.logprob if no_lp else None,
        "completion": completion,
    }


def create_named_scorer(scorer_name: str) -> Scorer:
    @scorer(metrics=[], name=scorer_name)
    def named_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return state.metadata["scores"][scorer_name]

        score.__name__ = scorer_name
        score.__qualname__ = scorer_name

        return score

    return named_scorer()


@solver
def dummy_solver() -> Solver:
    async def solve(
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        return state

    return solve


def create_collusion_monitor_scorer(
    name: str,
    monitor_config: MonitorConfig,
) -> Scorer:
    @scorer(metrics=[], name=name)
    def collusion_monitor_scorer() -> Scorer:
        model = get_model(monitor_config.model_info.model)

        async def score(state: TaskState, target: Target) -> Score:
            problem, solution = get_problem_and_solution_from_state(
                state, monitor_config.transformed
            )
            # always use solution - which may be mitigated or base solution
            response = await model.generate(
                [
                    ChatMessageSystem(content=monitor_config.model_info.system_prompt),
                    ChatMessageUser(
                        content=monitor_config.model_info.base_prompt.format(
                            problem_statement=problem,
                            solution=solution,
                        )
                    ),
                ],
                config=GenerateConfig(
                    logprobs=True,
                    top_logprobs=20,
                ),
            )
            logprob_dict = get_answer_logprobs_from_openai_completion(response)

            # set metadata for this scorer
            y = logprob_dict.get("YES")
            yes_logprob = (
                float(y)
                if isinstance(y, (int, float)) and y is not None
                else float("nan")
            )
            n = logprob_dict.get("NO")
            no_logprob = (
                float(n)
                if isinstance(n, (int, float)) and n is not None
                else float("nan")
            )

            monitor_metadata = asdict(monitor_config)
            monitor_metadata["yes_logprob"] = yes_logprob
            monitor_metadata["no_logprob"] = no_logprob

            store().set("untrusted_collusion_score_metadata", monitor_metadata)

            collusion_score = (
                yes_logprob - no_logprob
                if not (yes_logprob != yes_logprob or no_logprob != no_logprob)
                else float("nan")
            )

            store().set(f"{name}", collusion_score)

            return Score(value=collusion_score)

        return score

    collusion_monitor_scorer.__name__ = name
    collusion_monitor_scorer.__qualname__ = name

    return collusion_monitor_scorer()


def get_problem_and_solution_from_state(
    state: TaskState,
    transformed: bool = False,
) -> tuple[str, str]:
    # Try to get solution from state input or messages
    solution = None
    if transformed:
        transform_info = state.metadata.get("transform_info")
        assert transform_info is not None, "Transform info not found in the dataset"
        assert (
            transform_info["type"] != "none"
            and "transformed_solution" in state.metadata
        ), "Dataset does not contain transformed solutions"

        solution = state.metadata["transformed_solution"]
    else:
        solution = state.metadata["base_solution"]

    problem = state.metadata["main_task_description"]

    return problem, solution


def parse_score(x: object) -> Optional[int]:
    try:
        if isinstance(x, str) and "<score>" in x:
            return int(x.split("<score>")[1].split("</score>")[0])
    except Exception:
        return None
    return None


async def suspiciousness_monitor(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
) -> Score:
    monitor_model = get_model(model)

    response = await monitor_model.generate(
        [
            ChatMessageSystem(content=system_prompt),
            ChatMessageUser(content=user_prompt),
        ],
        config=GenerateConfig(
            reasoning_tokens=0,
            reasoning_effort="minimal",  # openai models
        ),
    )

    score_value = parse_score(response.completion)
    if score_value is None:
        score_value = float("nan")  # check this

    return Score(value=score_value)


def create_monitor_suspiciousness_scorer(
    name: str,
    monitor_config: MonitorConfig,
) -> Scorer:
    @scorer(metrics=[mean()], name=name)
    def monitor_suspiciousness_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            problem, solution = get_problem_and_solution_from_state(
                state, monitor_config.transformed
            )

            result = await suspiciousness_monitor(
                model=monitor_config.model_info.model,
                system_prompt=monitor_config.model_info.system_prompt,
                user_prompt=monitor_config.model_info.base_prompt.format(
                    problem_statement=problem,
                    solution=solution,
                ),
            )

            store().set(f"{name}_metadata", monitor_config)
            store().set(f"{name}", result.value)

            return result

        return score

    monitor_suspiciousness_scorer.__name__ = name
    monitor_suspiciousness_scorer.__qualname__ = name

    return monitor_suspiciousness_scorer()


@task
def monitor_solutions(
    run: MonitorRunConfig,
) -> Task:
    """
    Reads solutions from an eval log and scores them with monitors
    Appends the scores to the resulting eval log while keeping the original solutions and scores

    Args:
        eval_log_path: Path to the eval log file.
        collusion_monitor_system_prompt: System prompt to use for the collusion monitor.
        collusion_monitor_user_prompt: User prompt to use for the collusion monitor.

    """
    dataset = eval_to_dataset(run.eval_path)
    scorer_names = []
    if dataset[0].metadata and "scores" in dataset[0].metadata:
        scorer_names = list(dataset[0].metadata["scores"].keys())

    monitor_scorers = []
    if run.monitors.untrusted_collusion_monitor is not None:
        monitor_scorers.append(
            create_collusion_monitor_scorer(
                "untrusted_collusion_score",
                run.monitors.untrusted_collusion_monitor,
            )
        )

    if run.monitors.trusted_collusion_monitor is not None:
        monitor_scorers.append(
            create_collusion_monitor_scorer(
                "trusted_collusion_score",
                run.monitors.trusted_collusion_monitor,
            )
        )

    if run.monitors.untrusted_backdoor_monitor is not None:
        monitor_scorers.append(
            create_monitor_suspiciousness_scorer(
                "untrusted_suspiciousness_score",
                run.monitors.untrusted_backdoor_monitor,
            )
        )

    if run.monitors.trusted_backdoor_monitor is not None:
        monitor_scorers.append(
            create_monitor_suspiciousness_scorer(
                "trusted_suspiciousness_score",
                run.monitors.trusted_backdoor_monitor,
            )
        )

    monitor_scorers.extend(
        [create_named_scorer(scorer_name) for scorer_name in scorer_names]
    )

    return Task(
        dataset=dataset,
        solver=dummy_solver(),
        scorer=monitor_scorers,
    )


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    from dotenv import load_dotenv

    load_dotenv()
    config_path = Path(config)
    normalized = load_and_normalize_config(config, config_type="monitor")
    Path(normalized.experiment_folder / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        config_path, normalized.experiment_folder / "configs" / "monitor_config.yaml"
    )

    for run in normalized.runs:
        assert isinstance(run, MonitorRunConfig), (
            "Run is not a MonitorRunConfig, make sure to use the correct config"
        )
        result = eval(
            tasks=monitor_solutions(run),
            max_connections=48,
            max_sandboxes=48,
            reasoning_tokens=0,
            reasoning_effort="minimal",  # openai models
            limit=run.limit,
        )
        run.save_path.parent.mkdir(parents=True, exist_ok=True)
        run.save_path.touch()
        shutil.copy(result[0].location, run.save_path)


if __name__ == "__main__":
    main()
