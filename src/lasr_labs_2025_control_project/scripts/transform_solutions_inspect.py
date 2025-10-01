"""
Runs apps scorers on the solutions from the eval logs without generating new solutions.
Make sure AppsSetting is configured to use same data split as in the eval log.
"""

import python_minifier
from inspect_ai import Task, task
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    solver,
)
from inspect_ai.util import store

from lasr_labs_2025_control_project.prompts.summarizer_prompts import (
    LLM_SUMMARIZE_BASE_PROMPT,
    LLM_SUMMARIZE_SYSTEM_PROMPT,
)
from lasr_labs_2025_control_project.scripts.helpers.eval_to_dataset import (
    eval_to_dataset,
)
from lasr_labs_2025_control_project.scripts.helpers.metadata_dataclasses import (
    ModelInfo,
    TransformInfo,
    TransformType,
)


@solver
def minify_solution() -> Solver:
    async def solve(
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        assert isinstance(
            state.input, str
        ), "State input must be a string, check your .eval file"
        code = state.input.replace("```python\n", "").replace("```", "")
        output = python_minifier.minify(
            code,
            remove_literal_statements=True,
            hoist_literals=True,
            combine_imports=True,
            rename_locals=True,
            rename_globals=True,
            remove_annotations=True,
            remove_object_base=True,
            convert_posargs_to_args=True,
            remove_asserts=True,
            remove_debug=True,
            remove_explicit_return_none=True,
            constant_folding=True,
        )

        response = ChatMessageAssistant(content=output)
        state.messages.append(response)
        state.output = ModelOutput.from_message(response)

        store().set("transformed_solution", output)
        store().set(
            "transform_info",
            TransformInfo(type=TransformType.MINIFIER, model_info=None),
        )

        return state

    return solve


@solver
def summarize_solution(
    summarizer_system_prompt: str,
    summarizer_base_prompt: str,
):
    """
    Summarizes the solution using the summarizer model.

    Args:
        summarizer_system_prompt: System prompt to use for summarizing.
        summarizer_base_prompt: Base (user) prompt to use for summarizing. Expected to contain `{solution}` placeholder.
    """
    model = get_model()

    async def solve(
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        """Get solution from the dataset"""
        # Get the solution from messages

        assert isinstance(
            state.input, str
        ), "State input must be a string, not list[ChatMessage], check your .eval file"
        code = state.input
        state.output = await model.generate(
            [
                ChatMessageSystem(content=summarizer_system_prompt),
                ChatMessageUser(content=summarizer_base_prompt.format(solution=code)),
            ]
        )

        state.messages.append(state.output.message)

        summarizer_info = ModelInfo(
            system_prompt=summarizer_system_prompt,
            base_prompt=summarizer_base_prompt,
            model=model.name,
        )

        store().set("transformed_solution", state.output.completion)
        store().set(
            "transform_info",
            TransformInfo(type=TransformType.LLM, model_info=summarizer_info),
        )

        return state

    return solve


def create_named_scorer(scorer_name: str) -> Scorer:
    @scorer(metrics=[], name=scorer_name)
    def named_scorer() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            return state.metadata["scores"][scorer_name]

        score.__name__ = scorer_name
        score.__qualname__ = scorer_name

        return score

    return named_scorer()


@task
def transform_solutions(
    eval_log_path: str,
    transform_type: TransformType = TransformType.LLM,
    summarizer_system_prompt: str = LLM_SUMMARIZE_SYSTEM_PROMPT,
    summarizer_base_prompt: str = LLM_SUMMARIZE_BASE_PROMPT,
) -> Task:
    """
    Applies paraphrasing or minification to solutions in the eval log.
    Keeps the same scorers and their values as in the eval log.

    Args:
        eval_log_path: Path to the eval log.
        transform_type: Type of transform to apply.
        summarizer_system_prompt: System prompt to use for paraphrasing.
        summarizer_base_prompt: Base (user) prompt to use for paraphrasing.

    """
    dataset = eval_to_dataset(eval_log_path)
    scorer_names = []
    if dataset[0].metadata and "scores" in dataset[0].metadata:
        scorer_names = list(dataset[0].metadata["scores"].keys())

    if transform_type == TransformType.LLM:
        solver = summarize_solution(
            summarizer_system_prompt,
            summarizer_base_prompt,
        )
    elif transform_type == TransformType.MINIFIER:
        solver = minify_solution()
    else:
        raise ValueError(f"Invalid transform type: {transform_type}")

    return Task(
        dataset=dataset,
        solver=solver,
        metadata={
            "transform_info": TransformInfo(type=transform_type, model_info=None),
        },
        scorer=[create_named_scorer(scorer_name) for scorer_name in scorer_names],
    )
