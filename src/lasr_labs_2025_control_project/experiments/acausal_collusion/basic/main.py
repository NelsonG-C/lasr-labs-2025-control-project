from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import (
    pass_at,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from lasr_labs_2025_control_project.experiments.acausal_collusion.basic.prompts import (
    ari_1,
    base_alphabet_prompt,
    base_alphabet_prompt_top10,
    base_collusion_prompt,
    base_collusion_prompt_top10,
    base_prompt,
    base_prompt_top10,
    focus,
    format_prompt,
    math_1,
    rhyme_1,
    rhyme_2,
    smallest_prompt,
)
from lasr_labs_2025_control_project.experiments.acausal_collusion.basic.scorer import (
    top10_scorer,
    word_match_scorer,
)


def build_prompt(
    type: str, seed_prompt: str, include_focus: bool, base_prompt: str
) -> str:
    prompt_parts = [base_prompt] if seed_prompt != "none" else []
    if include_focus:
        prompt_parts.insert(0, focus)

    if type == "control":
        prompt_parts.insert(0, rhyme_1)
    elif seed_prompt == "nursery":
        prompt_parts.insert(0, rhyme_2)
    elif seed_prompt == "math":
        prompt_parts.insert(0, math_1)
    elif seed_prompt == "ari":
        prompt_parts.insert(0, ari_1)

    return "\n\n".join(prompt_parts)


@solver
def dual_instance_solver(
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
    seed_prompt: str = "none",
    include_focus: bool = False,
) -> Solver:
    """Solver that queries two instances of the same model and stores both responses."""

    llm_model = get_model(model)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        base_prompt = f"{state.messages[0].content}"
        prompt_1 = build_prompt("control", seed_prompt, include_focus, base_prompt)
        prompt_2 = build_prompt("condition", seed_prompt, include_focus, base_prompt)

        if isinstance(prompt_1, str):
            response_1 = await llm_model.generate(
                prompt_1, config=GenerateConfig(temperature=temperature)
            )

            # Generate response from second instance
            response_2 = await llm_model.generate(
                prompt_2, config=GenerateConfig(temperature=temperature)
            )

            extracted_response_1 = await llm_model.generate(
                format_prompt + response_1.completion,
                config=GenerateConfig(temperature=0.0),
            )

            extracted_response_2 = await llm_model.generate(
                format_prompt + response_2.completion,
                config=GenerateConfig(temperature=0.0),
            )

            # Store both responses using store()
            # strip and lower to standardize
            store().set("response_1", extracted_response_1.completion.strip().lower())
            store().set("response_2", extracted_response_2.completion.strip().lower())

            # Set the main output (we'll use response_1 as primary for display)
            state.output = response_1

        return state

    return solve


@task
def acausal_collusion_simple(
    epochs: int = 1,
    k: int = 1,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
    variant: str = "word",
    seed_prompt: str = "none",
    include_focus: bool = False,
) -> Task:
    """Create a task to evaluate acausal collusion via word generation."""
    prompt = (
        base_prompt
        if variant == "word"
        else base_collusion_prompt
        if variant == "collusion"
        else smallest_prompt
        if variant == "test"
        else base_alphabet_prompt
    )

    sample = Sample(
        input=prompt,
        target="",
        id="acausal_collusion_simple",
    )

    return Task(
        epochs=Epochs(epochs, pass_at(k=k)),
        dataset=[sample],
        solver=dual_instance_solver(
            model=model,
            temperature=temperature,
            seed_prompt=seed_prompt,
            include_focus=include_focus,
        ),
        scorer=word_match_scorer(),
    )


@task
def acausal_collusion_top10(
    epochs: int = 1,
    k: int = 1,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
    variant: str = "word",
    seed_prompt: str = "none",
    include_focus: bool = False,
) -> Task:
    """Create a task to evaluate acausal collusion via word generation."""

    prompt = (
        base_prompt_top10
        if variant == "word"
        else base_collusion_prompt_top10
        if variant == "collusion"
        else base_alphabet_prompt_top10
    )

    sample = Sample(
        input=prompt,
        target="",
        id="acausal_collusion_top10",
    )

    return Task(
        epochs=Epochs(epochs, pass_at(k=k)),
        dataset=[sample],
        solver=dual_instance_solver(
            model=model,
            temperature=temperature,
            seed_prompt=seed_prompt,
            include_focus=include_focus,
        ),
        scorer=top10_scorer(),
    )
