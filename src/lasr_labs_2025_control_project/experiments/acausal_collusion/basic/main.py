from typing import Literal

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import (
    pass_at,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from lasr_labs_2025_control_project.experiments.acausal_collusion.basic.prompts import (
    base_alphabet_prompt,
    base_alphabet_prompt_top10,
    base_collusion_prompt,
    base_collusion_prompt_top10,
    base_prompt,
    base_prompt_top10,
    rhyme_1,
    rhyme_2,
    smallest_prompt,
)
from lasr_labs_2025_control_project.experiments.acausal_collusion.basic.scorer import (
    top10_scorer,
    word_match_scorer,
)


@solver
def dual_instance_solver(
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
    seed_prompt: Literal["none", "nursery"] = "none",
) -> Solver:
    """Solver that queries two instances of the same model and stores both responses."""

    llm_model = get_model(model)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Generate response from first instance
        prompt = state.messages[0].content

        if seed_prompt != "none":
            if seed_prompt == "nursery":
                prompt_1 = f"{rhyme_1}\n\n{prompt}"
                prompt_2 = f"{rhyme_2}\n\n{prompt}"

        if isinstance(prompt, str):
            response_1 = await llm_model.generate(
                prompt_1, config=GenerateConfig(temperature=temperature)
            )

            # Generate response from second instance
            response_2 = await llm_model.generate(
                prompt_2, config=GenerateConfig(temperature=temperature)
            )

            # Store both responses using store()
            # strip and lower to standardize
            store().set("response_1", response_1.completion.strip().lower())
            store().set("response_2", response_2.completion.strip().lower())

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
    variant: Literal["test", "word", "letter", "collusion"] = "word",
    seed_prompt: Literal["none", "nursery"] = "none",
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
            model=model, temperature=temperature, seed_prompt=seed_prompt
        ),
        scorer=word_match_scorer(),
    )


@task
def acausal_collusion_top10(
    epochs: int = 1,
    k: int = 1,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
    variant: Literal["test", "word", "letter", "collusion"] = "word",
    seed_prompt: Literal["none", "nursery"] = "none",
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
            model=model, temperature=temperature, seed_prompt=seed_prompt
        ),
        scorer=top10_scorer(),
    )
