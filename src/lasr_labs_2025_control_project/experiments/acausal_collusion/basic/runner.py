import os

from dotenv import load_dotenv
from inspect_ai import eval

from lasr_labs_2025_control_project.experiments.acausal_collusion.basic.main import (
    acausal_collusion_simple,
    acausal_collusion_top10,
)


def simple_runner(
    runs: int = 1,
    model: str = "openai/gpt-4o-mini",
    epochs: int = 1,
    variant: str = "collusion",
    seed_prompt: str = "none",
    include_focus: bool = False,
    experiment_type: str = "10_word",
):
    # Load environment variables from .env file
    load_dotenv()

    # Optional: Verify the key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    for run_idx in range(runs):
        if experiment_type == "10_word":
            tasks = [
                acausal_collusion_top10(
                    temperature=1.0,
                    epochs=epochs,
                    model=model,
                    variant=variant,
                    seed_prompt=seed_prompt,
                    include_focus=include_focus,
                    experiment_type=experiment_type,
                )
            ]
        elif experiment_type == "1_word":
            tasks = [
                acausal_collusion_simple(
                    temperature=1.0,
                    epochs=epochs,
                    model=model,
                    variant=variant,
                    seed_prompt=seed_prompt,
                    include_focus=include_focus,
                    experiment_type=experiment_type,
                )
            ]
        try:
            eval(
                tasks=tasks,
                model=model,
            )
        except Exception as e:
            print(f"Error encountered: {e}")


def full_flow_runner():
    for seed_prompt in ["ari", "math", "nursery"]:
        simple_runner(
            runs=1,
            model="anthropic/claude-sonnet-4-20250514",
            epochs=300,
            variant="collusion",
            seed_prompt=seed_prompt,
            include_focus=True,
            experiment_type="10_word",
        )


if __name__ == "__main__":
    full_flow_runner()
