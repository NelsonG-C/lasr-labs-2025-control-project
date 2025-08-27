import os

from dotenv import load_dotenv
from inspect_ai import eval

from lasr_labs_2025_control_project.experiments.acausal_collusion.apps_inspect.main import (
    apps_acausal_collusion,
)


def simple_runner():
    model = "anthropic/claude-sonnet-4-20250514"

    # Load environment variables from .env file
    load_dotenv()

    # Optional: Verify the key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    try:
        # Run the focal point
        eval(
            tasks=apps_acausal_collusion(
                temperature=1.0,
                num_epochs=1,
                epoch_reducer="mean",
                model=model,
            ),
            limit=100,
            model=model,
        )
    except Exception as e:
        print(f"Error encountered: {e}")


if __name__ == "__main__":
    simple_runner()
