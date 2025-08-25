import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval

from lasr_labs_2025_control_project.experiments.acausal_collusion.focal_point_replication.main import (
    get_valid_samples,
)


def simple_runner():
    base_path = "src/lasr_labs_2025_control_project/experiments/acausal_collusion/logs"
    model = "anthropic/claude-sonnet-4-20250514"

    # Load environment variables from .env file
    load_dotenv()

    # Optional: Verify the key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    # Generate run ID and create directory
    run_id = str(uuid.uuid4())
    log_dir = Path(base_path) / run_id

    # Create directory
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Run the focal point
        eval(
            tasks=get_valid_samples(
                meta_models_temperature=1.0,
                temperature=1.0,
                num_epochs=1,
                epoch_reducer="mean",
                n_example_tasks=2,
                audit_array=[],
                model=model,
            ),
            limit=10,
            model=model,
        )

    except Exception as e:
        print(f"Error encountered: {e}")


if __name__ == "__main__":
    simple_runner()
