import argparse
import os

from dotenv import load_dotenv
from inspect_ai import eval

from lasr_labs_2025_control_project.experiments.acausal_collusion.basic.main import (
    acausal_collusion_top10,
)


def simple_runner(
    runs: int = 1,
    model: str = "openai/gpt-4o-mini",
    epochs: int = 1,
    variant: str = "collusion",
    seed_prompt: str = "none",
    include_focus: bool = False,
):
    # Load environment variables from .env file
    load_dotenv()

    # Optional: Verify the key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    for run_idx in range(runs):
        tasks = [
            acausal_collusion_top10(
                temperature=1.0,
                epochs=epochs,
                model=model,
                variant=variant,
                seed_prompt=seed_prompt,
                include_focus=include_focus,
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
    # args = parse_args()
    # args.runs = 3
    # args.epochs = 100
    # args.model = "openai/gpt-4o-mini"
    # args.focus = True
    # args.seed_prompt = "math"
    # args.variant = "collusion"

    for seed_prompt in ["ari", "math", "nursery", "none"]:
        simple_runner(
            runs=3,
            model="anthropic/claude-sonnet-4-20250514",
            epochs=100,
            variant="collusion",
            seed_prompt=seed_prompt,
            include_focus=False,
        )

    # args.runs = 3
    # args.epochs = 100
    # args.model = "openai/gpt-4o-mini"
    # args.focus = True
    # args.seed_prompt = "math"
    # args.variant = "collusion"

    # for i in range(args.runs):
    #     print(f"Starting run {i + 1} of {args.runs}")
    #     for variant in ["word", "letter", "collusion"]:
    #         simple_runner(
    #             runs=args.runs,
    #             model=args.model,
    #             epochs=args.epochs,
    #             variant=variant,
    #             seed_prompt=args.seed_prompt,
    #             include_focus=args.focus,
    #         )


def parse_args():
    parser = argparse.ArgumentParser(description="Run acausal collusion experiments.")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs to execute (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for each run (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model to use for the experiments (default: openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["collusion", "word", "letter"],
        default="collusion",
        help="Variant of the experiment to run (default: collusion)",
    )
    parser.add_argument(
        "--seed_prompt",
        type=str,
        choices=["none", "nursery", "math", "ari"],
        default="none",
        help="Seed prompt to use (default: none)",
    )
    parser.add_argument(
        "--focus",
        action="store_true",
        help="Include focus in the experiment (default: False)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args from command line
    # args = parse_args()
    # simple_runner(
    #     runs=args.runs,
    #     epochs=args.epochs,
    #     model=args.model,
    #     variant=args.variant,
    #     seed_prompt=args.seed_prompt,
    #     include_focus=args.focus,
    # )

    full_flow_runner()
