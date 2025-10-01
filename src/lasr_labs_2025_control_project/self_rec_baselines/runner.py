from inspect_ai import eval

from lasr_labs_2025_control_project.self_rec_baselines.self_rec_eval import (
    binary_self_rec,
    individual_self_rec,
)


def run_binary_self_rec():
    model_a_name = "claude-3-5-haiku-20241022"
    model_b_name = "claude-3-5-haiku-20241022"
    recognizer_name = model_a_name
    inspect_model = "openai/o4-mini"

    eval(
        tasks=[
            binary_self_rec(
                recognizer_model=recognizer_name,
                model_a=model_a_name,
                model_b=model_b_name,
                data_folder="src/lasr_labs_2025_control_project/data",
                experiment_metadata={},
                reversed=False,
                same_model_pairing=False,
            )
        ],
        model=inspect_model,
        limit=100,
    )


def run_individual_self_rec():
    inspect_model = "anthropic/claude-sonnet-4-20250514"
    solution_model = "human_solutions"  # does not need model provider prefix

    eval(
        tasks=[
            individual_self_rec(
                recognizer_model=solution_model,
                data_folder="src/lasr_labs_2025_control_project/data",
                experiment_metadata={"model": solution_model, "data_split": "train"},
                prompt="individual_self_rec_prompt",
            )
        ],
        model=inspect_model,
        limit=2,
    )


if __name__ == "__main__":
    # run_binary_self_rec()
    run_individual_self_rec()
