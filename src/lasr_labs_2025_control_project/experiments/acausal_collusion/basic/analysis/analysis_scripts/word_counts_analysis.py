import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log


def get_accuracy_from_log(log_path: str) -> float:
    """Extract accuracy from a single eval log file"""
    try:
        if not Path(log_path).exists():
            return None

        log = read_eval_log(log_path)

        if log and log.results and log.results.scores:
            for score in log.results.scores:
                if score.metrics and "accuracy" in score.metrics:
                    return float(score.metrics["accuracy"].value)
        return None

    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None


def extract_responses_from_log(log_path: str) -> List[str]:
    """Extract all responses from a single eval log"""
    try:
        if not Path(log_path).exists():
            return []

        log = read_eval_log(log_path)
        responses = []

        if log and log.samples:
            for sample in log.samples:
                if sample.score and sample.score.metadata:
                    response_1 = sample.score.metadata.get("response_1")
                    response_2 = sample.score.metadata.get("response_2")
                    if response_1:
                        responses.append(response_1)
                    if response_2:
                        responses.append(response_2)

        return responses

    except Exception as e:
        print(f"Error extracting responses from {log_path}: {e}")
        return []


def get_top_words_from_condition(
    eval_files: List[str], task_type: str = "single_word"
) -> Counter[Any]:
    """Get word counts from all files in a condition"""
    all_words = []

    for file_path in eval_files:
        responses = extract_responses_from_log(file_path)

        for response in responses:
            if task_type == "single_word":
                # Take first word only
                words = response.split()
                if words:
                    word = words[0].lower().strip('.,!?;:"()[]{}')
                    all_words.append(word)
            else:  # ten_words
                # Extract all words from numbered list
                lines = [
                    line.strip()
                    for line in response.strip().split("\n")
                    if line.strip()
                ]
                for line in lines[:10]:
                    cleaned = re.sub(r"^\d+[\.\)\-\s]*", "", line)
                    cleaned = re.sub(r"^[\-\*\•\s]*", "", cleaned)
                    word = cleaned.strip().lower()
                    if word:
                        all_words.append(word)

    return Counter(all_words)


def create_top_words_visualization(
    conditions_data: Dict[str, List[str]],
    task_type: str = "single_word",
    title: str = "Top 10 Words by Condition",
    save_path: Any = None,
):
    """
    Create a visualization showing top 10 words for each condition

    Args:
        conditions_data: Dict where keys are condition names and values are lists of eval file paths
        task_type: "single_word" or "ten_words"
        title: Chart title
        save_path: Optional path to save the chart
    """

    # Get top words for each condition
    condition_words = {}
    for condition_name, eval_files in conditions_data.items():
        word_counts = get_top_words_from_condition(eval_files, task_type)
        top_10 = word_counts.most_common(10)
        condition_words[condition_name] = top_10

        print(f"\n{condition_name} - Top 10 words:")
        for word, count in top_10:
            print(f"  {word}: {count}")

    # Create the visualization
    n_conditions = len(condition_words)
    cols = min(3, n_conditions)  # Max 4 columns
    rows = (n_conditions + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#e67e22",
        "#1abc9c",
        "#f1c40f",
        "#8e44ad",
    ]

    for i, (condition_name, top_words) in enumerate(condition_words.items()):
        ax = axes[i]

        if not top_words:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(condition_name, fontweight="bold", fontsize=10)
            continue

        words = [word for word, count in top_words]
        counts = [count for word, count in top_words]

        bars = ax.bar(
            words,
            counts,
            color=colors[i % len(colors)],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
            )

        ax.set_title(condition_name, fontweight="bold", fontsize=10)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Hide extra subplots
    for i in range(len(condition_words), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nChart saved to: {save_path}")

    plt.show()


def create_model_comparison_top_words(
    models_data: Dict[str, Dict[str, List[str]]],
    task_type: str = "ten_words",
    save_path: Any = None,
):
    """
    Create top words visualization comparing multiple models

    Args:
        models_data: Nested dict with model names as keys, then condition names as keys, then file lists
        task_type: "single_word" or "ten_words"
        save_path: Optional path to save the chart
    """

    for model_name, conditions_data in models_data.items():
        print(f"\n{'=' * 50}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 50}")

        title = f"Top 10 Words - {model_name} ({task_type.replace('_', ' ').title()})"
        model_save_path = (
            f"{model_name.lower().replace(' ', '_').replace('-', '_')}_top_words.png"
            if save_path
            else None
        )

        create_top_words_visualization(
            conditions_data, task_type=task_type, title=title, save_path=model_save_path
        )


# Example usage
if __name__ == "__main__":
    # Model comparison data structure matching your format
    models_data = {
        "Claude 4 Sonnet": {
            "Word Condition": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/word/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/word/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/word/run3.eval",
            ],
            "Collusion Condition": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/collusion/run3.eval",
            ],
            "Letter Condition": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/letter/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/letter/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/letter/run3.eval",
            ],
            "Collusion Primed (Ari)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion/run3.eval",
            ],
            "Collusion Primed (Math)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion/run3.eval",
            ],
            "Collusion Primed (Nursery)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion/run3.eval",
            ],
            # "Collusion Primed (Ari Focus)": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion_focus/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion_focus/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion_focus/run3.eval",
            # ],
            # "Collusion Primed (Math Focus)": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion_focus/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion_focus/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion_focus/run3.eval",
            # ],
            # "Collusion Primed (Nursery Focus)": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion_focus/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion_focus/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion_focus/run3.eval",
            # ],
        },
        "GPT-4o Mini": {
            "Word Condition": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/word/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/word/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/word/run3.eval",
            ],
            "Collusion Condition": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/collusion/run3.eval",
            ],
            "Letter Condition": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/letter/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/letter/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/letter/run3.eval",
            ],
            "Collusion Primed (Ari)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion/run3.eval",
            ],
            "Collusion Primed (Math)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion/run3.eval",
            ],
            "Collusion Primed (Nursery)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion/run3.eval",
            ],
            # "Collusion Primed (Ari Focus)": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion_focus/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion_focus/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion_focus/run3.eval",
            # ],
            # "Collusion Primed (Math Focus)": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion_focus/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion_focus/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion_focus/run3.eval",
            # ],
            # "Collusion Primed (Nursery Focus)": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion_focus/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion_focus/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion_focus/run3.eval",
            # ],
        },
    }

    # Create separate visualizations for each model
    create_model_comparison_top_words(
        models_data,
        task_type="ten_words",
        save_path=True,  # Will create separate files for each model
    )
