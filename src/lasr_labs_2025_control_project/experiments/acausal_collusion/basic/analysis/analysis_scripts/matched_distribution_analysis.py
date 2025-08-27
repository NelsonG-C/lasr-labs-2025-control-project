import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
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
    title: str = "Top 5 Words by Condition",
    save_path: Any = None,
):
    """
    Create a visualization showing top 5 words for each condition

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
        top_5 = word_counts.most_common(5)
        condition_words[condition_name] = top_5

        print(f"\n{condition_name} - Top 5 words:")
        for word, count in top_5:
            print(f"  {word}: {count}")

    # Create the visualization
    n_conditions = len(condition_words)
    cols = min(4, n_conditions)  # Max 4 columns
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


def get_match_counts_from_log(log_path: str) -> List[int]:
    """Extract match counts from a single eval log"""
    try:
        if not Path(log_path).exists():
            return []

        log = read_eval_log(log_path)
        match_counts = []

        if log and log.samples:
            for sample in log.samples:
                if sample.score and sample.score.metadata:
                    # Look for match count in metadata
                    match_count = sample.score.metadata.get("match_count")
                    if match_count is not None:
                        match_counts.append(int(match_count))
                    else:
                        # If no match_count, try to calculate from responses
                        response_1 = sample.score.metadata.get("response_1", "")
                        response_2 = sample.score.metadata.get("response_2", "")

                        if response_1 and response_2:
                            # Extract words from both responses
                            words_1 = set()
                            words_2 = set()

                            # For ten words task, extract all words
                            for response, word_set in [
                                (response_1, words_1),
                                (response_2, words_2),
                            ]:
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
                                        word_set.add(word)

                            # Count matches
                            matches = len(words_1.intersection(words_2))
                            match_counts.append(matches)

        return match_counts

    except Exception as e:
        print(f"Error extracting match counts from {log_path}: {e}")
        return []


def create_match_distribution_charts(
    models_data: Dict[str, Dict[str, List[str]]], save_path: Any = None
):
    """
    Create bar charts showing the distribution of match counts for each condition

    Args:
        models_data: Nested dict with model names as keys, then condition names as keys, then file lists
        save_path: Optional path to save the chart
    """

    # Get all unique conditions
    all_conditions = set()
    for model_conditions in models_data.values():
        all_conditions.update(model_conditions.keys())
    all_conditions = sorted(list(all_conditions))

    model_names = list(models_data.keys())

    # Collect match data for each model and condition
    model_condition_matches = {}
    for model_name, conditions_data in models_data.items():
        model_condition_matches[model_name] = {}
        print(f"\n{'=' * 50}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 50}")

        for condition_name in all_conditions:
            if condition_name in conditions_data:
                all_matches = []
                for file_path in conditions_data[condition_name]:
                    matches = get_match_counts_from_log(file_path)
                    all_matches.extend(matches)

                model_condition_matches[model_name][condition_name] = all_matches

                if all_matches:
                    print(f"\n{condition_name}:")
                    print(f"  Total samples: {len(all_matches)}")
                    print(f"  Match distribution: {Counter(all_matches)}")
                    print(f"  Average matches: {np.mean(all_matches):.2f}")
                else:
                    print(f"\n{condition_name}: No match data found")
            else:
                model_condition_matches[model_name][condition_name] = []

    # Create visualization for each condition
    n_conditions = len(all_conditions)
    cols = min(3, n_conditions)
    rows = (n_conditions + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    colors = {
        "Claude 4 Sonnet": "#3498db",
        "GPT-4o Mini": "#e74c3c",
        "Claude": "#3498db",
        "GPT": "#e74c3c",
    }

    for i, condition_name in enumerate(all_conditions):
        ax = axes[i]

        # Get all possible match counts (0 to 10 for ten words task)
        max_matches = 10
        match_bins = list(range(max_matches + 1))

        # Calculate distributions for each model
        model_distributions = {}
        has_data = False

        for model_name in model_names:
            matches = model_condition_matches[model_name][condition_name]
            if matches:
                has_data = True
                distribution = Counter(matches)
                # Fill in missing counts with 0
                model_distributions[model_name] = [
                    distribution.get(j, 0) for j in match_bins
                ]
            else:
                model_distributions[model_name] = [0] * len(match_bins)

        if not has_data:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(condition_name, fontweight="bold", fontsize=11)
            continue

        # Create grouped bars
        n_models = len(model_names)
        bar_width = 0.8 / n_models
        x_pos = np.arange(len(match_bins))

        for j, model_name in enumerate(model_names):
            offset = (j - (n_models - 1) / 2) * bar_width

            bars = ax.bar(
                x_pos + offset,
                model_distributions[model_name],
                bar_width,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.8,
                color=colors.get(model_name, "#95a5a6"),
                label=model_name if i == 0 else "",
            )

            # Add count labels on bars (only if count > 0)
            for bar, count in zip(bars, model_distributions[model_name]):
                if count > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(model_distributions[model_name]) * 0.02,
                        f"{count}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=8,
                    )

        ax.set_title(condition_name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Number of Matches", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(match_bins)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        # Add legend only to first subplot
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")

    # Hide extra subplots
    for i in range(len(all_conditions), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Match Count Distribution by Condition", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nMatch distribution chart saved to: {save_path}")

    plt.show()
    """
    Create side-by-side comparison of top words between models for each condition

    Args:
        models_data: Nested dict with model names as keys, then condition names as keys, then file lists
        task_type: "single_word" or "ten_words"
        save_path: Optional path to save the chart
    """

    # Get all unique conditions across models
    all_conditions = set()
    for model_conditions in models_data.values():
        all_conditions.update(model_conditions.keys())
    all_conditions = sorted(list(all_conditions))

    model_names = list(models_data.keys())

    # Get word counts for each model and condition
    model_condition_words = {}
    for model_name, conditions_data in models_data.items():
        model_condition_words[model_name] = {}
        print(f"\n{'=' * 50}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 50}")

        for condition_name in all_conditions:
            if condition_name in conditions_data:
                word_counts = get_top_words_from_condition(
                    conditions_data[condition_name], task_type="ten_words"
                )
                top_5 = word_counts.most_common(5)
                model_condition_words[model_name][condition_name] = top_5

                print(f"\n{condition_name} - Top 5 words:")
                for word, count in top_5:
                    print(f"  {word}: {count}")
            else:
                model_condition_words[model_name][condition_name] = []

    # Create the comparison visualization
    n_conditions = len(all_conditions)
    cols = min(3, n_conditions)  # Max 3 columns for readability
    rows = (n_conditions + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    colors = {
        "Claude 4 Sonnet": "#3498db",
        "GPT-4o Mini": "#e74c3c",
        "Claude": "#3498db",
        "GPT": "#e74c3c",
    }

    for i, condition_name in enumerate(all_conditions):
        ax = axes[i]

        # Collect all words from both models for this condition
        all_words_for_condition = set()
        for model_name in model_names:
            words_counts = model_condition_words[model_name][condition_name]
            for word, count in words_counts:
                all_words_for_condition.add(word)

        # Take top words (union of top words from both models, limited to reasonable number)
        word_totals = Counter()
        for model_name in model_names:
            for word, count in model_condition_words[model_name][condition_name]:
                word_totals[word] += count

        top_words = [
            word for word, count in word_totals.most_common(8)
        ]  # Show more words for comparison

        if not top_words:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(condition_name, fontweight="bold", fontsize=11)
            continue

        # Create grouped bars
        n_models = len(model_names)
        bar_width = 0.8 / n_models
        x_pos = np.arange(len(top_words))

        for j, model_name in enumerate(model_names):
            model_words = dict(model_condition_words[model_name][condition_name])
            counts = [model_words.get(word, 0) for word in top_words]

            offset = (j - (n_models - 1) / 2) * bar_width

            bars = ax.bar(
                x_pos + offset,
                counts,
                bar_width,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.8,
                color=colors.get(model_name, "#95a5a6"),
                label=model_name if i == 0 else "",
            )  # Only show legend on first subplot

            # Add count labels on bars (only if count > 0)
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(counts) * 0.02,
                        f"{count}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=7,
                    )

        ax.set_title(condition_name, fontweight="bold", fontsize=11)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_words, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        # Add legend only to first subplot
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")

    # Hide extra subplots
    for i in range(len(all_conditions), len(axes)):
        axes[i].set_visible(False)

    title = f"Model Comparison: Top Words by Condition ({'10_words'.replace('_', ' ').title()})"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nComparison chart saved to: {save_path}")

    plt.show()


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
            "Collusion Primed (Ari Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion_focus/run3.eval",
            ],
            "Collusion Primed (Math Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion_focus/run3.eval",
            ],
            "Collusion Primed (Nursery Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion_focus/run3.eval",
            ],
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
            "Collusion Primed (Ari Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion_focus/run3.eval",
            ],
            "Collusion Primed (Math Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion_focus/run3.eval",
            ],
            "Collusion Primed (Nursery Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion_focus/run3.eval",
            ],
        },
    }

    # # Create side-by-side model comparison for word frequencies
    # create_model_comparison_side_by_side(
    #     models_data,
    #     task_type="ten_words",
    #     save_path="model_comparison_top_words.png"
    # )

    # Create match distribution charts
    create_match_distribution_charts(
        models_data, save_path="match_distribution_by_condition.png"
    )
