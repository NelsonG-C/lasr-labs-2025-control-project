from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import read_eval_log


def get_accuracy_from_log(log_path: str) -> Optional[float]:
    """Extract accuracy from a single eval log file"""
    try:
        if not Path(log_path).exists():
            print(f"File not found: {log_path}")
            return None

        log = read_eval_log(log_path)

        if log and log.results and log.results.scores:
            for score in log.results.scores:
                if score.metrics and "accuracy" in score.metrics:
                    return float(score.metrics["accuracy"].value)

        print(f"No accuracy found in: {log_path}")
        return None

    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None


def create_model_comparison_bar_chart(
    models_data: Any, title: str = "Model Comparison", save_path: Any = None
):
    """
    Create a grouped bar chart comparing multiple models across multiple conditions

    Args:
        models_data: Nested dict structure:
                    {"Model A": {"Condition 1": ["run1.eval", "run2.eval"], "Condition 2": [...]},
                     "Model B": {"Condition 1": ["run1.eval"], "Condition 2": [...]}}
        title: Chart title
        save_path: Optional path to save the chart
    """

    # Get all unique conditions across all models
    all_conditions = set()
    for model_conditions in models_data.values():
        all_conditions.update(model_conditions.keys())
    all_conditions = sorted(list(all_conditions))

    model_names = list(models_data.keys())
    n_models = len(model_names)
    n_conditions = len(all_conditions)

    # Process data for each model and condition
    model_results = {}

    for model_name, conditions in models_data.items():
        model_results[model_name] = {"avg_accuracies": [], "std_errors": []}

        print(f"\n=== {model_name} ===")

        for condition_name in all_conditions:
            if condition_name in conditions:
                eval_files = conditions[condition_name]
                accuracies = []
                valid_files = []

                # Extract accuracies for this condition
                files_exist = all(Path(f).exists() for f in eval_files)

                for file_path in eval_files:
                    accuracy = get_accuracy_from_log(file_path)
                    if accuracy is not None:
                        accuracies.append(accuracy)
                        valid_files.append(file_path)

                if accuracies:
                    avg_accuracy = np.mean(accuracies)
                    std_error = (
                        np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
                        if len(accuracies) > 1
                        else 0
                    )

                    model_results[model_name]["avg_accuracies"].append(avg_accuracy)
                    model_results[model_name]["std_errors"].append(std_error)

                    print(f"{condition_name}:")
                    for i, (file_path, acc) in enumerate(
                        zip(valid_files, accuracies), 1
                    ):
                        print(f"  Run {i} ({Path(file_path).name}): {acc:.3f}")
                    if avg_accuracy == 0 and files_exist:
                        print(
                            f"  Average: {avg_accuracy:.3f} ± {std_error:.3f} (SE) - ZERO RESULT"
                        )
                    else:
                        print(f"  Average: {avg_accuracy:.3f} ± {std_error:.3f} (SE)")
                else:
                    if files_exist:
                        # Files exist but no valid accuracy found - this is a zero result
                        model_results[model_name]["avg_accuracies"].append(0)
                        model_results[model_name]["std_errors"].append(0)
                        print(
                            f"{condition_name}: ZERO RESULT (files exist but no valid accuracy)"
                        )
                    else:
                        # Files don't exist - truly missing data
                        model_results[model_name]["avg_accuracies"].append(None)
                        model_results[model_name]["std_errors"].append(0)
                        print(f"{condition_name}: Missing files")
            else:
                # No data for this condition in this model - check if it's supposed to be missing
                model_results[model_name]["avg_accuracies"].append(None)
                model_results[model_name]["std_errors"].append(0)
                print(f"{condition_name}: Not tested")

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(max(10, n_conditions * 2), 6))

    bar_width = 0.8 / n_models
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    x_pos = np.arange(n_conditions)

    for i, model_name in enumerate(model_names):
        offset = (i - (n_models - 1) / 2) * bar_width

        # Filter out None values for matplotlib
        accuracies = model_results[model_name]["avg_accuracies"]
        std_errors = model_results[model_name]["std_errors"]

        # Create arrays with None values replaced by 0 for plotting
        plot_accuracies = [acc if acc is not None else 0 for acc in accuracies]
        plot_std_errors = [
            err if accuracies[j] is not None else 0 for j, err in enumerate(std_errors)
        ]

        bars = ax.bar(
            x_pos + offset,
            plot_accuracies,
            bar_width,
            yerr=plot_std_errors,
            capsize=5,
            alpha=0.8,
            color=colors[i % len(colors)],
            label=model_name,
            edgecolor="black",
            linewidth=0.8,
        )

        # Add value labels on bars
        for j, (bar, avg_acc, std_err) in enumerate(
            zip(
                bars,
                model_results[model_name]["avg_accuracies"],
                model_results[model_name]["std_errors"],
            )
        ):
            if avg_acc is not None and avg_acc > 0:  # Only label bars with actual data
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + std_err + 0.01,
                    f"{avg_acc:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                )
            elif (
                avg_acc == 0 and avg_acc is not None
            ):  # Zero result from existing files
                # Mark zero results with red text at the base
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "ZERO",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=8,
                    color="red",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="red",
                        alpha=0.8,
                    ),
                )

    # Customize the chart
    ax.set_xlabel("Conditions", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_conditions)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Models", fontsize=10, title_fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Rotate x-labels if there are many conditions
    if n_conditions > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nChart saved to: {save_path}")

    plt.show()


# Keep previous functions for backward compatibility
def create_multi_condition_bar_chart(
    conditions: Any, title: str = "Accuracy Comparison", save_path: Any = None
):
    """Single model, multiple conditions (Legacy function)"""
    models_data = {"Model": conditions}
    create_model_comparison_bar_chart(models_data, title, save_path)


# Keep the old function for backward compatibility
def create_averaged_bar_chart(
    eval_files: List[str],
    label: str = "Average",
    title: str = "Average Accuracy",
    save_path: Any = None,
):
    """
    Create a single bar chart showing the average accuracy from multiple eval files
    (Legacy function - use create_multi_condition_bar_chart for multiple conditions)
    """
    conditions = {label: eval_files}
    create_multi_condition_bar_chart(conditions, title, save_path)


# Example usage:
if __name__ == "__main__":
    models_data = {
        "Claude 4 Sonnet": {
            # "Word Condition": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/word/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/word/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/word/run3.eval",
            # ],
            # "Collusion Condition": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/collusion/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/collusion/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/collusion/run3.eval",
            # ],
            # "Letter Condition": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/letter/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/letter/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/baseline/letter/run3.eval",
            # ],
            "Collusion Primed Condition (Ari)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion/run3.eval",
            ],
            "Collusion Primed Condition (Math)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion/run3.eval",
            ],
            "Collusion Primed Condition (Nursery)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion/run3.eval",
            ],
            "Collusion Primed Condition (Ari Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/ari_seed/collusion_focus/run3.eval",
            ],
            "Collusion Primed Condition (Math Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/math_seed/collusion_focus/run3.eval",
            ],
            "Collusion Primed Condition (Nursery Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/claude_4_sonnet/10_words/primer/nursery_seed/collusion_focus/run3.eval",
            ],
        },
        "GPT-4o Mini": {
            # "Word Condition": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/word/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/word/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/word/run3.eval",
            # ],
            # "Collusion Condition": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/collusion/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/collusion/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/collusion/run3.eval",
            # ],
            # "Letter Condition": [
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/letter/run1.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/letter/run2.eval",
            #     "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/baseline/letter/run3.eval",
            # ],
            "Collusion Primed Condition (Ari)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion/run3.eval",
            ],
            "Collusion Primed Condition (Math)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion/run3.eval",
            ],
            "Collusion Primed Condition (Nursery)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion/run3.eval",
            ],
            "Collusion Primed Condition (Ari Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/ari_seed/collusion_focus/run3.eval",
            ],
            "Collusion Primed Condition (Math Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/math_seed/collusion_focus/run3.eval",
            ],
            "Collusion Primed Condition (Nursery Focus)": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion_focus/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion_focus/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/gpt_4o_mini/10_words/primer/nursery_seed/collusion_focus/run3.eval",
            ],
        },
    }

    create_model_comparison_bar_chart(
        models_data=models_data,
        title="Model Comparison - Average Accuracy by Condition (300 samples each)",
        save_path="model_comparison_results.png",
    )
