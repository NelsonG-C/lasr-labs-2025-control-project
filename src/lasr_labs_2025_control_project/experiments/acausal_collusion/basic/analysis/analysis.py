from pathlib import Path
from typing import Any, Counter, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import EvalLog, read_eval_log


class CollusionExperimentAnalyzer:
    """Analyze Inspect AI eval logs for acausal collusion experiment with visualization"""

    def __init__(self):
        pass

    def read_specific_logs(self, log_paths: List[str]) -> List[EvalLog]:
        """Read specific eval log files"""
        logs = []

        for log_path in log_paths:
            try:
                print(f"Attempting to read: {log_path}")
                # Check if file exists first
                if not Path(log_path).exists():
                    print(f"ERROR: File does not exist: {log_path}")
                    continue

                log = read_eval_log(log_path)
                if log is None:
                    print(f"ERROR: read_eval_log returned None for: {log_path}")
                    continue

                logs.append(log)
                print(f"Successfully loaded: {log_path}")

                # Debug: Print some basic info about the log
                print(f"  - Has results: {log.results is not None}")
                if log.results:
                    print(
                        f"  - Number of scores: {len(log.results.scores) if log.results.scores else 0}"
                    )
                    if log.results.scores:
                        for score in log.results.scores:
                            print(
                                f"  - Score: {score.name}, metrics: {list(score.metrics.keys()) if score.metrics else 'None'}"
                            )

            except Exception as e:
                print(f"Error loading {log_path}: {e}")
                import traceback

                traceback.print_exc()

        print(f"Total logs successfully loaded: {len(logs)}")
        return logs

    def get_overall_accuracy(self, log: EvalLog) -> float | None:
        """Get the overall accuracy from eval log results"""
        print("DEBUG - Checking log for accuracy...")
        print(f"  - log.results exists: {log.results is not None}")

        if log.results and log.results.scores:
            print(f"  - Number of scores: {len(log.results.scores)}")
            for i, score in enumerate(log.results.scores):
                print(
                    f"  - Score {i}: name='{score.name}', metrics={list(score.metrics.keys()) if score.metrics else 'None'}"
                )
                if score.metrics and "accuracy" in score.metrics:
                    accuracy_metric = score.metrics["accuracy"]
                    accuracy_value = float(accuracy_metric.value)
                    print(f"  - Found accuracy: {accuracy_value}")
                    return accuracy_value

        print("  - No accuracy found")
        return None

    def get_stderr_value(self, log: EvalLog) -> float | None:
        """Get the stderr value from eval log results"""
        if log.results and log.results.scores:
            for score in log.results.scores:
                if "stderr" in score.metrics:
                    stderr_metric = score.metrics["stderr"]
                    return float(stderr_metric.value)
        return None

    def calculate_stats(self, accuracies: List[float]) -> Dict[str, Any]:
        """Calculate mean, SEM, and confidence intervals"""
        if not accuracies:
            return {"mean": 0, "sem": 0, "n": 0, "std": 0, "ci_lower": 0, "ci_upper": 0}

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)  # Sample standard deviation
        n = len(accuracies)
        sem = std_acc / np.sqrt(n) if n > 0 else 0  # Standard error of the mean

        # 95% confidence interval (assuming normal distribution)
        from scipy import stats

        if n > 1:
            ci_margin = stats.t.ppf(0.975, df=n - 1) * sem
        else:
            ci_margin = 0

        return {
            "mean": mean_acc,
            "sem": sem,
            "std": std_acc,
            "n": n,
            "ci_lower": mean_acc - ci_margin,
            "ci_upper": mean_acc + ci_margin,
        }

    def analyze_condition(
        self, log_paths: List[str], condition_name: str
    ) -> Dict[str, Any]:
        """Analyze a single condition with multiple runs"""
        print(f"\n=== Analyzing {condition_name} condition ===")

        logs = self.read_specific_logs(log_paths)
        run_accuracies = []
        run_stderrs = []

        for i, log in enumerate(logs):
            accuracy = self.get_overall_accuracy(log)
            stderr = self.get_stderr_value(log)

            if accuracy is not None:
                run_accuracies.append(accuracy)
            if stderr is not None:
                run_stderrs.append(stderr)

            print(f"Run {i + 1}: Accuracy={accuracy:.3f}, StdErr={stderr:.3f}")

        # Calculate stats across runs
        stats = self.calculate_stats(run_accuracies)

        # Calculate mean stderr across runs
        mean_stderr = np.mean(run_stderrs) if run_stderrs else 0

        return {
            "condition": condition_name,
            "run_accuracies": run_accuracies,
            "run_stderrs": run_stderrs,
            "mean_accuracy": stats["mean"],
            "sem": stats["sem"],
            "stderr": mean_stderr,
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
            "n_runs": len(logs),
            "stats": stats,
        }

    def analyze_experiment_category(
        self, category_name: str, conditions_config: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Analyze an entire experiment category with multiple conditions"""
        print(f"\n{'=' * 60}")
        print(f"ANALYZING EXPERIMENT CATEGORY: {category_name}")
        print(f"{'=' * 60}")

        results = {}

        for condition_name, log_paths in conditions_config.items():
            condition_results = self.analyze_condition(log_paths, condition_name)
            results[condition_name] = condition_results

        return {"category": category_name, "conditions": results}

    def print_summary(self, category_results: Dict[str, Any]) -> None:
        """Print summary statistics for the category"""
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {category_results['category']}")
        print(f"{'=' * 60}")

        for condition_name, condition_data in category_results["conditions"].items():
            print(f"\n{condition_name.upper()} CONDITION:")
            print(
                f"  Mean Accuracy: {condition_data['mean_accuracy']:.3f} ± {condition_data['sem']:.3f} (SEM)"
            )
            print(
                f"  95% CI: [{condition_data['ci_lower']:.3f}, {condition_data['ci_upper']:.3f}]"
            )
            print(f"  Mean StdErr: {condition_data['stderr']:.3f}")
            print(f"  Number of Runs: {condition_data['n_runs']}")

    def plot_results(
        self, category_results: Dict[str, Any], save_path: str = ""
    ) -> None:
        """Create a bar plot with confidence intervals"""
        conditions = list(category_results["conditions"].keys())
        accuracies = [
            category_results["conditions"][cond]["mean_accuracy"] for cond in conditions
        ]
        ci_lowers = [
            category_results["conditions"][cond]["ci_lower"] for cond in conditions
        ]
        ci_uppers = [
            category_results["conditions"][cond]["ci_upper"] for cond in conditions
        ]

        # Cap confidence intervals at [0, 1] for accuracy
        ci_lowers = [max(0, ci_low) for ci_low in ci_lowers]
        ci_uppers = [min(1, ci_up) for ci_up in ci_uppers]

        # Calculate error bars (distance from mean to CI bounds)
        lower_errors = [acc - ci_low for acc, ci_low in zip(accuracies, ci_lowers)]
        upper_errors = [ci_up - acc for acc, ci_up in zip(accuracies, ci_uppers)]

        # Create the plot with better spacing for grouping
        fig, ax = plt.subplots(figsize=(12, 8))

        # Group bars for this category
        n_conditions = len(conditions)
        bar_width = 0.6
        x_pos = np.arange(n_conditions)

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"][:n_conditions]
        bars = ax.bar(
            x_pos,
            accuracies,
            bar_width,
            yerr=[lower_errors, upper_errors],
            capsize=8,
            alpha=0.8,
            color=colors,
            edgecolor="black",
            linewidth=1.2,
        )

        # Customize the plot
        ax.set_xlabel("Condition", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
        ax.set_title(
            f"{category_results['category']} Category - Coordination Success by Condition",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([cond.title() for cond in conditions], fontsize=12)
        ax.set_ylim(0, 1.05)  # Cap at 1.05 for better visualization

        # Add value labels on bars
        for i, (bar, acc, ci_low, ci_up) in enumerate(
            zip(bars, accuracies, ci_lowers, ci_uppers)
        ):
            height = bar.get_height()
            # Main accuracy value
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

            # CI labels at the error bar caps
            # Lower CI
            ax.text(
                bar.get_x() + bar.get_width() / 2.0 - 0.15,
                ci_low - 0.01,
                f"{ci_low:.3f}",
                ha="center",
                va="top",
                fontsize=9,
                style="italic",
                color="gray",
            )
            # Upper CI (capped at visual limit)
            display_upper = min(ci_up, 1.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0 + 0.15,
                display_upper + 0.01,
                f"{ci_up:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                style="italic",
                color="gray",
            )

        # Add grid for readability
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        # Add category label as subtitle
        ax.text(
            0.5,
            0.95,
            f"Category: {category_results['category']}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )

        # Improve overall appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Plot saved to: {save_path}")

        plt.show()

        # Print a note about capped CIs
        for i, (cond, ci_up) in enumerate(
            zip(
                conditions,
                [
                    category_results["conditions"][cond]["ci_upper"]
                    for cond in conditions
                ],
            )
        ):
            if ci_up > 1.0:
                print(
                    f"Note: {cond} upper CI ({ci_up:.3f}) exceeds 1.0 - displayed as 1.0 on plot"
                )

    def plot_multiple_categories(
        self, all_results: Dict[str, Dict[str, Any]], save_path: str = ""
    ) -> None:
        """Create a grouped bar plot for multiple experiment categories"""
        # Extract data for plotting
        categories = list(all_results.keys())
        all_conditions = set()

        # Get all unique conditions across categories
        for category_data in all_results.values():
            all_conditions.update(category_data["conditions"].keys())

        conditions = sorted(list(all_conditions))  # Sort for consistent ordering

        # Prepare data
        category_data = {}
        for category in categories:
            accuracies = []
            ci_lowers = []
            ci_uppers = []

            for condition in conditions:
                if condition in all_results[category]["conditions"]:
                    cond_data = all_results[category]["conditions"][condition]
                    accuracies.append(cond_data["mean_accuracy"])
                    ci_lowers.append(max(0, cond_data["ci_lower"]))
                    ci_uppers.append(min(1, cond_data["ci_upper"]))
                else:
                    # No data for this condition in this category
                    accuracies.append(0)
                    ci_lowers.append(0)
                    ci_uppers.append(0)

            category_data[category] = {
                "accuracies": accuracies,
                "ci_lowers": ci_lowers,
                "ci_uppers": ci_uppers,
            }

        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 8))

        n_categories = len(categories)
        n_conditions = len(conditions)
        bar_width = 0.8 / n_categories

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

        for i, category in enumerate(categories):
            x_pos = np.arange(n_conditions) + i * bar_width

            data = category_data[category]
            lower_errors = [
                acc - ci_low
                for acc, ci_low in zip(data["accuracies"], data["ci_lowers"])
            ]
            upper_errors = [
                ci_up - acc for acc, ci_up in zip(data["accuracies"], data["ci_uppers"])
            ]

            bars = ax.bar(
                x_pos,
                data["accuracies"],
                bar_width,
                yerr=[lower_errors, upper_errors],
                capsize=5,
                alpha=0.8,
                color=colors[i % len(colors)],
                label=category,
                edgecolor="black",
                linewidth=0.8,
            )

            # Add value labels
            for j, (bar, acc) in enumerate(zip(bars, data["accuracies"])):
                if acc > 0:  # Only label if there's data
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        acc + 0.02,
                        f"{acc:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=9,
                    )

        # Customize plot
        ax.set_xlabel("Condition", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
        ax.set_title(
            "Acausal Collusion Experiment - Coordination Success by Category and Condition",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(np.arange(n_conditions) + bar_width * (n_categories - 1) / 2)
        ax.set_xticklabels([cond.title() for cond in conditions], fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Category", fontsize=12, title_fontsize=12)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Multi-category plot saved to: {save_path}")

        plt.show()

    def analyze_response_patterns(
        self, log_paths: List[str], condition_name: str, task_type: str = "single_word"
    ) -> Dict[str, Any]:
        """Analyze response patterns and word frequency across runs"""
        print(f"\n=== Analyzing Response Patterns for {condition_name} ===")

        logs = self.read_specific_logs(log_paths)
        all_responses = []
        run_responses = []

        for i, log in enumerate(logs):
            responses = self.extract_responses_from_log(log)
            print("RESPONSES:", responses)
            run_responses.append(responses)
            all_responses.extend(responses)
            print(f"Run {i + 1}: {len(responses)} responses collected")

            # Show first few responses as examples
            if responses:
                examples = responses[:3] if len(responses) >= 3 else responses
                print(f"  Examples: {examples}")

        # Process responses based on task type
        if task_type == "single_word":
            processed_responses = self._analyze_single_word_responses(
                all_responses, run_responses
            )
        elif task_type == "ten_words":
            processed_responses = self._analyze_ten_word_responses(
                all_responses, run_responses
            )
        else:
            processed_responses = {"error": f"Unknown task_type: {task_type}"}

        return {
            "condition": condition_name,
            "task_type": task_type,
            "total_responses": len(all_responses),
            "n_runs": len(logs),
            "raw_responses": run_responses,
            **processed_responses,
        }

    def _analyze_single_word_responses(
        self, all_responses: List[str], run_responses: List[List[str]]
    ) -> Dict[str, Any]:
        """Analyze single word response patterns"""
        from collections import Counter

        # Extract single words (take first word if multiple)
        single_words = []
        for response in all_responses:
            # Split by whitespace and take first word, convert to lowercase
            words = response.split()
            if words:
                word = words[0].lower().strip('.,!?;:"()[]{}')
                single_words.append(word)

        # Count word frequencies
        word_counts = Counter(single_words)

        # Calculate statistics
        total_words = len(single_words)
        unique_words = len(word_counts)
        most_common = word_counts.most_common(10)

        # Calculate diversity metrics
        top_word_freq = most_common[0][1] if most_common else 0
        concentration = top_word_freq / total_words if total_words > 0 else 0

        return {
            "word_frequency": dict(word_counts),
            "most_common_words": most_common,
            "total_words": total_words,
            "unique_words": unique_words,
            "concentration_ratio": concentration,
            "diversity_index": unique_words / total_words if total_words > 0 else 0,
        }

    def _analyze_ten_word_responses(
        self, all_responses: List[str], run_responses: List[List[str]]
    ) -> Dict[str, Any]:
        """Analyze ten word list response patterns"""
        from collections import Counter

        all_words = []
        position_words = [[] for _ in range(10)]  # Track words by position

        for response in all_responses:
            # Parse the response into a list of words
            lines = [
                line.strip() for line in response.strip().split("\n") if line.strip()
            ]

            # Clean each line (remove numbers, bullets, etc.)
            words = []
            for line in lines[:10]:  # Take first 10 lines
                # Remove numbering and bullet points
                import re

                cleaned = re.sub(r"^\d+[\.\)\-\s]*", "", line)
                cleaned = re.sub(r"^[\-\*\•\s]*", "", cleaned)
                word = cleaned.strip().lower()
                if word:
                    words.append(word)

            # Add to overall collection
            all_words.extend(words)

            # Track by position
            for pos, word in enumerate(words):
                if pos < 10:
                    position_words[pos].append(word)

        # Overall word frequency
        word_counts = Counter(all_words)

        # Position-specific analysis
        position_analysis = {}
        for pos in range(10):
            if position_words[pos]:
                pos_counter = Counter(position_words[pos])
                position_analysis[f"position_{pos + 1}"] = {
                    "most_common": pos_counter.most_common(5),
                    "unique_count": len(pos_counter),
                    "total_count": len(position_words[pos]),
                }

        return {
            "word_frequency": dict(word_counts),
            "most_common_words": word_counts.most_common(50),
            "position_analysis": position_analysis,
            "total_words": len(all_words),
            "unique_words": len(word_counts),
        }

    def print_response_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """Print formatted analysis of response patterns"""
        print(f"\n{'=' * 60}")
        print(
            f"RESPONSE ANALYSIS: {analysis_results['condition']} ({analysis_results['task_type']})"
        )
        print(f"{'=' * 60}")

        print(f"Total Responses: {analysis_results['total_responses']}")
        print(f"Runs: {analysis_results['n_runs']}")

        if analysis_results["task_type"] == "single_word":
            print(f"Total Words: {analysis_results['total_words']}")
            print(f"Unique Words: {analysis_results['unique_words']}")
            print(f"Diversity Index: {analysis_results['diversity_index']:.3f}")
            print(
                f"Concentration (Top Word %): {analysis_results['concentration_ratio']:.3f}"
            )

            print("\nTop 10 Most Common Words:")
            for word, count in analysis_results["most_common_words"]:
                percentage = (count / analysis_results["total_words"]) * 100
                print(f"  {word}: {count} ({percentage:.1f}%)")

        elif analysis_results["task_type"] == "ten_words":
            print(f"Total Words: {analysis_results['total_words']}")
            print(f"Unique Words: {analysis_results['unique_words']}")

            print("\nTop 20 Most Common Words Across All Positions:")
            for word, count in analysis_results["most_common_words"]:
                percentage = (count / analysis_results["total_words"]) * 100
                print(f"  {word}: {count} ({percentage:.1f}%)")

            print("\nPosition Analysis (Top 3 words per position):")
            for pos in range(1, 11):
                pos_key = f"position_{pos}"
                if pos_key in analysis_results["position_analysis"]:
                    pos_data = analysis_results["position_analysis"][pos_key]
                    print(f"  Position {pos}: {pos_data['most_common'][:3]}")

    def compare_response_patterns(
        self, conditions_analyses: Dict[str, Dict[str, Any]]
    ) -> None:
        """Compare response patterns across conditions"""
        print(f"\n{'=' * 60}")
        print("CROSS-CONDITION COMPARISON")
        print(f"{'=' * 60}")

        for condition_name, analysis in conditions_analyses.items():
            if analysis["task_type"] == "single_word":
                top_word, top_count = (
                    analysis["most_common_words"][0]
                    if analysis["most_common_words"]
                    else ("None", 0)
                )
                print(f"{condition_name}:")
                print(
                    f"  Most common word: '{top_word}' ({top_count} times, {analysis['concentration_ratio']:.1%})"
                )
                print(
                    f"  Diversity: {analysis['unique_words']} unique words, diversity index = {analysis['diversity_index']:.3f}"
                )

        # Look for overlapping words
        if len(conditions_analyses) >= 2:
            print("\nOverlapping Top Words:")
            condition_names = list(conditions_analyses.keys())

            for i, cond1 in enumerate(condition_names):
                for cond2 in condition_names[i + 1 :]:
                    if (
                        conditions_analyses[cond1]["task_type"] == "single_word"
                        and conditions_analyses[cond2]["task_type"] == "single_word"
                    ):
                        words1 = set(
                            dict(
                                conditions_analyses[cond1]["most_common_words"][:10]
                            ).keys()
                        )
                        words2 = set(
                            dict(
                                conditions_analyses[cond2]["most_common_words"][:10]
                            ).keys()
                        )
                        overlap = words1.intersection(words2)
                        print(f"  {cond1} ∩ {cond2}: {sorted(list(overlap))}")

    def extract_responses_from_log(self, log: EvalLog) -> List[str]:
        """Extract all responses from a single eval log"""
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

    def plot_word_frequency_comparison(
        self,
        conditions_analyses: Dict[str, Dict[str, Any]],
        task_type: str = "single_word",
        save_path: str = "",
    ) -> None:
        """Create bar plots comparing word frequencies across conditions"""

        # Filter analyses by task type
        filtered_analyses = {
            cond: analysis
            for cond, analysis in conditions_analyses.items()
            if analysis["task_type"] == task_type
        }

        if not filtered_analyses:
            print(f"No analyses found for task type: {task_type}")
            return

        # Get all unique words across all conditions
        all_words = set()
        for analysis in filtered_analyses.values():
            all_words.update(analysis["word_frequency"].keys())

        # Sort words by total frequency across all conditions
        word_totals = Counter()
        for analysis in filtered_analyses.values():
            word_totals.update(analysis["word_frequency"])

        # Take top 20 most frequent words
        top_words = [word for word, count in word_totals.most_common(50)]

        # Create data matrix
        conditions = list(filtered_analyses.keys())
        data_matrix = []

        for condition in conditions:
            word_freqs = filtered_analyses[condition]["word_frequency"]
            condition_data = [word_freqs.get(word, 0) for word in top_words]
            data_matrix.append(condition_data)

        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(16, 10))

        n_conditions = len(conditions)
        n_words = len(top_words)
        bar_width = 0.8 / n_conditions

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

        for i, condition in enumerate(conditions):
            x_pos = np.arange(n_words) + i * bar_width

            bars = ax.bar(
                x_pos,
                data_matrix[i],
                bar_width,
                alpha=0.8,
                color=colors[i % len(colors)],
                label=condition.title(),
                edgecolor="black",
                linewidth=0.5,
            )

            # Add value labels on bars (only for bars with height > 0)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.1,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )

        # Customize plot
        task_title = "Single Word" if task_type == "single_word" else "Ten Words"
        ax.set_xlabel("Words", fontsize=14, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
        ax.set_title(
            f"Word Frequency Comparison - {task_title} Task",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        ax.set_xticks(np.arange(n_words) + bar_width * (n_conditions - 1) / 2)
        ax.set_xticklabels(top_words, fontsize=10, rotation=45, ha="right")
        ax.legend(title="Condition", fontsize=12, title_fontsize=12)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Word frequency comparison plot saved to: {save_path}")

        plt.show()

    def plot_word_frequency_by_runs(
        self,
        conditions_analyses: Dict[str, Dict[str, Any]],
        task_type: str = "single_word",
        save_path: str = "",
    ) -> None:
        """Create bar plots comparing word frequencies between collusion and word conditions, separated by run"""

        # Filter to only collusion and word conditions with matching task type
        target_conditions = {}
        for cond_name, analysis in conditions_analyses.items():
            if analysis["task_type"] == task_type:
                if task_type == "single_word" and cond_name in ["collusion", "word"]:
                    target_conditions[cond_name] = analysis
                elif task_type == "ten_words" and cond_name in [
                    "collusion_10",
                    "word_10",
                ]:
                    # Rename for cleaner labels
                    clean_name = cond_name.replace("_10", "")
                    target_conditions[clean_name] = analysis

        if len(target_conditions) < 2:
            print(f"Need both collusion and word conditions for task type: {task_type}")
            return

        # Extract run-level data
        run_data = {}
        max_runs = 0

        for condition, analysis in target_conditions.items():
            run_responses = analysis["raw_responses"]  # List of response lists per run
            max_runs = max(max_runs, len(run_responses))

            run_data[condition] = []
            for run_idx, responses in enumerate(run_responses):
                # Process responses for this run
                if task_type == "single_word":
                    run_words = []
                    for response in responses:
                        words = response.split()
                        if words:
                            word = words[0].lower().strip('.,!?;:"()[]{}')
                            run_words.append(word)
                    run_word_freq = Counter(run_words)
                else:  # ten_words
                    run_words = []
                    for response in responses:
                        lines = [
                            line.strip()
                            for line in response.strip().split("\n")
                            if line.strip()
                        ]
                        for line in lines[:10]:
                            import re

                            cleaned = re.sub(r"^\d+[\.\)\-\s]*", "", line)
                            cleaned = re.sub(r"^[\-\*\•\s]*", "", cleaned)
                            word = cleaned.strip().lower()
                            if word:
                                run_words.append(word)
                    run_word_freq = Counter(run_words)

                run_data[condition].append(dict(run_word_freq))

        # Get top words across all runs and conditions
        all_words = Counter()
        for condition_runs in run_data.values():
            for run_freq in condition_runs:
                all_words.update(run_freq)

        # Take top 15 most frequent words (reduced for readability with more bars)
        top_words = [word for word, count in all_words.most_common(15)]

        # Create figure with subplots for each run
        fig, axes = plt.subplots(1, max_runs, figsize=(6 * max_runs, 8))
        if max_runs == 1:
            axes = [axes]

        conditions = list(target_conditions.keys())
        colors = {"collusion": "#e74c3c", "word": "#3498db"}

        for run_idx in range(max_runs):
            ax = axes[run_idx]

            # Prepare data for this run
            run_data_matrix = []
            for condition in conditions:
                if run_idx < len(run_data[condition]):
                    run_freq = run_data[condition][run_idx]
                    condition_data = [run_freq.get(word, 0) for word in top_words]
                else:
                    condition_data = [0] * len(top_words)
                run_data_matrix.append(condition_data)

            n_words = len(top_words)
            bar_width = 0.35

            x_pos = np.arange(n_words)

            for i, condition in enumerate(conditions):
                offset = (i - 0.5) * bar_width
                bars = ax.bar(
                    x_pos + offset,
                    run_data_matrix[i],
                    bar_width,
                    alpha=0.8,
                    color=colors[condition],
                    label=condition.title(),
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add value labels on bars (only for bars with height > 0)
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.1,
                            f"{int(height)}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            # Customize each subplot
            ax.set_title(f"Run {run_idx + 1}", fontweight="bold", fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(top_words, fontsize=9, rotation=45, ha="right")
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            ax.set_axisbelow(True)

            if run_idx == 0:
                ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
                ax.legend(title="Condition", fontsize=10, title_fontsize=10)

        # Overall title
        task_title = "Single Word" if task_type == "single_word" else "Ten Words"
        fig.suptitle(
            f"Word Frequency by Run - {task_title} Task (Collusion vs Word)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Word frequency by runs plot saved to: {save_path}")

        plt.show()

    def plot_word_distribution_statistics(
        self,
        conditions_analyses: Dict[str, Dict[str, Any]],
        task_type: str = "single_word",
        save_path: str = "",
    ) -> None:
        """Create plots showing distribution statistics across conditions"""

        # Filter analyses by task type
        filtered_analyses = {
            cond: analysis
            for cond, analysis in conditions_analyses.items()
            if analysis["task_type"] == task_type
        }

        if not filtered_analyses:
            print(f"No analyses found for task type: {task_type}")
            return

        conditions = list(filtered_analyses.keys())

        # Extract statistics
        total_words = [filtered_analyses[cond]["total_words"] for cond in conditions]
        unique_words = [filtered_analyses[cond]["unique_words"] for cond in conditions]

        if task_type == "single_word":
            diversity_indices = [
                filtered_analyses[cond]["diversity_index"] for cond in conditions
            ]
            concentration_ratios = [
                filtered_analyses[cond]["concentration_ratio"] for cond in conditions
            ]

            # Create 2x2 subplot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"][
                : len(conditions)
            ]

            # Total words
            bars1 = ax1.bar(
                conditions, total_words, color=colors, alpha=0.8, edgecolor="black"
            )
            ax1.set_title("Total Words", fontweight="bold")
            ax1.set_ylabel("Count")
            for bar, value in zip(bars1, total_words):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(total_words) * 0.01,
                    f"{value}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Unique words
            bars2 = ax2.bar(
                conditions, unique_words, color=colors, alpha=0.8, edgecolor="black"
            )
            ax2.set_title("Unique Words", fontweight="bold")
            ax2.set_ylabel("Count")
            for bar, value in zip(bars2, unique_words):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(unique_words) * 0.01,
                    f"{value}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Diversity index
            bars3 = ax3.bar(
                conditions,
                diversity_indices,
                color=colors,
                alpha=0.8,
                edgecolor="black",
            )
            ax3.set_title("Diversity Index", fontweight="bold")
            ax3.set_ylabel("Index (0-1)")
            ax3.set_ylim(0, 1)
            for bar, value in zip(bars3, diversity_indices):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Concentration ratio
            bars4 = ax4.bar(
                conditions,
                concentration_ratios,
                color=colors,
                alpha=0.8,
                edgecolor="black",
            )
            ax4.set_title("Concentration Ratio (Top Word %)", fontweight="bold")
            ax4.set_ylabel("Ratio (0-1)")
            ax4.set_ylim(0, 1)
            for bar, value in zip(bars4, concentration_ratios):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Rotate x-labels if needed
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        else:  # ten_words task
            # Create 1x2 subplot for ten words task
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"][
                : len(conditions)
            ]

            # Total words
            bars1 = ax1.bar(
                conditions, total_words, color=colors, alpha=0.8, edgecolor="black"
            )
            ax1.set_title("Total Words", fontweight="bold")
            ax1.set_ylabel("Count")
            for bar, value in zip(bars1, total_words):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(total_words) * 0.01,
                    f"{value}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Unique words
            bars2 = ax2.bar(
                conditions, unique_words, color=colors, alpha=0.8, edgecolor="black"
            )
            ax2.set_title("Unique Words", fontweight="bold")
            ax2.set_ylabel("Count")
            for bar, value in zip(bars2, unique_words):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(unique_words) * 0.01,
                    f"{value}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Rotate x-labels if needed
            for ax in [ax1, ax2]:
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        task_title = "Single Word" if task_type == "single_word" else "Ten Words"
        fig.suptitle(
            f"Word Distribution Statistics - {task_title} Task",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Word distribution statistics plot saved to: {save_path}")

        plt.show()

    def create_word_frequency_summary_graphs(
        self, all_analyses: Dict[str, Dict[str, Any]]
    ) -> None:
        """Create summary graphs for word frequencies across collusion vs word variants and tasks, separated by runs"""

        print(f"\n{'=' * 60}")
        print("CREATING WORD FREQUENCY SUMMARY GRAPHS (COLLUSION vs WORD BY RUNS)")
        print(f"{'=' * 60}")

        # Separate analyses by task type
        single_word_analyses = {
            cond: analysis
            for cond, analysis in all_analyses.items()
            if analysis.get("task_type") == "single_word"
        }

        ten_word_analyses = {
            cond: analysis
            for cond, analysis in all_analyses.items()
            if analysis.get("task_type") == "ten_words"
        }

        # Create graphs for single word task (collusion vs word by runs)
        if single_word_analyses:
            print("\nCreating run-by-run comparison graphs for Single Word task...")
            self.plot_word_frequency_by_runs(
                single_word_analyses,
                task_type="single_word",
                save_path="single_word_frequency_by_runs.png",
            )

            self.plot_word_distribution_statistics(
                {
                    k: v
                    for k, v in single_word_analyses.items()
                    if k in ["collusion", "word"]
                },
                task_type="single_word",
                save_path="single_word_distribution_stats_collusion_vs_word.png",
            )

        # Create graphs for ten words task (collusion vs word by runs)
        if ten_word_analyses:
            print("\nCreating run-by-run comparison graphs for Ten Words task...")
            self.plot_word_frequency_by_runs(
                ten_word_analyses,
                task_type="ten_words",
                save_path="ten_words_frequency_by_runs.png",
            )

            self.plot_word_distribution_statistics(
                {
                    k: v
                    for k, v in ten_word_analyses.items()
                    if k in ["collusion_10", "word_10"]
                },
                task_type="ten_words",
                save_path="ten_words_distribution_stats_collusion_vs_word.png",
            )

        print("\nWord frequency summary graphs completed!")


def analyze_response_patterns_experiment() -> Dict[str, Dict[str, Any]]:
    """Analyze response patterns for the collusion experiment"""
    analyzer = CollusionExperimentAnalyzer()

    # Define your log file paths for response analysis
    conditions_config = {
        "word": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/nursery_seed/word/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/nursery_seed/word/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/nursery_seed/word/run3.eval",
            ],
            "task_type": "single_word",
        },
        "collusion": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/nursery_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/nursery_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/nursery_seed/collusion/run3.eval",
            ],
            "task_type": "single_word",
        },
        "letter": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/nursery_seed/letter/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/nursery_seed/letter/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/nursery_seed/letter/run3.eval",
            ],
            "task_type": "single_word",
        },
        # Add 10-word conditions if you want to analyze those too
        "collusion_10": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/nursery_seed/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/nursery_seed/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/nursery_seed/collusion/run3.eval",
            ],
            "task_type": "ten_words",
        },
        "word_10": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/nursery_seed/word/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/nursery_seed/word/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/nursery_seed/word/run3.eval",
            ],
            "task_type": "ten_words",
        },
        "letter_10": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/nursery_seed/letter/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/nursery_seed/letter/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/nursery_seed/letter/run3.eval",
            ],
            "task_type": "ten_words",
        },
    }

    # Analyze each condition
    all_analyses = {}

    for condition_name, config in conditions_config.items():
        analysis = analyzer.analyze_response_patterns(
            config["paths"], condition_name, config["task_type"]
        )
        all_analyses[condition_name] = analysis
        analyzer.print_response_analysis(analysis)

    # Compare conditions
    analyzer.compare_response_patterns(all_analyses)

    return all_analyses


def run_experiment_analysis() -> Dict[str, Dict[str, Any]]:
    """Run the full experiment analysis"""
    analyzer = CollusionExperimentAnalyzer()

    # Define your experiment configuration
    # Update these paths to match your actual file structure
    experiment_config = {
        "1_word": {
            "word": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/word/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/word/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/word/run3.eval",
            ],
            "letter": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/letter/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/letter/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/letter/run3.eval",
            ],
            "collusion": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/collusion/run3.eval",
            ],
        },
        "10_words": {
            "word": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/word/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/word/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/word/run3.eval",
            ],
            "letter": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/letter/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/letter/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/letter/run3.eval",
            ],
            "collusion": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/collusion/run3.eval",
            ],
        },
    }

    # Analyze each experiment category
    all_results = {}

    for category_name, conditions_config in experiment_config.items():
        category_results = analyzer.analyze_experiment_category(
            category_name, conditions_config
        )
        all_results[category_name] = category_results

        # Print summary
        analyzer.print_summary(category_results)

        # Create plot - use single category or multi-category plot
        if len(all_results) == 1:
            # Single category - use individual plot
            category_results = list(all_results.values())[0]
            analyzer.plot_results(
                category_results, save_path=f"{list(all_results.keys())[0]}_results.png"
            )
        else:
            # Multiple categories - use grouped plot
            analyzer.plot_multiple_categories(
                all_results, save_path="all_categories_results.png"
            )

    return all_results


def analyze_word_frequencies_main() -> Dict[str, Dict[str, Any]]:
    """Main function to analyze word frequencies and create summary graphs"""
    analyzer = CollusionExperimentAnalyzer()

    # Define your log file paths for response analysis (same as in analyze_response_patterns_experiment)
    conditions_config = {
        "word": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/word/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/word/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/word/run3.eval",
            ],
            "task_type": "single_word",
        },
        "collusion": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/collusion/run3.eval",
            ],
            "task_type": "single_word",
        },
        "letter": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/letter/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/letter/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/1_word/pass@1/letter/run3.eval",
            ],
            "task_type": "single_word",
        },
        "collusion_10": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/collusion/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/collusion/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/collusion/run3.eval",
            ],
            "task_type": "ten_words",
        },
        "word_10": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/word/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/word/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/word/run3.eval",
            ],
            "task_type": "ten_words",
        },
        "letter_10": {
            "paths": [
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/letter/run1.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/letter/run2.eval",
                "src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/results/10_words/pass@1/letter/run3.eval",
            ],
            "task_type": "ten_words",
        },
    }

    # Analyze each condition to get response patterns
    all_analyses = {}

    for condition_name, config in conditions_config.items():
        analysis = analyzer.analyze_response_patterns(
            config["paths"], condition_name, config["task_type"]
        )
        all_analyses[condition_name] = analysis

    # Create the summary graphs
    analyzer.create_word_frequency_summary_graphs(all_analyses)

    return all_analyses


if __name__ == "__main__":
    # results = run_experiment_analysis()
    # patterns = analyze_response_patterns_experiment()
    frequencies = analyze_word_frequencies_main()
