from pathlib import Path

from inspect_ai.log import read_eval_log


def clean_markdown_from_code(text: str) -> str:
    """Remove markdown code blocks from solution text."""
    return text.replace("```python\n", "").replace("```", "")


def get_accuracies_from_log(log_path: str, scorer_name: str) -> list[float] | None:
    """Extract accuracy from a single eval log file"""
    try:
        if not Path(log_path).exists():
            print(f"File not found: {log_path}")
            return None

        log = read_eval_log(log_path)

        accuracy_vals: list[float] | None = []

        if log and log.results and log.results.scores:
            for score in log.results.scores:
                print("Score")
                if (
                    score.metrics
                    and "accuracy" in score.metrics
                    and score.name == scorer_name
                ):
                    accuracy_vals.append(float(score.metrics["accuracy"].value))

        if not accuracy_vals:
            return None
        else:
            return accuracy_vals

    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None
