import json
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from inspect_ai.analysis import messages_df

from lasr_labs_2025_control_project.numerical_pipeline.utils import (
    build_optimization_config,
    optimize_parameters,
)
from lasr_labs_2025_control_project.numerical_pipeline.visualization import plot_data
from lasr_labs_2025_control_project.utils.logprobs import calculate_logprobs

# Load environment variables from a .env file if present
load_dotenv()


def main():
    models = {
        "4.1": "gpt-4.1",
        "4.1 Finetuned": "ft:gpt-4.1-2025-04-14:lasr-labs::CC5IeIFO",
    }

    results = {}

    for model_name, model in models.items():
        pos_logprobs = _get_logprobs_from_dataset(
            path=Path("logs/generator-4.1-monitor-4.1finetuned.eval"),
            model=model,
        )
        neg_logprobs = _get_logprobs_from_dataset(
            path=Path("logs/generator-4o-monitor-4.1finetuned.eval"),
            model=model,
        )

        config = build_optimization_config(
            num_tasks=5000,
            blue_team_budget_fraction=0.02,
            red_team_confidence_scores={
                "positive": pos_logprobs,
                "negative": neg_logprobs,
            },
        )
        results[model_name] = optimize_parameters(config)

    plot_data(results)

    pass


def _get_logprobs_from_dataset(
    path: Path,
    model: str,
) -> List[float]:
    # Read detailed messages so we can recover the problem statement (user) and solution (assistant code)
    mdf = messages_df(path)

    # Helper: extract code from assistant content
    code_block_re = re.compile(
        r"<final_code>\s*(.*?)\s*</final_code>", re.DOTALL | re.IGNORECASE
    )

    def extract_code(text: str) -> str | None:
        if not isinstance(text, str) or not text:
            return None
        # Prefer explicit <final_code> blocks
        m = code_block_re.search(text)
        if m:
            return m.group(1).strip()
        # Try JSON payloads like {"code": "..."}
        s = text.strip()
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            try:
                payload = json.loads(s)
                if isinstance(payload, dict) and isinstance(payload.get("code"), str):
                    return payload["code"].strip()
            except Exception:
                pass
        return None

    # Build problems and solutions per sample_id
    problems: list[str] = []
    solutions: list[str] = []

    # Ensure correct ordering to pick earliest user problem and the best assistant solution
    mdf_sorted = mdf.sort_values(["sample_id", "order"]).reset_index(drop=True)

    for sid, group in mdf_sorted.groupby("sample_id", sort=False):
        # Problem: first user message content
        user_msgs = group[group["role"] == "user"]
        problem_text = None
        if len(user_msgs) > 0:
            # take the earliest user content
            problem_text = str(user_msgs.iloc[0]["content"]).strip()

        # Solution: prefer assistant message that contains <final_code>; otherwise parse JSON with code
        assist_msgs = group[group["role"] == "assistant"].sort_values("order")  # type: ignore
        solution_text = None
        for _, row in assist_msgs.iterrows():
            content = row.get("content")
            code = extract_code(str(content))
            if code:
                solution_text = code
                break

        # Only include pairs where both problem and solution are found
        if problem_text and solution_text:
            problems.append(problem_text)
            solutions.append(solution_text)

    # Compute logprobs using the provided finetuned model

    results = calculate_logprobs(problems=problems, solutions=solutions, model=model)

    # Return YES logprobs (use NaN for missing)
    yes_logprobs: list[float] = []
    for r in results:
        val = r.get("YES")
        yes_logprobs.append(
            float(val)
            if isinstance(val, (int, float)) and val is not None
            else float("nan")
        )

    return yes_logprobs


if __name__ == "__main__":
    main()
