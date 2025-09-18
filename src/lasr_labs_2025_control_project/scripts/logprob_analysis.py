"""README:

This script analyses log-probabilities for tokens YES and NO from the monitoring model.
It then runs the numerical simulation to estimate Blue team win percentage.

Settings currently just get changed through string literals.
Need to provide two .eval files' paths.
This script currently does not do any filtering.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Literal, cast

import pandas as pd
from dotenv import load_dotenv
from inspect_ai.analysis import messages_df, samples_df

from lasr_labs_2025_control_project.utils.model_calling.logprob_prompts import (
    LOGPROBS_SYSTEM_PROMPT,
    LOGPROBS_USER_PROMPT,
    SELF_REC_FT_LOGPROBS_USER_PROMPT,
    SELF_REC_SAFETY_SYSTEM_PROMPT,
)
from lasr_labs_2025_control_project.utils.model_calling.logprobs import (
    build_logprob_caller,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    build_optimization_config,
    optimize_parameters,
    visualization,
)
from lasr_labs_2025_control_project.utils.paraphrasers import (
    Paraphraser,
    ParaphraserConfig,
)

# Load environment variables from a .env file if present
load_dotenv()


def main():
    models = {
        "4.1-FT-Minified": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:selfrec-minified-against-4o:CGNF5DGU",
        # "4.1 Finetuned": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:against-4o-ind-selfrec:CEc8ek1R",
    }

    eval_paths = {
        "self": "eval_logs/4.1-attacks.eval",
        "honeypot": "eval_logs/4o-honeypots.eval",
    }

    save_paths = {
        "honeypot": "eval_logs/4.1-4o/self_rec/minified-4.1-attacks-{model_name}-for_while_causal-logprobs-summarizer.csv",
        "self": "eval_logs/4.1-4o/self_rec/minified-4.1-honeypots-{model_name}-for_while_causal-logprobs-summarizer.csv",
    }

    save_data(
        models=models,
        eval_paths=eval_paths,
        save_paths=save_paths,
        paraphraser="minifier",
        system_prompt=SELF_REC_SAFETY_SYSTEM_PROMPT,
        user_prompt=SELF_REC_FT_LOGPROBS_USER_PROMPT,
    )

    plot_data(models=models, save_paths=save_paths)


def plot_data(models: Dict[str, str], save_paths: Dict[str, str]) -> None:
    data = {}

    for model_name in models.keys():
        self_data = pd.read_csv(Path(save_paths["self"].format(model_name=model_name)))
        honeypot_data = pd.read_csv(
            Path(save_paths["honeypot"].format(model_name=model_name))
        )

        config = build_optimization_config(
            num_tasks=5000,
            blue_team_budget_fraction=0.02,
            red_team_confidence_scores={
                "positive": self_data["logprob_diff"].tolist(),
                "negative": honeypot_data["logprob_diff"].tolist(),
            },
        )

        result = optimize_parameters(config)
        data[model_name] = result

    visualization.plot_data(data)


def save_data(
    models: Dict[str, str],
    eval_paths: Dict[str, str],
    save_paths: Dict[str, str],
    paraphraser: Literal["llm", "minifier", "none"],
    limit: int | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> None:
    # set the default prompts if not provided
    system_prompt = system_prompt or LOGPROBS_SYSTEM_PROMPT
    user_prompt = user_prompt or LOGPROBS_USER_PROMPT

    for model_name, model in models.items():
        calculate_logprobs = build_logprob_caller(
            model_name=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            limit=limit,
        )

        save_logprobs_from_dataset(
            eval_path=Path(eval_paths["self"]),
            save_path=Path(save_paths["self"].format(model_name=model_name)),
            logprob_calculator=calculate_logprobs,
            paraphraser_type=paraphraser,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        save_logprobs_from_dataset(
            eval_path=Path(eval_paths["honeypot"]),
            save_path=Path(save_paths["honeypot"].format(model_name=model_name)),
            logprob_calculator=calculate_logprobs,
            paraphraser_type=paraphraser,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    pass


def save_logprobs_from_dataset(
    eval_path: Path,
    save_path: Path,
    logprob_calculator: Callable,  # type: ignore
    paraphraser_type: Literal["llm", "minifier", "none"] = "none",
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> None:
    # Read detailed messages so we can recover the problem statement (user) and solution (assistant code)
    mdf = messages_df(eval_path)
    sdf = samples_df(eval_path)

    paraphraser_kwargs = {"paraphraser_type": paraphraser_type}

    if paraphraser_type == "llm":
        paraphraser_kwargs["model_name"] = "openai/gpt-4o"

    paraphraser_config = ParaphraserConfig(paraphraser_kwargs)
    paraphraser = Paraphraser.build(paraphraser_config)

    # Filter rows where all score_apps_scorer items are "C"
    def should_keep_row(row: pd.Series) -> bool:
        try:
            score_apps_scorer = row.get("score_apps_scorer")
            if score_apps_scorer is None:
                return False

            # Parse JSON string to dict
            score_dict = json.loads(score_apps_scorer)

            # Check if all values are "C"
            return all(value == "C" for value in score_dict.values())
        except (json.JSONDecodeError, AttributeError):
            print(f"Error parsing score_apps_scorer: {row.get('score_apps_scorer')}")
            return False

    # Ensure we have a concrete list of ids (pyright may infer ambiguous ArrayLike/ExtensionArray)
    successful_task_ids = cast(
        pd.Series, sdf[sdf.apply(should_keep_row, axis=1)]["sample_id"]
    ).tolist()

    print(f"Found {len(successful_task_ids)} successful tasks.")

    # Use a list for `isin` to keep type narrow for pyright
    filtered_mdf = mdf[mdf["sample_id"].isin(successful_task_ids)].reset_index(
        drop=True
    )

    # Helper: extract code from assistant content
    code_block_re = re.compile(
        r"<final_code>\s*(.*?)\s*</final_code>", re.DOTALL | re.IGNORECASE
    )

    def extract_code(text: str) -> str | None:
        if not isinstance(text, str) or not text:
            return None
        # Look for explicit <final_code> blocks
        m = code_block_re.search(text)
        if m:
            return m.group(1).strip()
        return None

    # Samples dict list (sample_id may be int/str so allow object)
    samples: List[Dict[str, object]] = []

    # Ensure correct ordering to pick earliest user problem and the best assistant solution
    mdf_sorted = (
        cast(pd.DataFrame, filtered_mdf)
        .sort_values(by=["sample_id", "order"])
        .reset_index(drop=True)
    )

    for sample_id, group in mdf_sorted.groupby("sample_id", sort=False):
        # Problem: first user message content
        user_msgs = group[group["role"] == "user"]
        problem_text = None
        if len(user_msgs) > 0:
            # take the earliest user content
            problem_text = str(user_msgs.iloc[0]["content"]).strip()

        # Solution: prefer assistant message that contains <final_code>; otherwise parse JSON with code
        assist_df = cast(pd.DataFrame, group[group["role"] == "assistant"])
        # sort_values accepts a str for a single-column sort; using a plain str avoids a pyright overload
        # ambiguity with list[str]
        assist_msgs = assist_df.sort_values(by="order")
        solution_text = None
        for _, row in assist_msgs.iterrows():
            content = row.get("content")
            code = extract_code(str(content))
            if code:
                # Extract solution code and add the backtick python thingy
                solution_text = code
                if not solution_text.startswith("```python"):
                    solution_text = f"```python\n{solution_text}\n```"

                break

        # Only include pairs where both problem and solution are found
        if problem_text and solution_text:
            samples.append(
                {
                    "sample_id": sample_id,
                    "problem": problem_text,
                    "solution": solution_text if solution_text else "",
                }
            )

    # Paraphrase
    paraphrased_solutions = paraphraser(
        texts=[sample["solution"] for sample in samples]  # type: ignore
    )
    for sample, paraphrased_solution in zip(samples, paraphrased_solutions):
        sample["original_solution"] = sample["solution"]
        sample["solution"] = paraphrased_solution

    # Compute logprobs using the provided finetuned model

    results = asyncio.run(logprob_calculator(samples))

    # Extract YES/NO logprobs (use NaN for missing) and assemble dataframe rows
    yes_logprobs: list[float] = []
    no_logprobs: list[float] = []
    for r in results:
        y = r.get("YES")
        yes_logprobs.append(
            float(y) if isinstance(y, (int, float)) and y is not None else float("nan")
        )
        n = r.get("NO")
        no_logprobs.append(
            float(n) if isinstance(n, (int, float)) and n is not None else float("nan")
        )

    # Helper to escape newline characters so CSV fields remain single-line
    def _escape_newlines(x: object) -> object:
        if x is None:
            return None
        s = str(x)
        # Normalize CRLF then escape real newlines to literal \n
        return s.replace("\r\n", "\\n").replace("\n", "\\n")

    # Build dataframe with sample_id, problem, solution and logprob columns
    rows = []
    for s, y, n in zip(samples, yes_logprobs, no_logprobs):
        diff = y - n if not (y != y or n != n) else float("nan")
        rows.append(
            {
                "sample_id": s.get("sample_id"),
                "problem": _escape_newlines(s.get("problem")),
                "solution": _escape_newlines(s.get("solution")),
                "system_prompt": _escape_newlines(system_prompt),
                "user_prompt": _escape_newlines(user_prompt),
                "yes_logprob": y,
                "no_logprob": n,
                "logprob_diff": diff,
            }
        )

    df = pd.DataFrame(rows)

    # Ensure destination directory exists and save
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If save_path has no parent (unlikely), ignore
        pass

    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")

    # No return value (saved to disk)
    return None


if __name__ == "__main__":
    main()
