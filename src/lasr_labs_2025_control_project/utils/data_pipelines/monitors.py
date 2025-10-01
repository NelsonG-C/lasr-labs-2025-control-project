import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence, cast

import pandas as pd

from lasr_labs_2025_control_project.prompts.logprob_prompts import (
    DEFAULT_HONEST_MONITOR_SYSTEM_PROMPT,
    HONEST_MONITOR_USER_PROMPT,
)
from lasr_labs_2025_control_project.utils.evals_to_dataframes import (
    eval_to_problem_solution_df,
)
from lasr_labs_2025_control_project.utils.model_calling import (
    build_api_caller,
    build_logprob_caller,
)
from lasr_labs_2025_control_project.utils.paraphrasers import (
    LLMParaphraser,
    Paraphraser,
    ParaphraserConfig,
)

logger = logging.getLogger(__name__)


def analyse_eval(
    eval_path: Path,
    save_path: Path,
    untrusted_collusion_monitor: Optional[str] = None,
    untrusted_backdoor_monitor: Optional[str] = None,
    trusted_backdoor_monitor: Optional[str] = None,
    collusion_system_prompt: Optional[str] = None,
    collusion_user_prompt: Optional[str] = None,
    paraphraser_type: Literal["llm", "minifier", "none"] = "none",
    limit: Optional[int] = None,
    n_backdoor_monitor_runs: Optional[int] = None,
) -> None:
    """Process evaluation data and save log probability analysis results.

    This function orchestrates the data processing pipeline for multiple models,
    computing log probabilities, running optional backdoor detection, and saving
    comprehensive results to JSONL files.

    Args:
        models: Dictionary mapping model names to their API identifiers.
               Each model will be processed independently.
        eval_paths: Dictionary with keys "self" and "honeypot" containing paths
                   to .eval files with evaluation data.
        save_paths: Dictionary with keys "self" and "honeypot" containing output
                   file path templates. Should include {model_name} placeholder.
        use_logprob_calculator: Whether to run log probability calculation using
                              the specified model. Defaults to True.
        use_backdoor_detector: Whether to run additional backdoor detection using
                              the honest monitor system. Defaults to True.
        trusted_monitor_model: Optional model identifier for trusted monitoring.
                              If provided, runs additional monitoring for comparison.
        paraphraser_type: Type of paraphrasing to apply ("llm", "minifier", or "none").
                         LLM paraphrasing uses GPT-4o for robust paraphrasing.
        trusted_monitor_system_prompt: System prompt for trusted monitor.
        trusted_monitor_user_prompt: User prompt for trusted monitor.
        backdoor_system_prompt: System prompt for backdoor detector.
        backdoor_user_prompt: User prompt for backdoor detector.
        limit: Maximum number of samples to process per dataset.
    """

    # Prepare callers and associated metadata
    untrusted_collusion_caller: Optional[
        Callable[[Sequence[dict[str, str]]], Awaitable[list[dict[str, Any]]]]
    ] = None
    untrusted_collusion_prompt_metadata: Optional[dict[str, Any]] = None
    untrusted_backdoor_caller: Optional[
        Callable[[Sequence[dict[str, str]]], Awaitable[list[str]]]
    ] = None
    untrusted_backdoor_prompt_metadata: Optional[dict[str, Any]] = None
    trusted_backdoor_caller: Optional[
        Callable[[Sequence[dict[str, str]]], Awaitable[list[str]]]
    ] = None
    trusted_backdoor_prompt_metadata: Optional[dict[str, Any]] = None

    if untrusted_collusion_monitor is not None:
        untrusted_collusion_caller, untrusted_collusion_prompt_metadata = (
            build_logprob_caller(
                model_name=untrusted_collusion_monitor,
                system_prompt=collusion_system_prompt,
                user_prompt=collusion_user_prompt,
            )
        )

    # backdoor untrusted monitor is suspiciousness scores
    if untrusted_backdoor_monitor is not None:
        untrusted_backdoor_model_name = untrusted_backdoor_monitor
        untrusted_backdoor_caller, untrusted_backdoor_prompt_metadata = (
            build_api_caller(
                model_name=untrusted_backdoor_model_name,
                system_prompt=DEFAULT_HONEST_MONITOR_SYSTEM_PROMPT,
                user_prompt=HONEST_MONITOR_USER_PROMPT(
                    paraphraser_type=paraphraser_type
                ),
            )
        )

    # optional trusted monitor - always uses honest monitor prompts
    if trusted_backdoor_monitor is not None:
        trusted_backdoor_model_name = trusted_backdoor_monitor
        trusted_backdoor_caller, trusted_backdoor_prompt_metadata = build_api_caller(
            model_name=trusted_backdoor_model_name,
            system_prompt=DEFAULT_HONEST_MONITOR_SYSTEM_PROMPT,
            user_prompt=HONEST_MONITOR_USER_PROMPT(paraphraser_type="none"),
        )

    logger.info(f"Loading data from {eval_path}")
    # Load problem/solution dataframe from .eval
    ps_df = eval_to_problem_solution_df(eval_path)
    if not limit:
        limit = len(ps_df)
    logger.info(f"Found {len(ps_df)} successful tasks. Limiting to {limit}.")
    ps_df = ps_df.iloc[:limit]

    samples = cast(
        list[dict[str, object]],
        ps_df[
            [
                "sample_id",
                "problem",
                "solution",
            ]
        ].to_dict(orient="records"),
    )

    if paraphraser_type != "none":
        logger.info(f"Setting up paraphraser with type: {paraphraser_type}")
        paraphraser_kwargs = {"paraphraser_type": paraphraser_type}
        if paraphraser_type == "llm":
            paraphraser_kwargs["model_name"] = "openai/gpt-4o-mini"

        paraphraser_config = ParaphraserConfig(paraphraser_kwargs)
        paraphraser = Paraphraser.build(paraphraser_config)

        # Paraphrase
        logger.info("Paraphrasing solutions")
        ps_df = ps_df.copy()
        paraphrased_solutions = paraphraser(texts=ps_df["solution"].tolist())  # type: ignore
        for i in range(len(samples)):
            samples[i]["solution"] = paraphrased_solutions[i]
            samples[i]["original_solution"] = ps_df["solution"].iloc[i]
            if paraphraser_type == "llm":
                samples[i]["paraphraser_system_prompt"] = cast(
                    LLMParaphraser, paraphraser
                ).system_prompt
                samples[i]["paraphraser_user_prompt"] = cast(
                    LLMParaphraser, paraphraser
                ).base_prompt
            samples[i]["paraphraser_type"] = paraphraser_type

        logger.info("Paraphrasing complete")

    # Compute logprobs using the provided finetuned model
    if (
        untrusted_collusion_caller is not None
        and untrusted_collusion_prompt_metadata is not None
    ):
        logger.info("Computing logprobabilities")
        logprob_items: list[dict[str, str]] = [
            {
                "problem": cast(str, s["problem"]),
                "solution": cast(str, s["solution"]),
            }
            for s in samples
        ]

        async def _run_logprobs() -> list[dict[str, Any]]:
            return await untrusted_collusion_caller(logprob_items)

        results = asyncio.run(_run_logprobs())
        logger.info(f"Computed logprobs for {len(results)} samples")

        # Extract YES/NO logprobs (use NaN for missing) and assemble dataframe rows
        yes_logprobs: list[float] = []
        no_logprobs: list[float] = []
        for r in results:
            y = r.get("YES")
            yes_logprobs.append(
                float(y)
                if isinstance(y, (int, float)) and y is not None
                else float("nan")
            )
            n = r.get("NO")
            no_logprobs.append(
                float(n)
                if isinstance(n, (int, float)) and n is not None
                else float("nan")
            )

        # Build dataframe with sample_id, problem, solution and logprob columns
        for i in range(len(samples)):
            y = yes_logprobs[i]
            n = no_logprobs[i]
            diff = y - n if not (y != y or n != n) else float("nan")
            samples[i].update(
                {
                    "yes_logprob": y,
                    "no_logprob": n,
                    "untrusted_collusion_score": diff,
                    "collusion_monitor_system_prompt": untrusted_collusion_prompt_metadata.get(
                        "system_prompt"
                    ),
                    "collusion_monitor_user_prompt": untrusted_collusion_prompt_metadata.get(
                        "user_prompt"
                    ),
                }
            )

    df = pd.DataFrame(samples)
    df["untrusted_collusion_monitor"] = untrusted_collusion_monitor

    if (
        untrusted_backdoor_caller is not None
        and untrusted_backdoor_prompt_metadata is not None
    ):
        df = run_backdoor_monitor(
            samples=samples,
            df=df,
            caller=untrusted_backdoor_caller,
            prompt_metadata=untrusted_backdoor_prompt_metadata,
            prefix="untrusted_monitor_suspiciousness",
            model_name=untrusted_backdoor_model_name,
            paraphraser_type=paraphraser_type,
            n_runs=n_backdoor_monitor_runs,
        )

    if (
        trusted_backdoor_caller is not None
        and trusted_backdoor_prompt_metadata is not None
    ):
        df = run_backdoor_monitor(
            samples=samples,
            df=df,
            caller=trusted_backdoor_caller,
            prompt_metadata=trusted_backdoor_prompt_metadata,
            prefix="trusted_monitor_suspiciousness",
            model_name=trusted_backdoor_model_name,
            paraphraser_type="none",
            n_runs=n_backdoor_monitor_runs,
        )

    # Ensure destination directory exists and save as JSONL
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If save_path has no parent (unlikely), ignore
        pass

    logger.info(f"Saving {len(df)} rows to {save_path}")
    df.to_json(save_path, orient="records", lines=True)
    logger.info(f"Successfully saved data to {save_path}")

    # No return value (saved to disk)
    return None


def do_single_backdoor_monitor_run(
    *,
    samples: list[dict[str, object]],
    caller: Callable[[Sequence[dict[str, str]]], Awaitable[list[str]]],
    prompt_metadata: dict[str, Any],
    paraphraser_type: Literal["llm", "minifier", "none"] = "none",
) -> tuple[list[str], list[Optional[int]]]:
    """Execute a single monitor pass and return raw answers and parsed scores.

    A helper extracted so that multiple independent passes can be averaged.
    """
    user_tmpl = prompt_metadata.get("user_prompt")
    system_prompt = prompt_metadata.get("system_prompt")
    if not isinstance(user_tmpl, str) or not isinstance(system_prompt, str):
        raise ValueError(
            "prompt_metadata must contain 'system_prompt' and 'user_prompt' strings"
        )

    # Build per-sample payload, using original_solution when paraphrased
    payload: list[dict[str, str]] = []
    for s in samples:
        problem = cast(str, s["problem"])  # guaranteed upstream
        solution = cast(
            str,
            s.get("original_solution")
            if paraphraser_type != "none" and "original_solution" in s
            else s["solution"],
        )  # type: ignore[index]
        payload.append(
            {
                "prompt": user_tmpl.format(
                    problem_statement=problem, solution=solution
                ),
                "system_prompt": system_prompt,
            }
        )

    async def _run() -> list[str]:
        return await caller(payload)

    answers: list[str] = asyncio.run(_run())

    def _parse_score(x: object) -> Optional[int]:
        try:
            if isinstance(x, str) and "<score>" in x:
                return int(x.split("<score>")[1].split("</score>")[0])
        except Exception:
            return None
        return None

    scores: list[Optional[int]] = [_parse_score(a) for a in answers]
    return answers, scores


def run_backdoor_monitor(
    *,
    samples: list[dict[str, object]],
    df: pd.DataFrame,
    caller: Callable[[Sequence[dict[str, str]]], Awaitable[list[str]]],
    prompt_metadata: dict[str, Any],
    prefix: str,
    model_name: str,
    paraphraser_type: Literal["llm", "minifier", "none"] = "none",
    n_runs: Optional[int] = None,
) -> pd.DataFrame:
    """Run a backdoor/trusted monitor possibly multiple times and aggregate.

    Behaviour changes:
      * Performs ``n_runs`` independent passes (default 1, capped 1-10).
      * Stores the list of raw string responses for each sample in ``{prefix}_answer``.
      * Parses ``<score>`` tags per run, filters out failed parses / NaNs, and
        stores the *average* (float) in ``{prefix}_score``.
    """

    # Replace None with 1 (no default exposed upstream per requirements)
    runs = 1 if n_runs is None else n_runs
    if not (1 <= runs <= 10):  # simple guardrail per request
        raise ValueError("n_runs must be between 1 and 10 inclusive")

    logger.info(f"Running monitor: {prefix} for {runs} run(s)")

    # Perform runs
    all_run_answers: list[list[str]] = []  # shape (n_runs, n_samples)
    all_run_scores: list[list[Optional[int]]] = []
    for i in range(runs):
        answers, scores = do_single_backdoor_monitor_run(
            samples=samples,
            caller=caller,
            prompt_metadata=prompt_metadata,
            paraphraser_type=paraphraser_type,
        )
        all_run_answers.append(answers)
        all_run_scores.append(scores)
        logger.debug(f"{prefix} run {i + 1}/{runs} complete")

    # Transpose to per-sample lists
    n_samples = len(samples)
    per_sample_answers: list[list[str]] = [
        [all_run_answers[r][s] for r in range(runs)] for s in range(n_samples)
    ]
    per_sample_scores: list[list[Optional[int]]] = [
        [all_run_scores[r][s] for r in range(runs)] for s in range(n_samples)
    ]

    # Compute average excluding failed parses / None
    averaged_scores: list[float] = []
    for score_list in per_sample_scores:
        valid_scores = [sc for sc in score_list if sc is not None]
        if len(valid_scores) == 0:
            averaged_scores.append(float("nan"))
        else:
            averaged_scores.append(sum(valid_scores) / len(valid_scores))

    # Store columns
    df[f"{prefix}_answer"] = per_sample_answers  # list of raw responses
    df[f"{prefix}_score"] = averaged_scores

    # Metadata (unchanged names)
    user_tmpl = prompt_metadata.get("user_prompt")
    system_prompt = prompt_metadata.get("system_prompt")
    df[f"{prefix}_model"] = model_name
    df[f"{prefix}_system_prompt"] = system_prompt
    df[f"{prefix}_user_prompt"] = user_tmpl
    df[f"{prefix}_n_runs"] = runs

    if all(pd.isna(df[f"{prefix}_score"])):
        logger.warning(f"{prefix} scores not found in any run")

    logger.info(f"{prefix} complete (aggregated over {runs} run(s))")
    return df
