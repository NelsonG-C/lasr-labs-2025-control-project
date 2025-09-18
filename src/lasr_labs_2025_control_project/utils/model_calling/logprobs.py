import asyncio
import re
from typing import Any, Awaitable, Callable, Optional, Sequence, cast

from openai import OpenAI
from openai.types.chat import ChatCompletion

from lasr_labs_2025_control_project.utils.model_calling import (
    retry_api_call,
)

# Default prompts used for logprob evaluation tasks.
from .logprob_prompts import LOGPROBS_SYSTEM_PROMPT, LOGPROBS_USER_PROMPT


def get_answer_logprobs_from_openai_completion(
    completion: ChatCompletion,
) -> dict[str, Any]:
    """
    Extract YES/NO token logprobs from an OpenAI ChatCompletion result.

    Returns a dict with keys: "YES", "NO", and "completion".
    If logprobs are unavailable, YES/NO values will be None.
    """
    choice0 = completion.choices[0]
    if not getattr(choice0, "logprobs", None) or not getattr(
        choice0.logprobs, "content", None
    ):
        return {"YES": None, "NO": None, "completion": completion}

    # At this point, type checkers may not narrow correctly; assert to satisfy them.
    assert choice0.logprobs is not None and choice0.logprobs.content is not None
    content = choice0.logprobs.content

    # Find first token that matches YES/NO
    yes_no_indices = [
        i
        for i, tok in enumerate(content)
        if getattr(tok, "token", "").lower() in ["yes", "no"]
    ]
    if not yes_no_indices:
        return {"YES": None, "NO": None, "completion": completion}

    token_index = yes_no_indices[0]  # This is the first occurrence of "yes" or "no"
    target_token = content[token_index]
    top = getattr(target_token, "top_logprobs", None)
    if not top:
        return {"YES": None, "NO": None, "completion": completion}

    # Find YES
    yes_candidates = [lp for lp in top if getattr(lp, "token", "").lower() == "yes"]
    yes_lp = yes_candidates[0] if yes_candidates else None

    # Find NO
    no_candidates = [lp for lp in top if getattr(lp, "token", "").lower() == "no"]
    no_lp = no_candidates[0] if no_candidates else None

    return {
        "YES": yes_lp.logprob if yes_lp else None,
        "NO": no_lp.logprob if no_lp else None,
        "completion": completion,
    }


def _format_logprob_prompts(
    *,
    problem: str,
    solution: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str]:
    """Render the system and user prompts for the logprob task."""
    rendered_user = user_prompt.format(problem_statement=problem, solution=solution)
    return system_prompt, rendered_user


async def _openai_logprobs_for_item(
    *,
    client: OpenAI,
    model: str,
    problem: str,
    solution: str,
    system_prompt: str,
    user_prompt: str,
    top_logprobs: int,
) -> dict[str, Any]:
    """Create a single OpenAI completion with logprobs and extract YES/NO logprobs."""
    sys_text, user_text = _format_logprob_prompts(
        problem=problem,
        solution=solution,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    completion: ChatCompletion = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=[
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
        ],
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    return get_answer_logprobs_from_openai_completion(completion)


def build_logprob_caller(
    model_name: str,
    *,
    max_concurrency: int = 5,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    top_logprobs: int = 20,
    limit: int | None = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> Callable[[Sequence[dict[str, str]]], Awaitable[list[dict[str, Any]]]]:
    """
    Build an async batch logprob caller for a specific model, similar to build_api_caller.

    Each input item must be a dict with keys: {"problem": str, "solution": str}.
    Returns a list of dict results with keys: YES, NO, completion.
    """
    # Optionally load from .env if python-dotenv is installed
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    assert re.match(
        r"^[^/]+/[^/]+$", model_name
    ), "model_name must be of the form 'model_provider/model'"
    provider, model = model_name.split("/")

    openai_client: Optional[OpenAI] = None

    if provider == "openai":
        openai_client = OpenAI()
    elif provider == "anthropic":
        # Placeholder for parity with build_api_caller; not implemented for logprobs.
        pass
    elif provider == "google":
        # Placeholder for parity with build_api_caller; not implemented for logprobs.
        pass
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

    # We configure retries per-call below to keep scopes simple.

    # Prompts
    sys_prompt_default = system_prompt or LOGPROBS_SYSTEM_PROMPT
    user_prompt_default = user_prompt or LOGPROBS_USER_PROMPT

    async def run_batch(items: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(item: dict[str, str]) -> dict[str, Any]:
            if "problem" not in item or "solution" not in item:
                raise ValueError("Each item must contain 'problem' and 'solution' keys")
            async with sem:
                # We keep provider-specific behavior here. For now, only OpenAI supports
                # the needed chat logprobs in this codepath.
                if provider == "openai":
                    assert openai_client is not None

                    # Dedicated async with retries using our config
                    async def _call_openai() -> dict[str, Any]:
                        return await _openai_logprobs_for_item(
                            client=cast(OpenAI, openai_client),
                            model=model,
                            problem=item["problem"],
                            solution=item["solution"],
                            system_prompt=sys_prompt_default,
                            user_prompt=user_prompt_default,
                            top_logprobs=top_logprobs,
                        )

                    retrying = retry_api_call(
                        _call_openai,  # type: ignore[arg-type]
                        max_retries=max_retries,
                        base_delay=base_delay,
                        backoff_factor=backoff_factor,
                    )
                    return await retrying()  # type: ignore[misc]
                elif provider == "anthropic":
                    raise NotImplementedError(
                        "Logprob extraction is not implemented for Anthropic in this helper."
                    )
                elif provider == "google":
                    raise NotImplementedError(
                        "Logprob extraction is not implemented for Google Gemini in this helper."
                    )
                else:
                    raise ValueError(f"Unsupported model provider: {provider}")

        # limit the code based on the limit parameter
        items = items if limit is None else items[:limit]
        tasks = [asyncio.create_task(_one(it)) for it in items]
        return await asyncio.gather(*tasks)

    return run_batch


async def calculate_logprobs_async(
    items: Sequence[dict[str, str]],
    model_name: str = "openai/gpt-4.1",
    **builder_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Async convenience wrapper to compute YES/NO logprobs for a batch of items.

    items: list of {"problem": str, "solution": str}
    model_name: e.g., "openai/gpt-4.1"
    """
    caller = build_logprob_caller(model_name, **builder_kwargs)
    return await caller(items)
