import asyncio
import logging
import re
import time
from typing import Any, Awaitable, Callable, Optional, Sequence, TypeVar, cast

from anthropic import Anthropic
from google import genai
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry_api_call(
    func: F, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0
) -> F:
    """
    Decorator to retry API calls with exponential backoff.

    Works for both sync and async functions. For async coroutines, delays use
    asyncio.sleep; for sync functions, time.sleep is used.

    Args:
        func: The function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        backoff_factor: Factor by which to multiply delay after each retry
    """

    if asyncio.iscoroutinefunction(func):

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001 - surface original error after retries
                    if attempt == max_retries:
                        logger.error(
                            f"API call failed after {max_retries + 1} attempts: {str(e)}"
                        )
                        raise Exception(
                            f"Failed after {max_retries + 1} attempts: {str(e)}"
                        )

                    delay = base_delay * (backoff_factor**attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)

            raise Exception("Unreachable code reached in retry_api_call (async)")

        return cast(F, async_wrapper)

    # Fallback to sync wrapper
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - surface original error after retries
                if attempt == max_retries:
                    logger.error(
                        f"API call failed after {max_retries + 1} attempts: {str(e)}"
                    )
                    raise Exception(
                        f"Failed after {max_retries + 1} attempts: {str(e)}"
                    )

                delay = base_delay * (backoff_factor**attempt)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)

        raise Exception("Unreachable code reached in retry_api_call (sync)")

    return cast(F, wrapper)


def _make_api_call_impl(
    model_name: str,
    system_prompt: str,
    prompt: str,
    openai_client: Optional[OpenAI] = None,
    anthropic_client: Optional[Anthropic] = None,
    google_client: Optional[genai.Client] = None,
) -> str:
    assert re.match(r"^[^/]+/[^/]+$", model_name), (
        "model_name must be of the form 'model_provider/model'"
    )

    model_provider, model_name = model_name.split("/")

    if model_provider == "openai":
        print("Using OpenAI model")
        if not openai_client:
            raise ValueError("OpenAI client not provided")

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        completion = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return completion.choices[0].message.content  # type: ignore
    elif model_provider == "anthropic":
        print("Using Anthropic model")
        if not anthropic_client:
            raise ValueError("Anthropic client not provided")

        messages: list[Any] = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        completion = anthropic_client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=messages,
            max_tokens=8000,
        )
        if completion.content[0].type == "text":
            return completion.content[0].text
        else:
            return ""
    elif model_provider == "google":
        if not google_client:
            raise ValueError("Google Gemini client not provided")

        response = google_client.models.generate_content(
            model=model_name,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"system_instruction": {"parts": [{"text": system_prompt}]}},
        )
        if response and response.text is not None:
            return response.text
        else:
            raise ValueError("No text content in Google Gemini response")

    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")


@retry_api_call
def make_api_call(
    model_name: str,
    system_prompt: str,
    prompt: str,
    openai_client: Optional[OpenAI] = None,
    anthropic_client: Optional[Anthropic] = None,
    google_client: Optional[genai.Client] = None,
) -> str:
    """Synchronous API call with retry via decorator."""
    return _make_api_call_impl(
        model_name=model_name,
        system_prompt=system_prompt,
        prompt=prompt,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
        google_client=google_client,
    )


async def make_api_call_async(
    model_name: str,
    system_prompt: str,
    prompt: str,
    openai_client: Optional[OpenAI] = None,
    anthropic_client: Optional[Anthropic] = None,
    google_client: Optional[genai.Client] = None,
) -> str:
    """
    Async wrapper for make_api_call using a worker thread, with retry.

    Useful when running many calls concurrently under an asyncio event loop.
    """
    return await asyncio.to_thread(
        _make_api_call_impl,
        model_name,
        system_prompt,
        prompt,
        openai_client,
        anthropic_client,
        google_client,
    )


async def batch_make_api_calls(
    calls: Sequence[dict[str, Any]],
    max_concurrency: int = 5,
) -> list[str]:
    """
    Run multiple make_api_call_async with a concurrency limit.

    Each item in `calls` should be a dict with keys matching make_api_call_async
    parameters: model_name, system_prompt, prompt, and optional clients.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(params: dict[str, Any]) -> str:
        async with semaphore:
            return await make_api_call_async(**params)

    tasks = [asyncio.create_task(_run_one(c)) for c in calls]
    return await asyncio.gather(*tasks)


def build_api_caller(
    model_name: str,
    *,
    max_concurrency: int = 5,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable[[Sequence[dict[str, str]]], Awaitable[list[str]]]:
    """
    Build an async batch API caller for a specific model.

    Loads API keys from environment (using python-dotenv if available),
    constructs the correct client, and returns an async function that accepts a
    list of items with keys: {"system_prompt": str, "prompt": str} and returns
    a list of responses.
    """
    # Optionally load from .env if python-dotenv is installed
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    assert re.match(r"^[^/]+/[^/]+$", model_name), (
        "model_name must be of the form 'model_provider/model'"
    )
    provider, _ = model_name.split("/")

    openai_client: Optional[OpenAI] = None
    anthropic_client: Optional[Anthropic] = None
    google_client: Optional[genai.Client] = None

    if provider == "openai":
        # OpenAI client picks up API key from env by default
        openai_client = OpenAI()
    elif provider == "anthropic":
        anthropic_client = Anthropic()
    elif provider == "google":
        google_client = genai.Client()
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

    # Configure per-caller async retry wrapper
    retrying_async_call = retry_api_call(
        make_api_call_async,
        max_retries=max_retries,
        base_delay=base_delay,
        backoff_factor=backoff_factor,
    )

    async def run_batch(items: Sequence[dict[str, str]]) -> list[str]:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(item: dict[str, str]) -> str:
            if "system_prompt" not in item or "prompt" not in item:
                raise ValueError(
                    "Each item must contain 'system_prompt' and 'prompt' keys"
                )
            async with sem:
                return await retrying_async_call(
                    model_name=model_name,
                    system_prompt=item["system_prompt"],
                    prompt=item["prompt"],
                    openai_client=openai_client,
                    anthropic_client=anthropic_client,
                    google_client=google_client,
                )

        tasks = [asyncio.create_task(_one(it)) for it in items]
        return await asyncio.gather(*tasks)

    return run_batch
