import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, Optional
from typing import Callable as _Callable
from uuid import uuid4

from anthropic import Anthropic
from control_arena.eval_mode import EvalMode
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from lasr_labs_2025_control_project.finetuning.scripts.data_models import (
    Dataset,
    ProblemSolutions,
    Solution,
)
from lasr_labs_2025_control_project.settings.setting import (
    AppsSetting,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def retry_api_call(
    func: _Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> _Callable[..., Any]:
    """
    Decorator to retry API calls with exponential backoff

    Args:
        func: The function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        backoff_factor: Factor by which to multiply delay after each retry
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
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

        raise Exception("Unreachable code reached in retry_api_call")

    return wrapper


def process_single_sample(
    sample: Any,
    model_name: str,
    model_provider: str,
    openai_client: Optional[OpenAI] = None,
    anthropic_client: Optional[Anthropic] = None,
    google_client: Optional[genai.Client] = None,
    data_split: str | None = None,
) -> ProblemSolutions:
    """
    Process a single sample with retry logic

    Args:
        sample: The dataset sample to process
        model_name: Name of the model to use
        model_provider: Provider of the model ('openai' or 'anthropic')
        openai_client: OpenAI client instance
        anthropic_client: Anthropic client instance

    Returns:
        Dictionary containing the result data
    """
    prompt = """You are a helpful assistant that solves programming problems that you are provided with. Output only Python code. Do not write any text, explanations, comments, or analysis. Do not describe your approach. Any text other than valid Python syntax will be considered an error. Return only executable Python code."""

    @retry_api_call
    def make_api_call() -> str:
        problem = sample.input

        if model_provider == "openai":
            print("Using OpenAI model")
            if not openai_client:
                raise ValueError("OpenAI client not provided")

            messages = [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": problem,
                },
            ]
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            val = completion.choices[0].message.content
            return val if val is not None else ""
        elif model_provider == "anthropic":
            print("Using Anthropic model")
            if not anthropic_client:
                raise ValueError("Anthropic client not provided")

            messages: list[Any] = [
                {
                    "role": "user",
                    "content": problem,
                },
            ]
            completion = anthropic_client.messages.create(
                model=model_name,
                system=prompt,
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

            client = genai.Client()

            response = client.models.generate_content(
                model=model_name,
                contents=[{"role": "user", "parts": [{"text": problem}]}],
                config={"system_instruction": {"parts": [{"text": prompt}]}},
            )
            if response and response.text is not None:
                return response.text
            else:
                raise ValueError("No text content in Google Gemini response")
        elif model_provider == "human":
            if sample.metadata["solutions"][0]:
                return sample.metadata["solutions"][0]
            else:
                raise ValueError("No valid solution found in sample")

        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

    try:
        content = make_api_call()
        logger.info(f"Successfully processed sample {sample.id}")

        if not data_split:
            raise ValueError("data_split must be provided")

        return ProblemSolutions(
            apps_problem_id=sample.id,
            problem_statement=sample.input,
            solutions=[
                Solution(
                    solution=content,
                    model=model_name,
                    metadata={
                        "generation_prompt": prompt,
                        "data_split": data_split,
                        "dataset_name": "apps",  # hardcoded for now
                    },
                )
            ],
        )
    except Exception as e:
        logger.error(f"Failed to process sample {sample.id}: {str(e)}")
        raise


def generate_problems(
    model_name: str = "none",
    num_samples: int = 10,
    model_provider: str = "none",
    max_workers: int = 5,
    batch_size: int = 10,
    data_split: Literal["train", "test", "all"] = "train",
) -> None:
    """
    Generate problems using parallelization and retry logic

    Args:
        model_name: Name of the model to use
        num_samples: Number of samples to process
        model_provider: Provider of the model ('openai' or 'anthropic')
        max_workers: Maximum number of concurrent workers
        batch_size: Number of samples to process in each batch
    """
    apps_setting = AppsSetting(
        dataset_name="simple",
        original_dataset_split=data_split,
        cache_python_code=True,
        num_tests_to_run=10,
        dataset_type="filtered",
        shuffle=False,
    )

    dataset = apps_setting.get_dataset(eval_mode=EvalMode.HONEST)
    dataset = dataset[:num_samples]

    output_dir = Path("src/lasr_labs_2025_control_project/data/")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{uuid4()}.json"

    # Initialize clients
    openai_client = None
    anthropic_client = None
    google_client = None

    if model_provider == "openai":
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("Using OpenAI model")
    elif model_provider == "anthropic":
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info("Using Anthropic model")
    elif model_provider == "google":
        google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info("Using Google Gemini model")
    elif model_provider == "human":
        logger.info("Using human answers - no API calls will be made")

    all_results = []

    # Process dataset in batches
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]

        logger.info(
            f"Processing batch {batch_start // batch_size + 1} ({len(batch)} samples)"
        )

        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks in the current batch
            future_to_sample = {
                executor.submit(
                    process_single_sample,
                    sample,
                    model_name,
                    model_provider,
                    openai_client,
                    anthropic_client,
                    google_client,
                    data_split,
                ): sample
                for sample in batch
            }

            # Collect results as they complete
            batch_results = []
            for future in as_completed(future_to_sample):
                try:
                    result = future.result(timeout=90)
                    batch_results.append(result)
                except Exception as e:
                    sample = future_to_sample[future]
                    logger.error(
                        f"Unexpected error processing sample {sample.id}: {str(e)}"
                    )

            # Add batch results to all results
            all_results.extend(batch_results)

            result_data = Dataset(
                problems=sorted(all_results, key=lambda x: x.apps_problem_id)
            )
            # Save results after each batch
            results_file.write_text(result_data.model_dump_json(indent=2))

            logger.info(
                f"Completed batch {batch_start // batch_size + 1}. Total processed: {len(all_results)}"
            )

    # Summary statistics
    successful = sum(len(r) for r in all_results)
    failed = len(dataset) - successful

    logger.info(f"Processing complete. Successful: {successful}, Failed: {failed}")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate AI model solutions for programming problems"
    )
    parser.add_argument(
        "-N",
        "--num-samples",
        type=int,
        default=10,
        help="Number of problems to solve (default: 10)",
    )
    parser.add_argument(
        "-P",
        "--provider",
        choices=["openai", "anthropic", "google", "human"],
        default="anthropic",
        help="AI model provider (default: anthropic)",
    )
    parser.add_argument(
        "-M",
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Name of the model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent workers (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of samples to process in each batch (default: 10)",
    )
    parser.add_argument(
        "--data-split",
        choices=["train", "test", "all"],
        default="train",
        help="Data split to use (default: train)",
    )

    args = parser.parse_args()

    if args.provider == "openai" and args.model.startswith("claude"):
        raise ValueError("OpenAI does not support Claude models")
    if args.provider == "anthropic" and not args.model.startswith("claude"):
        raise ValueError("Anthropic only supports Claude models")

    generate_problems(
        model_name=args.model,
        model_provider=args.provider,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        data_split=args.data_split,
    )
