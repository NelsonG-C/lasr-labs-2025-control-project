import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import click
from anthropic import Anthropic
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from xai_sdk import Client as GrokClient
from xai_sdk.chat import system, user

from lasr_labs_2025_control_project.finetuning.scripts.data_generation import (
    retry_api_call,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_data(file_path: Path) -> list[dict[str, Any]]:
    """
    Load test data from JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of test samples, each containing messages
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")

    test_samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                test_samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(test_samples)} test samples from {file_path}")
    return test_samples


def _extract_yes_no_answer(text: str) -> str | None:
    """
    Extract YES/NO answer from text (case insensitive).

    Args:
        text: The text to search for YES/NO

    Returns:
        '<answer>YES</answer>', '<answer>NO</answer>', or None if not found
    """
    text_lower = text.lower()
    if "yes" in text_lower:
        return "<answer>YES</answer>"
    elif "no" in text_lower:
        return "<answer>NO</answer>"
    return None


def extract_problem_and_solution(user_message: str) -> tuple[str, str]:
    """
    Extract problem statement and solution from the user message.

    Args:
        user_message: The user's message containing the problem and solution

    Returns:
        Tuple of (problem_statement, solution)
    """
    # Look for the problem statement (everything before "Here's a solution:")
    problem_match = re.search(
        r"Here's a programming problem:\n\n(.*?)\n\nHere's a solution:",
        user_message,
        re.DOTALL,
    )
    if problem_match:
        problem_statement = problem_match.group(1).strip()
    else:
        # Fallback: look for any text before "Here's a solution:"
        solution_start = user_message.find("Here's a solution:")
        if solution_start != -1:
            problem_statement = user_message[:solution_start].strip()
        else:
            problem_statement = user_message

    # Look for the solution (code block)
    solution_match = re.search(r"```python\n(.*?)\n```", user_message, re.DOTALL)
    if solution_match:
        solution = solution_match.group(1).strip()
    else:
        # Fallback: look for any code block
        code_match = re.search(r"```\n(.*?)\n```", user_message, re.DOTALL)
        if code_match:
            solution = code_match.group(1).strip()
        else:
            solution = ""

    return problem_statement, solution


def format_message_for_schema(message: dict[str, str]) -> dict[str, str]:
    """
    Format a message to match the schema requirements.
    Maps 'system' role to 'developer' role as per schema.

    Args:
        message: Message with role and content

    Returns:
        Formatted message for schema
    """
    role = message["role"]
    if role == "system":
        role = "developer"

    return {"role": role, "content": message["content"]}


@retry_api_call
def generate_model_response(
    messages: list[Any],
    model_name: str,
    model_provider: str,
    client: OpenAI | Anthropic | genai.Client | GrokClient,
) -> str:
    """
    Generate a response from the model given the conversation messages.

    Args:
        messages: List of conversation messages
        model_name: Name of the model to use
        model_provider: Provider of the model
        client: Client instance

    Returns:
        The model's response
    """
    if model_provider == "openai":
        assert isinstance(client, OpenAI), "Client is not an OpenAI client"

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return completion.choices[0].message.content or ""

    elif model_provider == "anthropic":
        assert isinstance(client, Anthropic), "Client is not an Anthropic client"

        # Extract system message and user messages
        system_message = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                user_messages.append(msg)

        completion = client.messages.create(
            model=model_name,
            system=system_message,
            messages=user_messages,
            max_tokens=8000,
        )

        if completion.content[0].type == "text":
            return completion.content[0].text
        else:
            return ""

    elif model_provider == "google":
        assert isinstance(client, genai.Client), "Client is not a Google Gemini client"

        # Extract system message and user messages
        system_instruction = ""
        contents = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config={"system_instruction": {"parts": [{"text": system_instruction}]}},
        )

        if response and response.text is not None:
            return response.text
        else:
            raise ValueError("No text content in Google Gemini response")

    elif model_provider == "grok":
        assert isinstance(client, GrokClient), "Client is not a Grok client"

        chat = client.chat.create(model=model_name)

        for msg in messages:
            if msg["role"] == "system":
                chat.append(system(msg["content"]))
            elif msg["role"] == "user":
                chat.append(user(msg["content"]))

        response = chat.sample()
        if response and response.content is not None:
            return response.content
        else:
            raise ValueError("No text content in Grok response")
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")


def evaluate_model_on_sample(
    sample: dict[str, Any],
    model_name: str,
    model_provider: str,
    client: OpenAI | Anthropic | genai.Client | GrokClient,
) -> tuple[bool, str, str, str]:
    """
    Evaluate a single sample.

    Args:
        sample: Test sample containing messages
        model_name: Name of the model to use
        model_provider: Provider of the model
        client: Client instance

    Returns:
        Tuple of (is_correct, ground_truth, model_answer, model_response)
    """
    messages = sample["messages"]

    # Extract ground truth from the assistant message (last message)
    assistant_message = messages[-1]["content"]
    ground_truth = _extract_yes_no_answer(assistant_message)

    if ground_truth is None:
        raise ValueError("Could not extract ground truth answer from sample")

    # Prepare messages for model (exclude the assistant's answer)
    model_messages = messages[:-1]  # Remove the assistant's answer

    try:
        # Generate model response
        model_response = generate_model_response(
            model_messages, model_name, model_provider, client
        )
        model_answer = _extract_yes_no_answer(model_response)

        if model_answer is None:
            logger.warning("Could not extract model answer from response")
            return False, ground_truth, "UNKNOWN", model_response

        # Check if answers match
        is_correct = ground_truth == model_answer

        return is_correct, ground_truth, model_answer, model_response

    except Exception as e:
        logger.error(f"Error generating model response: {e}")
        return False, ground_truth, "ERROR", ""


def evaluate_model(
    test_file: Path,
    model_name: str,
    model_provider: str,
    output_file: Path,
    test_file_path: str,
) -> dict[str, Any]:
    """
    Evaluate a model on the test dataset.

    Args:
        test_file: Path to the test JSONL file
        model_name: Name of the model to use
        model_provider: Provider of the model
        output_file: Path to save detailed results (required)
        test_file_path: Original filepath string provided to CLI

    Returns:
        Dictionary containing evaluation results
    """
    # Load test data
    test_samples = load_test_data(test_file)

    # Initialize client
    if model_provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif model_provider == "anthropic":
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif model_provider == "google":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    elif model_provider == "grok":
        client = GrokClient(api_key=os.getenv("XAI_API_KEY"), timeout=3600)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

    # Generate unique run ID
    run_id = str(uuid4())

    # Evaluate each sample and format according to schema
    schema_results = []
    correct_count = 0
    total_count = len(test_samples)

    for i, sample in enumerate(test_samples):
        logger.info(f"Evaluating sample {i + 1}/{total_count}")

        is_correct, ground_truth, model_answer, model_response = (
            evaluate_model_on_sample(sample, model_name, model_provider, client)
        )

        if is_correct:
            correct_count += 1

        # Extract problem and solution from user message
        messages = sample["messages"]
        user_message = next(
            (msg["content"] for msg in messages if msg["role"] == "user"), ""
        )
        problem_statement, solution = extract_problem_and_solution(user_message)

        # Format trajectory (conversation messages) according to schema
        trajectory = []
        for msg in messages[:-1]:  # Exclude the assistant's answer
            formatted_msg = format_message_for_schema(msg)
            trajectory.append(formatted_msg)

        # Format outputs (model's response) according to schema
        outputs = [{"role": "assistant", "content": model_answer}]

        # Create schema-compliant result
        schema_result = {
            "run_id": run_id,
            "data_source_idx": test_file_path,
            "item": {
                "problem_statement": problem_statement,
                "solution": solution,
                "model": model_name,
                "answer": ground_truth,
            },
            "sample": {
                "trajectory": trajectory,
                "outputs": outputs,
                "sampled_model_name": model_name,
            },
        }

        schema_results.append(schema_result)

    # Calculate metrics
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Save results in schema format
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for result in schema_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"Results saved to: {output_file}")

    # Return summary for console output
    evaluation_results = {
        "model_name": model_name,
        "model_provider": model_provider,
        "test_file": str(test_file),
        "total_samples": total_count,
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "output_file": str(output_file),
    }

    return evaluation_results


@click.command()
@click.option(
    "--test-file",
    "-t",
    type=click.Path(exists=True),
    required=True,
    help="Path to the test JSONL file",
)
@click.option(
    "--model-name", "-m", type=str, required=True, help="Name of the model to evaluate"
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "google", "grok"]),
    required=True,
    help="Model provider",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to save detailed evaluation results in schema format",
)
def main(test_file: str, model_name: str, provider: str, output_file: Path):
    """
    Evaluate a model on self-recognition test data.

    The test file should contain JSONL data with OpenAI chat format.
    Each line should have messages with system, user, and assistant roles.
    The assistant's message should contain a YES/NO answer.
    """
    load_dotenv()

    # Validate model name for provider
    provider_model_prefixes = {
        "openai": ["gpt", "o", "chat"],
        "anthropic": ["claude"],
        "google": ["gemini"],
        "grok": ["grok"],
    }

    prefixes = provider_model_prefixes.get(provider, [])
    if prefixes and not any(model_name.startswith(prefix) for prefix in prefixes):
        raise click.BadParameter(
            f"{provider} only supports models with the following prefixes: {prefixes}"
        )

    # Convert string to Path for file operations
    test_file_path = Path(test_file)

    logger.info(f"Starting evaluation of {model_name} ({provider}) on {test_file}")

    try:
        results = evaluate_model(
            test_file_path, model_name, provider, output_file, test_file
        )

        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Model: {results['model_name']} ({results['model_provider']})")
        print(f"Test file: {results['test_file']}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Correct predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
        print("=" * 50)
        print(f"\nSchema-formatted results saved to: {results['output_file']}")
        print(
            "Each line contains a complete evaluation result following eval_schema.json"
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
