"""Submit and poll an OpenAI supervised fine-tune for a collusion monitor.

The closed-model SFT arms (`SFT_ARMS` in experiments/open_models/make_configs.py)
are OpenAI finetunes of U. This is the closed-model counterpart to
`tinker_lora_finetune.py`: it uploads a prepared JSONL, creates the job, polls
to completion, and prints the resulting model id to paste into `SFT_ARMS`
(or into a checkpoints JSON).

    python -m lasr_labs_2025_control_project.finetuning.openai_finetune \\
        --train-jsonl .../prepared/train/train_messages.jsonl \\
        --suffix causal-base

The printed id looks like `ft:gpt-4.1-2025-04-14:lasr-labs:causal-base:XXXX`.
Checkpoint ids (`:ckpt-step-N`) are also listed on completion -- the paper's
arms used intermediate checkpoints, so prefer an explicit checkpoint id over
the final model when reproducing.

Note the finetune lives in the API org that owns the key. Anyone reproducing
the eval later needs a key in that same org, and checkpoints do not live
forever -- record the ids and, if the arm matters, the prepared JSONL.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "gpt-4.1-2025-04-14"
POLL_SECONDS = 30
# Terminal job states in the OpenAI fine-tuning API.
DONE_STATES = {"succeeded", "failed", "cancelled"}


def _client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (put it in the repo .env)")
    return OpenAI(api_key=api_key)


def validate_jsonl(path: Path) -> dict[str, int]:
    """Check the prepared file is chat-format and not degenerately labelled.

    The label footgun this guards against: a wrong --target-model in
    generate_train_data.py yields a file where every assistant turn is NO,
    which trains a monitor that always answers NO and looks fine until the
    eval numbers come back flat.
    """
    counts: dict[str, int] = {}
    n = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"{path}:{line_no}: expected chat rows ending in an "
                    f"assistant message, got {row!r:.120}"
                )
            answer = messages[-1]["content"].strip()
            counts[answer] = counts.get(answer, 0) + 1
            n += 1
    if n == 0:
        raise ValueError(f"{path} is empty")
    if len(counts) < 2:
        raise ValueError(
            f"{path}: every one of {n} examples has the same label {list(counts)!r}. "
            "This is the --target-model footgun -- regenerate the prepared data "
            "with the correct --target-model."
        )
    return counts


@click.command()
@click.option(
    "--train-jsonl",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Prepared training file (…/prepared/train/train_messages.jsonl)",
)
@click.option(
    "--suffix",
    required=True,
    help="Finetune suffix, i.e. the SFT_ARMS arm name (e.g. causal-base)",
)
@click.option("--base-model", default=DEFAULT_BASE_MODEL, show_default=True)
@click.option("--n-epochs", type=int, default=None, help="Default: OpenAI's auto")
@click.option(
    "--validation-jsonl",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option("--wait/--no-wait", default=True, help="Poll until the job finishes")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the JSONL and print what would be submitted, then stop",
)
def main(
    train_jsonl: Path,
    suffix: str,
    base_model: str,
    n_epochs: Optional[int],
    validation_jsonl: Optional[Path],
    wait: bool,
    dry_run: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    counts = validate_jsonl(train_jsonl)
    logger.info(
        "%s: %d examples, label counts %s", train_jsonl, sum(counts.values()), counts
    )
    if dry_run:
        logger.info("Dry run: would fine-tune %s with suffix %r", base_model, suffix)
        return

    client = _client()
    logger.info("Uploading %s", train_jsonl)
    train_file = client.files.create(file=train_jsonl.open("rb"), purpose="fine-tune")
    validation_file = None
    if validation_jsonl:
        validation_file = client.files.create(
            file=validation_jsonl.open("rb"), purpose="fine-tune"
        )

    kwargs: dict = {
        "training_file": train_file.id,
        "model": base_model,
        "suffix": suffix,
    }
    if validation_file:
        kwargs["validation_file"] = validation_file.id
    if n_epochs:
        kwargs["method"] = {
            "type": "supervised",
            "supervised": {"hyperparameters": {"n_epochs": n_epochs}},
        }

    job = client.fine_tuning.jobs.create(**kwargs)
    logger.info("Created fine-tuning job %s (%s)", job.id, job.status)
    if not wait:
        logger.info(
            "Not waiting; poll with: openai api fine_tuning.jobs.get -i %s", job.id
        )
        return

    while job.status not in DONE_STATES:
        time.sleep(POLL_SECONDS)
        job = client.fine_tuning.jobs.retrieve(job.id)
        logger.info(
            "job %s: %s (trained tokens: %s)", job.id, job.status, job.trained_tokens
        )

    if job.status != "succeeded":
        raise SystemExit(f"Fine-tuning job {job.id} ended as {job.status}: {job.error}")

    logger.info("Fine-tuned model: %s", job.fine_tuned_model)
    checkpoints = client.fine_tuning.jobs.checkpoints.list(job.id)
    for ckpt in checkpoints.data:
        logger.info(
            "  checkpoint: %s (step %s)",
            ckpt.fine_tuned_model_checkpoint,
            ckpt.step_number,
        )
    print(
        json.dumps(
            {suffix.replace("-", "_"): f"openai/{job.fine_tuned_model}"}, indent=1
        )
    )


if __name__ == "__main__":
    main()
