"""LoRA fine-tune the untrusted model (U) on the self-recognition task via Tinker.

This is the open-weights analogue of the OpenAI fine-tuning step in the original
SFT experiments. It trains a LoRA adapter on **U** (default
`Qwen/Qwen3-30B-A3B-Instruct-2507`, a 30B model) from the prepared chat JSONL
produced by `finetuning/binary_tuning` — each line is
`{"messages": [system, user, assistant]}` where the assistant content is
`<answer>YES</answer>` / `<answer>NO</answer>` (the self-recognition label).

Only the assistant answer tokens contribute to the loss (the prompt is masked
out), and tokenisation matches inference (`utils.tinker_provider`, thinking
disabled), so the resulting checkpoint behaves consistently when plugged into
the collusion monitor.

It saves sampler weights and prints the `tinker://…` checkpoint path; put that
path in a monitor config's `untrusted_collusion_monitor` model field
(the provider loads `tinker://…` via `model_path`).

Usage:
  python -m lasr_labs_2025_control_project.finetuning.tinker_lora_finetune \
      --train-jsonl <prepared/train/train_messages.jsonl> \
      [--base-model Qwen/Qwen3-30B-A3B-Instruct-2507] \
      [--rank 32] [--lr 1e-4] [--epochs 3] [--batch-size 64] \
      [--checkpoint-name self_rec_u] [--out-path-file checkpoint_path.txt] \
      [--max-examples N] [--seed 0]
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator, Optional

import click

from lasr_labs_2025_control_project.utils import tinker_provider as tp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The untrusted model. We SFT U, not the (small) trusted model — keep this as
# the default and do not point it at T.
DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def load_examples(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read prepared training JSONL; each line has a `messages` list."""
    examples: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(
                    f"line missing 'messages' key: {line[:120]}"
                )
            examples.append(obj)
    if not examples:
        raise ValueError(f"no examples found in {jsonl_path}")
    return examples


def build_datum(messages: list[dict[str, str]], base_model: str):
    """Turn one chat example into a Tinker cross-entropy Datum.

    The full sequence is the rendered [system, user] prompt + the assistant
    answer tokens + EOS. Next-token targets are the sequence shifted by one, and
    weights are 1.0 only on positions that predict an answer/EOS token (the
    prompt is not trained on).
    """
    from tinker import types

    tok = tp._tokenizer(base_model)
    prompt_msgs = [m for m in messages if m["role"] in ("system", "user")]
    assistant = next(m for m in messages if m["role"] == "assistant")

    prompt_ids = list(
        tp._apply_chat_template(base_model, prompt_msgs, add_generation_prompt=True)
    )
    answer_ids = list(tok.encode(assistant["content"], add_special_tokens=False))
    eos_id = getattr(tok, "eos_token_id", None)
    tail_ids = answer_ids + ([eos_id] if eos_id is not None else [])

    tokens = prompt_ids + tail_ids
    if len(tokens) < 2 or not tail_ids:
        raise ValueError("example too short after tokenisation")

    model_input_ids = tokens[:-1]
    target_tokens = tokens[1:]
    # target position i predicts tokens[i+1]; train only where that target is a
    # tail (answer/EOS) token, i.e. index i+1 >= len(prompt_ids).
    weights = [1.0 if (i + 1) >= len(prompt_ids) else 0.0 for i in range(len(target_tokens))]

    return types.Datum(
        model_input=types.ModelInput.from_ints(model_input_ids),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def _batches(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _mean_loss(fb_output: Any, batch: list[Any]) -> Optional[float]:
    """Best-effort weighted mean NLL for logging.

    cross_entropy returns per-datum logprobs of the target tokens; the training
    loss is the weight-weighted negative logprob. Returns None if the output
    shape isn't as expected (logging only — never fails the run).
    """
    try:
        outs = fb_output.loss_fn_outputs
        total_nll, total_w = 0.0, 0.0
        for datum, out in zip(batch, outs):
            lp = out.get("logprobs")
            if lp is None:
                return None
            lp_vals = list(getattr(lp, "data", lp))
            w_vals = list(datum.loss_fn_inputs["weights"].data)
            for logp, w in zip(lp_vals, w_vals):
                total_nll += -float(logp) * float(w)
                total_w += float(w)
        return total_nll / total_w if total_w > 0 else None
    except Exception:
        return None


def finetune(
    train_jsonl: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    rank: int = 32,
    lr: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 64,
    checkpoint_name: str = "self_rec_u",
    max_examples: Optional[int] = None,
    seed: int = 0,
) -> str:
    """Run LoRA SFT and return the `tinker://…` sampler-weights path."""
    import tinker
    from tinker import types

    examples = load_examples(train_jsonl)
    if max_examples is not None:
        examples = examples[:max_examples]
    logger.info(
        "Loaded %d examples; base_model=%s rank=%d lr=%g epochs=%d batch_size=%d",
        len(examples), base_model, rank, lr, epochs, batch_size,
    )

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=base_model, rank=rank
    )

    data = [build_datum(ex["messages"], base_model) for ex in examples]
    rng = random.Random(seed)

    step = 0
    for epoch in range(epochs):
        rng.shuffle(data)
        for batch in _batches(data, batch_size):
            fb_future = training_client.forward_backward(batch, "cross_entropy")
            opt_future = training_client.optim_step(
                types.AdamParams(learning_rate=lr)
            )
            fb_output = fb_future.result()
            opt_future.result()
            step += 1
            loss = _mean_loss(fb_output, batch)
            logger.info(
                "epoch %d step %d (%d examples)%s",
                epoch, step, len(batch),
                f" loss={loss:.4f}" if loss is not None else "",
            )

    resp = training_client.save_weights_for_sampler(name=checkpoint_name).result()
    logger.info("Saved sampler weights: %s", resp.path)
    return resp.path


@click.command()
@click.option("--train-jsonl", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--base-model", type=str, default=DEFAULT_BASE_MODEL, show_default=True)
@click.option("--rank", type=int, default=32, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--epochs", type=int, default=3, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--checkpoint-name", type=str, default="self_rec_u", show_default=True)
@click.option("--max-examples", type=int, default=None)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--out-path-file",
    type=click.Path(path_type=Path),
    default=None,
    help="If given, write the tinker:// checkpoint path here for config generation.",
)
def main(
    train_jsonl: Path,
    base_model: str,
    rank: int,
    lr: float,
    epochs: int,
    batch_size: int,
    checkpoint_name: str,
    max_examples: Optional[int],
    seed: int,
    out_path_file: Optional[Path],
):
    path = finetune(
        train_jsonl=train_jsonl,
        base_model=base_model,
        rank=rank,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_name=checkpoint_name,
        max_examples=max_examples,
        seed=seed,
    )
    click.echo(path)
    if out_path_file is not None:
        out_path_file.write_text(path + "\n")


if __name__ == "__main__":
    main()
