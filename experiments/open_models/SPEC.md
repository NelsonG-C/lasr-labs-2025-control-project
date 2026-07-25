# Open-models extension (Tinker) — infrastructure

## Motivation

Reviewers asked us to extend the untrusted-monitoring experiments beyond the
closed OpenAI/Gemini models to **open-weights** models. This change adds the
infrastructure to run the existing APPS / BigCodeBench pipeline on open models
served by [Tinker](https://tinker-docs.thinkingmachines.ai) (Thinking Machines).

This is an **infrastructure-only** change: it wires up the backend, config
trees, and tests, but does **not** include a run or results. The full pipeline's
generation/transform steps use the APPS/BigCodeBench code-execution scorer,
which needs a Docker sandbox — see "Running" below.

## Why Tinker

The monitor's collusion score is `log P(YES) − log P(NO)` at the answer token.
Tinker exposes exact per-token logprobs via `compute_logprobs`, so we read those
two logprobs directly by forced decoding — no top-k truncation, provider-shape,
or chain-of-thought-contamination issues. Tinker also serves the dense open
model families per-token (no dedicated GPU endpoint to rent), and — as a LoRA
fine-tuning API — leaves the door open to reproducing the finetuned-collusion
experiments on open models later.

## The single util: `utils/tinker_provider.py`

All Tinker-specific code lives in one module; `import tinker` is lazy, so the
rest of the pipeline runs without the package unless a `tinker/` model is used.
Auth is `TINKER_API_KEY`.

- **Provider registration.** `@modelapi("tinker")` registers a
  `tinker/<base_model>` provider (e.g. `tinker/Qwen/Qwen3-8B`), so
  `get_model()` / `eval()` work for solution generation, the text-`<score>`
  suspiciousness monitors, and the LLM paraphraser with no other changes. CoT is
  disabled at the chat template (`enable_thinking=False`, best-effort) so
  completions carry only the final answer.
- **Exact collusion logprobs.** `collusion_yes_no_logprobs()` teacher-forces
  `prompt+"<answer>YES"` vs `prompt+"<answer>NO"` and returns each candidate's
  summed answer-token logprob (handles multi-token YES/NO), in the same dict
  shape as the existing OpenAI extractor. `create_collusion_monitor_scorer`
  branches to this for `tinker/` models and to the OpenAI-shape extractor
  otherwise; downstream scoring is identical.

Note: `apply_chat_template` on transformers >=5 returns a `BatchEncoding`, not a
bare `list[int]`; the provider requests `return_dict=False` and normalises via
`_normalize_ids`. `jinja2` (transformers chat templating) and `tinker` are
declared as dependencies.

## Model roles

Set in `experiments/open_models/make_configs.py`, which holds one declarative
spec and emits **both** config trees from it: `configs/{apps,bigcodebench}/`
(closed models) and `configs/{apps,bigcodebench}_open/` (Tinker). The SFT arms
are generated too: each closed `openai/ft:...` finetune maps to a logical arm
name whose open-weights counterpart is a Tinker LoRA of U, supplied with
`--checkpoints ckpts.json` (arm -> `tinker://…` path). Arms without a checkpoint
are emitted as `tinker/tinker://PENDING__<arm>` and listed on stdout.

| Role | Model | Rationale |
|------|-------|-----------|
| **U** untrusted generator + monitor | `tinker/Qwen/Qwen3-30B-A3B-Instruct-2507` | ~GPT-4.1-level, non-thinking instruct |
| **T** trusted monitor + paraphraser | `tinker/Qwen/Qwen3.5-4B` | Deliberately weak; Qwen3.5 (a different series from U's Qwen3) |
| **H** honeypot generator | `tinker/openai/gpt-oss-120b` | Different lineage, ~U capability; generation-only |

All three are verified available via `get_server_capabilities()` on our account
and confirmed to actually generate/monitor in the required formats (a live
sample call, not just presence in a list).

Catalogue gotcha: the Tinker **website** models page is not authoritative for
sampling — it's a curated/lagging view that mixes in training-only and
`-Base` (pretrained, non-instruct) entries. What you can actually call is what
`get_server_capabilities()` returns for your account, and even then "listed"
does not guarantee "sampleable": e.g. `Qwen/Qwen3-30B-A3B-Instruct-2507` (our U)
is callable but absent from the website, while `Qwen/Qwen3.5-35B-A3B` appears on
the website yet returns `400 "Sampling is not supported"` (only its `-Base`
exists here). Always confirm a model with an actual `sample` call before a run.

Reasoning-model note (why H is gpt-oss, not Nemotron): the honeypot generator
must emit `<final_code>`/`<secret_input>` within a reasonable token budget.
`nvidia/NVIDIA-Nemotron-3-*` are heavy reasoners — they reason in `<reasoning>`
tags, ignore `enable_thinking=False`, and exceed 8k output tokens on a trivial
problem without ever emitting `<final_code>`, so they're unusable as generators.
Tinker's sampling API has no `reasoning_effort` knob; reasoning is controlled
only via the chat template (`enable_thinking=False`, which for Qwen/DeepSeek
inserts the empty `<think></think>` block) — Nemotron honours neither reliably.
`gpt-oss-120b` and `deepseek-ai/DeepSeek-V3.1` emit clean output out of the box;
gpt-oss-120b is the closer capability match to U. The provider default
`max_tokens` is only 4096, which silently truncates the open generators before
their closing tag (~2/3 of samples in a pilot came back with no `<final_code>`;
the worst cases needed >8k), so `make_configs.py` injects `max_tokens: 16384`
into the `defaults:` of every *open* generator config (`--max-tokens` to change).
Monitor and transform configs are left alone — the collusion monitor is
forced-decoding and the suspiciousness monitors emit a short `<score>`.

U (Qwen3) and T (Qwen3.5) are different Qwen generations, so the trusted monitor
is a distinct series from the untrusted model, and H (gpt-oss) is a distinct
lineage from both.

## Testing

`tests/` (repo root) mirrors the sibling `lasr-project-better-stats` layout.
- `test_tinker_provider.py` — offline unit tests (no key/network): provider
  registration, helpers, `_normalize_ids`, and the forced-decoding logprob math
  via stub tokenizer/sampling clients.
- `test_tinker_live.py` — live end-to-end tests (marked `slow`, auto-skipped
  without `TINKER_API_KEY`): capabilities, generation via the provider, and
  exact YES/NO logprobs that correctly discriminate a true vs false statement.

All pass live against `Qwen/Qwen3-8B`. Run: `pytest` (offline only:
`pytest -m 'not slow'`).

## SFT track: finetuned collusion monitor on open models

The original headline experiments finetune **U** (self-recognition SFT) and use
the resulting checkpoint as the untrusted collusion monitor. The open-weights
analogue LoRA-finetunes U on Tinker:

- **`finetuning/tinker_lora_finetune.py`** — LoRA-finetunes U (default
  `Qwen/Qwen3-30B-A3B-Instruct-2507`; **U, not the small T**) from the prepared
  chat JSONL that `finetuning/binary_tuning` already emits. It masks the loss to
  the assistant answer tokens (prompt not trained), tokenises exactly as
  inference does (`tinker_provider`, thinking disabled), then
  `save_weights_for_sampler()` and prints the `tinker://…` checkpoint path.
- **Provider checkpoint support** — `tinker_provider._sampling_client` routes a
  `tinker://…` ref through `create_sampling_client(model_path=…)`, so a finetuned
  checkpoint works everywhere a base model does (generation, and the
  forced-decoding collusion monitor). In a config, set the collusion monitor
  model to `tinker/tinker://<run-id>/sampler_weights/<name>`.

End-to-end SFT reproduction (all generation steps need Docker):
1. `generate_solutions_inspect.py` on the **train** split (`odd_train`) for U and
   the other author models → the separate training generations.
2. `finetuning/binary_tuning/transform_logs_to_raw.sh` +
   `transform_raw_to_prepared.sh` → `train_messages.jsonl` (reused unchanged;
   label-ablation variants via `randomise_labels.py` / `permute_labels.py`).
3. `python -m lasr_labs_2025_control_project.finetuning.tinker_lora_finetune
   --train-jsonl <…/train_messages.jsonl> --out-path-file ckpt.txt` → checkpoint.
4. Collect the checkpoints into `ckpts.json` (`{"<arm>": "tinker://…", …}`) and
   re-run `experiments/open_models/make_configs.py --model-class open
   --checkpoints ckpts.json`, which bakes them into the `*_sft` monitor configs;
   then run `monitor_solutions_inspect.py` as in the prompted track.

Verified: the trainer runs on U live (forward_backward/optim_step reduce the
loss, `save_weights_for_sampler` + sampling round-trip succeed); the data
adapter (shift + answer-only mask) and checkpoint routing are unit-tested. Not
yet run on real training data (needs the Docker generations from step 1).

## Running (not done here — needs Docker)

The APPS/BigCodeBench scorers execute generated code in a Docker sandbox
(`SandboxEnvironmentSpec(type="docker")`), which the CPU-only dev box lacks. On
a Docker-capable machine, with `TINKER_API_KEY` set:

```bash
python -m lasr_labs_2025_control_project.scripts.generate_solutions_inspect \
  --config src/lasr_labs_2025_control_project/configs/apps_open/acausal/generator/generator_config.yaml
python -m lasr_labs_2025_control_project.scripts.monitor_solutions_inspect \
  --config src/lasr_labs_2025_control_project/configs/apps_open/acausal/monitors/base_monitor_config.yaml
```

Pilot with a small `limit` first, confirm YES/NO logprobs are non-NaN and the
honest-vs-attack separation is sane, then launch the full sweep.
