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

Set in `experiments/open_models/make_open_configs.py`, which generates
`configs/apps_open/` and `configs/bigcodebench_open/` from the closed-model
configs (skipping the 12 `openai/ft:...` SFT configs, which are OpenAI-specific).

| Role | Model | Rationale |
|------|-------|-----------|
| **U** untrusted generator + monitor | `tinker/Qwen/Qwen3-30B-A3B-Instruct-2507` | ~GPT-4.1-level, non-thinking instruct |
| **T** trusted monitor + paraphraser | `tinker/Qwen/Qwen3-8B` | Deliberately weak, same Qwen3 series |
| **H** honeypot generator | `tinker/openai/gpt-oss-120b` | Different lineage, ~U capability; generation-only |

All three are verified available via `get_server_capabilities()` on our account
and confirmed to generate/monitor in the required formats.

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
`max_tokens` is 4096 so generators reach their closing tag.

Caveat to revisit before the real run:
- T (`Qwen3-8B`) is a smaller same-series model rather than a literal "previous
  series"; the capability gap to U is what matters for the trusted-weak role.

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
