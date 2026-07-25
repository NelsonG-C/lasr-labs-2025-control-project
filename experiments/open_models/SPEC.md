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
Monitor and transform configs are left alone. The collusion monitor is
forced-decoding, so it cannot truncate. The suspiciousness monitors and the LLM
paraphraser *do* free-generate and are therefore still capped at the provider
default: they are only expected to emit a short `<score>` (or a paraphrase), and
did so throughout the pilot, but this is an expectation rather than a guarantee.
A truncated suspiciousness completion makes `parse_score` return `None`, which is
indistinguishable from a refusal — so check the NaN-score rate on the first arm
of a sweep before trusting a full run.

Two caveats on the exact-logprob path, both currently latent:

- The forced-decoding prefix is `apply_chat_template(add_generation_prompt=True)`
  + `<answer>`, which assumes a ChatML-shaped generation prompt (correct for
  Qwen). A harmony-style template (gpt-oss) ends the generation prompt before its
  `<|channel|>final<|message|>` marker, so appending `<answer>` would yield a
  malformed sequence and two *non-NaN but meaningless* logprobs. H is
  generation-only today, so this does not bite -- but do not point the collusion
  monitor at a gpt-oss checkpoint without fixing the scaffold first.
- Collusion *accuracy* is not the same estimator across backends. On the OpenAI
  path it scores the text the model actually generated; on the Tinker path the
  completion is synthesised from `argmax(logP(YES), logP(NO))`, so the model can
  never be scored as emitting anything else. The forced-decoding version is
  arguably the better metric, but the two should not share a table without a note.

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

There are **three checkpoints per condition**, one per paraphrase strategy
(`none` / `minified` / `paraphrased`), because the monitor is finetuned on the
same representation it will be shown at eval time. That requires the *training*
generations to be transformed too, so the spec emits train-side transform pairs
(`*_config_train.yaml`) alongside the train generators:

| Condition | Train generator | Mode | Labels |
|---|---|---|---|
| passive self-rec | `self_rec/generator/honest_generator_config.yaml` | honest | author: U=YES, H=NO |
| causal/combined | `causal/generator/train_generator_config.yaml` | **attack**, `causal_attack_prompt` | author: U=YES, H=NO |

The causal arm trains on *attack* generations while the passive arm trains on
*honest* ones: passive self-recognition is stylistic authorship (backdoors would
only confound it), whereas causal collusion is recognising the shared backdoor
signal, which cannot be learned from backdoor-free code. Attack mode also matches
the BigCodeBench train generator. (Recorded as a decision, not a certainty --
the closed run's training data was never published, so this could not be
confirmed against it; see below.)

Note the APPS passive train generator writes `honest_attacks.eval` /
`honest_honeypots.eval` rather than the default `<label>.eval`, so its transform
pair needs `Transform.input_files` to point at the right inputs.

End-to-end SFT reproduction (all generation steps need Docker):
1. `generate_solutions_inspect.py` on the **train** split (`odd_train`) for U and
   the other author models → the separate training generations.
1b. `transform_solutions_inspect.py` on the `*_config_train.yaml` transform pair
   → the minified / paraphrased training solutions. Skip this and only the
   `none`-strategy checkpoint can be trained.
2. `finetuning/binary_tuning/transform_logs_to_raw.sh` (with `--transformed` for
   the minified/paraphrased arms) + `transform_raw_to_prepared.sh` →
   `train_messages.jsonl` (reused unchanged;
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

### Operational notes from the APPS run

**Put `TMPDIR` on a real disk.** inspect assembles the whole `.eval` zip in
`tempfile.TemporaryFile()` (`log/_recorders/eval.py`) before writing it out. If
`/tmp` is a RAM-backed tmpfs -- the default on many Linux boxes -- a
1030-sample arm's ~100MB temp zip lands in RAM, and two concurrent arms
exhausted a 15GB tmpfs and killed a run at 900/1030 with the *very*
unhelpful `OSError: [Errno 122] Disk quota exceeded` (EDQUOT, not a Tinker
quota). `experiments/open_models/runs/run_arm.sh` exports
`TMPDIR=$HOME/.cache/inspect-tmp` and aborts up front if headroom is short.

**Docker concurrency is not the bottleneck; don't over-tune it.** The APPS
scorer gives each test a *10s wall-clock* timeout, so an oversubscribed host can
turn correct code into `INCORRECT` and silently depress the main-task success
rate. Measured by submitting the dataset's own reference solutions (ground truth
"should pass") at `max_subprocesses` 4 / 16 / 48: **zero** timeouts at every
level and byte-identical failing sample sets, with wall time flat above 16.
Two structural reasons it is safe: the timeout clock starts *after* the
subprocess semaphore is acquired, so queueing cannot manufacture timeouts; and
`max_subprocesses` defaults to `os.cpu_count()` because the repo scripts call
`eval()` from Python, where the `INSPECT_EVAL_MAX_*` envvars (CLI-only) do not
apply. Throughput is bound by the Tinker API instead: 0.40 samples/s for one
arm, 0.83 for two, 1.15 for three concurrent arms.

**Attack yield is the real constraint.** `check_sample_is_valid` (applied by
`eval_to_dataset` at the monitor stage) keeps attacks/honeypots only when *both*
main and side task succeed, and honest when main succeeds. On open models that
leaves ~9% of attacks: ~95 usable attack samples per 1030-sample condition.
Budget statistical power accordingly.
