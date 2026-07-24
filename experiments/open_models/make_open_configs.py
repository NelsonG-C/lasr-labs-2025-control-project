"""Generate open-model (Tinker) config trees from the closed-model configs.

Copies configs/apps -> configs/apps_open and configs/bigcodebench ->
configs/bigcodebench_open, swapping the closed U/T/H models for open-weights
models served by Tinker (via the tinker/ inspect provider registered in
utils/tinker_provider.py), and repointing experiment folders at *_open so the
original results are never clobbered.

Configs that reference OpenAI finetuned models (openai/ft:...) are skipped:
those SFT experiments are OpenAI-specific. Reproducing them on open weights is a
separate follow-up (Tinker is a LoRA fine-tuning API, so it is feasible, but out
of scope for this infrastructure change).

Model roles (verify availability with
`tinker.ServiceClient().get_server_capabilities().supported_models` before a run
-- the live catalogue changes):
  U (untrusted generator + collusion/backdoor monitor): ~GPT-4.1-level
  T (trusted backdoor monitor + LLM paraphraser):        deliberately weak
  H (honeypot generator only):                           different lineage, ~U cap.

H must follow the generation format (emit <final_code>/<secret_input>) within a
reasonable token budget. Heavy-reasoning models are unsuitable as generators:
NVIDIA Nemotron-3 reasons in <reasoning> tags, ignores enable_thinking, and
exceeds 8k tokens on trivial problems without emitting <final_code>. gpt-oss-120b
and DeepSeek-V3.1 emit clean output out of the box; gpt-oss-120b is the closer
capability match to U.

Run from the repo root: python experiments/open_models/make_open_configs.py
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "src/lasr_labs_2025_control_project/configs"

# closed model -> open (Tinker) replacement
MODEL_MAP = {
    "openai/gpt-4.1-2025-04-14": "tinker/Qwen/Qwen3-30B-A3B-Instruct-2507",  # U
    "openai/gpt-4o-mini": "tinker/Qwen/Qwen3-8B",  # T (weak, same Qwen3 series)
    "google/gemini-2.5-flash": "tinker/openai/gpt-oss-120b",  # H (different lineage, ~U cap.)
}

skipped, written = [], []
for setting in ["apps", "bigcodebench"]:
    for src in sorted((CONFIGS / setting).rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(CONFIGS / setting)
        dst = CONFIGS / f"{setting}_open" / rel
        text = src.read_text()
        if src.suffix == ".yaml" and "openai/ft:" in text:
            skipped.append(str(rel))
            continue
        for old, new in MODEL_MAP.items():
            text = text.replace(old, new)
        text = text.replace(
            f"experiments/{setting}", f"experiments/{setting}_open"
        ).replace(f"configs/{setting}/", f"configs/{setting}_open/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        written.append(str(dst.relative_to(CONFIGS)))

print(f"wrote {len(written)} files, skipped {len(skipped)} finetune configs")
for s in skipped:
    print("  skipped:", s)
