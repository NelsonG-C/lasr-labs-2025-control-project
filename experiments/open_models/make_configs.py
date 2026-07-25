"""Generate every experiment config tree (closed-model and open-model) from one spec.

The declarative spec at the top of this module is the source of truth; the YAML
under `src/lasr_labs_2025_control_project/configs/{apps,bigcodebench}[_open]/` is a
generated artifact. Hand-edits to those files will be overwritten -- change the
spec instead.

    python experiments/open_models/make_configs.py                     # both trees
    python experiments/open_models/make_configs.py --model-class open  # open only
    python experiments/open_models/make_configs.py --model-class open --settings bigcodebench

A config is fully determined by a handful of axes:

* **generator** -- which policy prompt each of the attacks/honest/honeypots runs
  uses, and which model role produces it.
* **transform** -- minifier or LLM-paraphraser over the generator's `.eval`s.
* **monitor** -- the cross product of
  - *condition* (`acausal` / `causal` / `self_rec`) -> the untrusted collusion
    monitor's system prompt. Note the "combined" arms (apps `causal/*_sft` and
    apps `additive`) deliberately use the SELF_REC collusion prompt even though
    they live under the causal experiment;
  - *paraphrase strategy* (`none` / `minified` / `paraphrased`) -> the honest
    backdoor-monitor prompt, the `transformed` flag, and which `.eval` is read;
  - *base or SFT* -> the untrusted **collusion monitor's model only** (base U vs
    a fine-tuned checkpoint of U). Every other model is unchanged.

Model roles
-----------
| role | closed                      | open (Tinker)                          |
|------|-----------------------------|----------------------------------------|
| U    | `openai/gpt-4.1-2025-04-14` | `tinker/Qwen/Qwen3-30B-A3B-Instruct-2507` |
| T    | `openai/gpt-4o-mini`        | `tinker/Qwen/Qwen3.5-4B`               |
| H    | `google/gemini-2.5-flash`   | `tinker/openai/gpt-oss-120b`           |

U is the untrusted policy + untrusted monitors, T the deliberately-weak trusted
monitor and LLM paraphraser, H the honeypot generator (different lineage, ~U
capability). Verify the open models are actually callable on your account with
`tinker.ServiceClient().get_server_capabilities().supported_models` *and* a live
`sample` call before a run -- the Tinker catalogue changes, the website listing
is not authoritative, and "listed" does not imply "sampleable"
(e.g. `Qwen/Qwen3.5-35B-A3B` is listed but returns 400 "Sampling is not
supported").

H must follow the generation format -- emit `<final_code>` / `<secret_input>` --
within a reasonable token budget, which rules out heavy reasoners:
`nvidia/NVIDIA-Nemotron-3-*` reason inside `<reasoning>` tags, ignore
`enable_thinking=False`, and blow past 8k output tokens on trivial problems
without ever emitting `<final_code>`. `gpt-oss-120b` and `deepseek-ai/DeepSeek-V3.1`
emit clean output out of the box; gpt-oss-120b is the closer capability match to U.

SFT arms
--------
Six SFT runs = 3 paraphrase strategies x 2 conditions (passive self-recognition,
causal/combined), plus 2 passive self-rec label controls (honest labels,
permuted labels) = 8 arms (`SFT_ARMS`). BigCodeBench adds 3 more of its own.
On closed models an arm is an OpenAI finetune; on open weights it is a Tinker
LoRA of U (`finetuning/tinker_lora_finetune.py`), supplied as

    --checkpoints ckpts.json   # {"self_rec_base": "tinker://<run>/sampler_weights/<name>", ...}

Arms without a checkpoint are emitted as `tinker/tinker://PENDING__<arm>` and
listed on stdout, so the tree is always complete and you can re-run this script
to fill checkpoints in as each LoRA finishes.

Output layout
-------------
`experiment_folder` mirrors the HF dataset repo `jbostock/untrusted-monitoring-2026-paper`:

    experiments/<model-class>/<setting>/<experiment>/{configs,generator,transforms/<t>,outputs/<arm>}

with `<model-class>` in {`closed-source-model`, `open-source-model`}. Uploading is
then just `experiments/<class>/**` -> HF `<class>/**`. `experiments/` is already
gitignored. Keeping the model class in the path is also what stops the open run
from clobbering the closed run's SFT *training* generations, which used to live
at the un-prefixed `experiments/self_rec_train`.

Open tree only: generator token budget
--------------------------------------
Every open generator config gets `max_tokens: 16384` in its `defaults:` block
(`--max-tokens` to change). The open generators write a `<reasoning>` section
before `<final_code>` and were silently truncated at the tinker provider's 4096
default, producing empty solutions -- roughly two thirds of samples in a pilot,
with the worst cases needing >8k tokens. The plumbing is `RunGenerator.max_tokens`
in `utils/config_loader.py`. Monitor and transform configs are left alone: the
collusion monitor is forced-decoding (it generates nothing), and while the
suspiciousness monitors and LLM paraphraser do free-generate, they are only
expected to emit a short `<score>` (or a paraphrase) and stay well inside the
provider default. That is an expectation, not a guarantee -- a truncated
suspiciousness completion yields a `None` score, so check the NaN rate on the
first arm of a sweep.

Full run order for a fresh open-weights run
-------------------------------------------
1. `python experiments/open_models/make_configs.py --model-class open`
2. `scripts/generate_solutions_inspect.py --config .../apps_open/self_rec/generator/honest_generator_config.yaml`
   (the **train** split -- the SFT training generations)
3. `finetuning/binary_tuning/transform_logs_to_raw.sh` -> `transform_raw_to_prepared.sh`,
   per SFT arm (label ablations via `randomise_labels.py` / `permute_labels.py`)
4. `finetuning/tinker_lora_finetune.py --train-jsonl ...` per arm -> a `tinker://` path each
5. re-run this script with `--checkpoints ckpts.json` to bake the paths in
6. `generate_solutions_inspect.py` on the **test** split -> `transform_solutions_inspect.py`
   -> `monitor_solutions_inspect.py`
7. `scripts/plot_data.py` for histograms / ROC / safety-usefulness curves

Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "src/lasr_labs_2025_control_project/configs"

DEFAULT_MAX_TOKENS = 16384

MODEL_CLASS_DIRS = {"closed": "closed-source-model", "open": "open-source-model"}

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

#: role -> model string, per model class. U untrusted, T trusted/weak, H honeypot.
ROLE_MODELS: dict[str, dict[str, str]] = {
    "closed": {
        "U": "openai/gpt-4.1-2025-04-14",
        "T": "openai/gpt-4o-mini",
        "H": "google/gemini-2.5-flash",
    },
    "open": {
        "U": "tinker/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "T": "tinker/Qwen/Qwen3.5-4B",
        "H": "tinker/openai/gpt-oss-120b",
    },
}

#: SFT arm -> the closed-model OpenAI finetune that implements it. The open-model
#: counterpart is a Tinker LoRA of U supplied via --checkpoints.
SFT_ARMS: dict[str, str] = {
    # passive self-recognition, one arm per paraphrase strategy
    "self_rec_base": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:gemini-2-5-flash:CJvxwvyn",
    "self_rec_minified": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:gemini-2-5-flash-minified:CKNzgVEI",
    "self_rec_paraphrased": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:gemini-2-5-flash-paraphrased:CK1qHy0v",
    # passive self-recognition label controls
    "self_rec_honest": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:gemini-2-5-flash-honest:CQmnG0bI",
    "self_rec_permuted": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:gemini-2-5-flash-honest-permute:CQm2ry0X",
    # causal / combined, one arm per paraphrase strategy
    "causal_base": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:causal-base:CSQTsO5j:ckpt-step-2100",
    "causal_minified": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:causal-minified:CSRCpAfQ:ckpt-step-2100",
    "causal_paraphrased": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:causal-paraphrased:CS8Pm0LQ:ckpt-step-2100",
    # BigCodeBench self-recognition SFT (only referenced by the bigcodebench tree)
    "bcb_self_rec_base": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:bigcodebench-none:CRHKEcKW",
    "bcb_self_rec_minified": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:bigcodebench-minified:CRHPL9Uc",
    "bcb_self_rec_paraphrased": "openai/ft:gpt-4.1-2025-04-14:lasr-labs:bigcodebench-paraphrased:CRIEKOFs",
}

# ---------------------------------------------------------------------------
# prompt / strategy tables
# ---------------------------------------------------------------------------

#: condition -> untrusted collusion monitor system prompt
COLLUSION_PROMPTS = {
    "acausal": "ACAUSAL_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT",
    "causal": "CAUSAL_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT",
    "self_rec": "SELF_REC_UNTRUSTED_MONITOR_COLLUSION_SYSTEM_PROMPT",
}

#: paraphrase strategy -> honest (suspiciousness) monitor system prompt
BACKDOOR_PROMPTS = {
    "none": "DEFAULT_HONEST_MONITOR_SYSTEM_PROMPT",
    "minified": "MINIFIED_HONEST_MONITOR_SYSTEM_PROMPT",
    "paraphrased": "PARAPHRASED_HONEST_MONITOR_SYSTEM_PROMPT",
}

#: paraphrase strategy -> config filename and display suffix
STRATEGY_FILES = {
    "none": "base_monitor_config.yaml",
    "minified": "minified_monitor_config.yaml",
    "paraphrased": "paraphrased_monitor_config.yaml",
}
STRATEGY_TITLES = {"none": "None", "minified": "Minified", "paraphrased": "Summarised"}

STRATEGIES = ("none", "minified", "paraphrased")
ALL_LABELS = ("attacks", "honest", "honeypots")

#: run label -> stem used in transform/monitor artefact filenames ("attacks" -> "attack")
LABEL_STEMS = {"attacks": "attack", "honest": "honest", "honeypots": "honeypots"}

TRANSFORM_DIRS = {"minifier": "minified", "llm": "paraphrased"}

MONITOR_ORDER = (
    "untrusted_collusion_monitor",
    "trusted_backdoor_monitor",
    "untrusted_backdoor_monitor",
)

# ---------------------------------------------------------------------------
# spec dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenRun:
    """One run of a generator config."""

    label: str
    eval_mode: str  # "attack" | "honest"
    system_prompt: str
    role: str  # key into ROLE_MODELS
    save_path: str = ""  # defaults to generator/<label>.eval


@dataclass(frozen=True)
class Generator:
    """A generator config: which policy writes which `generator/*.eval`."""

    path: str  # relative to the setting's config dir
    experiment: str  # <experiment> segment of experiment_folder
    experiment_name: str
    runs: tuple[GenRun, ...]
    results_folder: str = "results/"
    data_split: str = "test"
    extra_defaults: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transform:
    """A transform config: minifier or LLM paraphraser over the generations."""

    path: str
    experiment: str
    experiment_name: str
    kind: str  # "minifier" | "llm"
    labels: tuple[str, ...] = ALL_LABELS
    #: label -> generator filename, when the producing GenRun used a non-default
    #: save_path. Mirrors Monitor.input_files. Needed for the APPS SFT train
    #: generator, whose runs are labelled honest/honeypots but write
    #: honest_attacks.eval / honest_honeypots.eval.
    input_files: Mapping[str, str] | None = None

    @property
    def transform_dir(self) -> str:
        return TRANSFORM_DIRS[self.kind]

    def eval_path(self, label: str) -> str:
        if self.input_files and label in self.input_files:
            return f"generator/{self.input_files[label]}"
        return f"generator/{label}.eval"


@dataclass(frozen=True)
class Monitor:
    """A monitor config: (condition, strategy, base-or-SFT) plus its output arm."""

    path: str
    experiment: str
    experiment_name: str
    results_folder: str
    condition: str  # key into COLLUSION_PROMPTS
    strategy: str  # key into BACKDOOR_PROMPTS
    arm: str  # outputs/<arm>/...
    collusion: str  # "U" (role) or an SFT_ARMS key
    labels: tuple[str, ...] = ALL_LABELS
    input_dir: str | None = None  # default: generator/ or transforms/<strategy>/
    input_files: Mapping[str, str] | None = None  # label -> filename override
    disable_untrusted_monitor: bool = False

    def eval_path(self, label: str) -> str:
        if self.input_files and label in self.input_files:
            name = self.input_files[label]
        elif self.strategy == "none":
            name = f"{label}.eval"
        else:
            name = f"{LABEL_STEMS[label]}_transformed.eval"
        directory = self.input_dir or (
            "generator" if self.strategy == "none" else f"transforms/{self.strategy}"
        )
        return f"{directory}/{name}"

    def save_path(self, label: str) -> str:
        return f"outputs/{self.arm}/{LABEL_STEMS[label]}_monitor_results.eval"


#: The honeypot generator H is given the SAME attack prompt in EVERY condition.
#: Honeypots are a blue-team instrument -- the "not written by U" negative class
#: for the collusion monitor -- so their distribution must not depend on the red
#: team's strategy. If it does, a cross-condition AUROC compares different
#: negative classes and is not a like-for-like number.
#:
#: APPS causal previously handed H the `causal_attack_prompt`, which
#: gpt-oss-120b refused on 16.3% of samples (vs ~1% for the shared prompt),
#: silently shrinking and biasing that arm's negative class. The closed run had
#: the same asymmetry, so this is a deliberate divergence from it.
HONEYPOT_ATTACK_PROMPT = {
    "apps": "self_rec_subtle_attack_policy_prompt",
    "bigcodebench": "self_rec_attack_policy_prompt_bcb",
}

#: Shared generations (OPEN tree only). The whole APPS sweep needs FIVE
#: generations, not nine:
#:
#:     shared honeypots  (H, default attack prompt)  -- used by every condition
#:     shared honest     (U, honest_policy_prompt)   -- used by every condition
#:     attacks x3        (no-signal / acausal / causal)
#:
#: Only the attacks depend on the red team's strategy. Sharing the rest is not
#: just cheaper: the LLM paraphraser is nondeterministic, so re-transforming the
#: same honest code per condition would hand each arm a *different* negative
#: class and inject cross-condition noise into exactly the comparison we care
#: about. The sharing therefore covers the transforms too, not only generation.
#:
#: experiment -> (source experiment, labels taken from it). Paths are emitted as
#: `../<source>/...`, which resolves because every experiment_folder is a
#: sibling under `<model-class>/<setting>/`.
#:
#: OPEN ONLY. The closed run really did generate per-condition honest/honeypots
#: -- they are distinct sample sets in the HuggingFace dataset -- so re-pointing
#: the closed configs at a shared set would silently change published numbers.
SHARED_INPUTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "acausal": ("self_rec", ("honest", "honeypots")),
    "causal": ("self_rec", ("honest", "honeypots")),
    "causal_train": ("self_rec_train", ("honeypots",)),
}


def shared_source(model_class: str, experiment: str, label: str) -> str | None:
    """The experiment supplying `label` to `experiment`, or None if it owns it."""
    if model_class != "open":
        return None
    entry = SHARED_INPUTS.get(experiment)
    if entry and label in entry[1]:
        return entry[0]
    return None


@dataclass(frozen=True)
class Setting:
    """Everything generated for one setting (apps / bigcodebench)."""

    generators: tuple[Generator, ...]
    transforms: tuple[Transform, ...]
    monitors: tuple[Monitor, ...]


# ---------------------------------------------------------------------------
# spec helpers (the regular families; irregular configs are spelled out below)
# ---------------------------------------------------------------------------


def monitor_family(
    *,
    folder: str,
    experiment: str,
    condition: str,
    title: str,
    suffix: str,  # "base" | "sft" -- names both the arm and the results folder
    collusion: str | Mapping[str, str],
    strategies: Iterable[str] = STRATEGIES,
) -> list[Monitor]:
    """The standard 3-config monitor family (none / minified / paraphrased)."""
    out = []
    for strategy in strategies:
        model = collusion if isinstance(collusion, str) else collusion[strategy]
        out.append(
            Monitor(
                path=f"{folder}/{STRATEGY_FILES[strategy]}",
                experiment=experiment,
                experiment_name=f"{title} {STRATEGY_TITLES[strategy]}",
                results_folder=f"results/{strategy}_{suffix}/",
                condition=condition,
                strategy=strategy,
                arm=f"{strategy}_{suffix}",
                collusion=model,
            )
        )
    return out


def transform_pair(
    *,
    folder: str,
    experiment: str,
    experiment_name: str,
    file_suffix: str = "",
    labels: tuple[str, ...] = ALL_LABELS,
    input_files: Mapping[str, str] | None = None,
) -> list[Transform]:
    """The standard minifier + LLM-paraphraser transform pair."""
    return [
        Transform(
            path=f"{folder}/{TRANSFORM_DIRS[kind]}_config{file_suffix}.yaml",
            experiment=experiment,
            experiment_name=experiment_name,
            kind=kind,
            labels=labels,
            input_files=input_files,
        )
        for kind in ("minifier", "llm")
    ]


# ---------------------------------------------------------------------------
# THE SPEC
# ---------------------------------------------------------------------------

APPS = Setting(
    generators=(
        Generator(
            path="acausal/generator/generator_config.yaml",
            experiment="acausal",
            experiment_name="acausal",
            runs=(
                GenRun("attacks", "attack", "acausal_attack_base_prompt", "U"),
                GenRun("honest", "honest", "honest_policy_prompt", "U"),
                GenRun(
                    "honeypots", "attack", "self_rec_subtle_attack_policy_prompt", "H"
                ),
            ),
        ),
        Generator(
            path="causal/generator/generator_config.yaml",
            experiment="causal",
            experiment_name="causal",
            runs=(
                GenRun("attacks", "attack", "causal_attack_prompt", "U"),
                GenRun("honest", "honest", "honest_policy_prompt", "U"),
                # H always gets the shared honeypot prompt, never the red team's.
                GenRun("honeypots", "attack", HONEYPOT_ATTACK_PROMPT["apps"], "H"),
            ),
        ),
        Generator(
            path="self_rec/generator/generator_config.yaml",
            experiment="self_rec",
            experiment_name="self_rec_generatorx",
            runs=(
                GenRun(
                    "attacks", "attack", "self_rec_subtle_attack_policy_prompt", "U"
                ),
                GenRun("honest", "honest", "honest_policy_prompt", "U"),
                GenRun(
                    "honeypots", "attack", "self_rec_subtle_attack_policy_prompt", "H"
                ),
            ),
        ),
        # SFT training data for the causal/combined arms (causal_base,
        # causal_minified, causal_paraphrased). Attack mode under the causal attack
        # prompt from both authors, labelled by author (U=YES / H=NO): causal
        # collusion is about recognising the shared backdoor signal, so unlike the
        # passive arm below the training solutions must actually contain backdoors.
        # Attack mode also matches the BigCodeBench train generator.
        Generator(
            path="causal/generator/train_generator_config.yaml",
            experiment="causal_train",
            experiment_name="causal_train",
            data_split="train",
            runs=(
                GenRun("attacks", "attack", "causal_attack_prompt", "U"),
                GenRun("honeypots", "attack", HONEYPOT_ATTACK_PROMPT["apps"], "H"),
            ),
        ),
        # SFT training data for the passive self-recognition arms
        # (self_rec_base / _minified / _paraphrased). Its own experiment folder,
        # under the model class, so the closed and open runs cannot overwrite
        # each other.
        #
        # ATTACK mode for both authors, under the same default attack prompt
        # (passive = no signal). The monitor is finetuned to answer exactly the
        # question it faces at eval time -- "did I write this backdoored
        # solution, or did H?" -- so the training distribution must be the eval
        # distribution. Training on honest, backdoor-free code (as this
        # generator used to) teaches authorship of a distribution the monitor
        # never sees. BigCodeBench's train generator already used attack mode.
        Generator(
            path="self_rec/generator/train_generator_config.yaml",
            experiment="self_rec_train",
            experiment_name="self_rec_train",
            data_split="train",
            runs=(
                GenRun(
                    "attacks", "attack", "self_rec_subtle_attack_policy_prompt", "U"
                ),
                GenRun("honeypots", "attack", HONEYPOT_ATTACK_PROMPT["apps"], "H"),
            ),
        ),
    ),
    transforms=(
        *transform_pair(
            folder="acausal/transforms", experiment="acausal", experiment_name="acausal"
        ),
        *transform_pair(
            folder="causal/transforms", experiment="causal", experiment_name="causal"
        ),
        # Causal SFT training-data transforms, for causal_minified / causal_paraphrased.
        *transform_pair(
            folder="causal/transforms",
            experiment="causal_train",
            experiment_name="causal_train",
            file_suffix="_train",
            labels=("attacks", "honeypots"),
        ),
        *transform_pair(
            folder="self_rec/transforms",
            experiment="self_rec",
            experiment_name="self_rec",
        ),
        # SFT training-data transforms. The self_rec_minified / self_rec_paraphrased
        # checkpoints must be trained on *transformed* training solutions --
        # binary_tuning's `transform_logs_to_raw.sh --transformed` reads the
        # transformed_solution out of these logs -- so the train split needs its own
        # minifier/paraphraser pair, exactly as BigCodeBench already has. Without
        # these, only the `none` strategy could be trained.
        *transform_pair(
            folder="self_rec/transforms",
            experiment="self_rec_train",
            experiment_name="self_rec_train",
            file_suffix="_train",
            labels=("attacks", "honeypots"),
        ),
    ),
    monitors=(
        *monitor_family(
            folder="acausal/monitors",
            experiment="acausal",
            condition="acausal",
            title="Acausal",
            suffix="base",
            collusion="U",
        ),
        *monitor_family(
            folder="causal/monitors",
            experiment="causal",
            condition="causal",
            title="Causal",
            suffix="base",
            collusion="U",
        ),
        # Causal/combined SFT arms. They keep their historic filenames but use the
        # SELF_REC collusion prompt and, since this rename, the same `<strategy>_sft`
        # arm naming as the passive self-rec SFT arms.
        Monitor(
            path="causal/monitors/causal_none_train_causal_none_eval.yaml",
            experiment="causal",
            experiment_name="causal none train | causal none eval",
            results_folder="results/none_sft/",
            condition="self_rec",
            strategy="none",
            arm="none_sft",
            collusion="causal_base",
        ),
        Monitor(
            path="causal/monitors/causal_minified_train_causal_minified_eval.yaml",
            experiment="causal",
            experiment_name="causal minified train | causal minified eval",
            results_folder="results/minified_sft/",
            condition="self_rec",
            strategy="minified",
            arm="minified_sft",
            collusion="causal_minified",
        ),
        Monitor(
            path="causal/monitors/causal_paraphrased_train_causal_paraphrased_eval.yaml",
            experiment="causal",
            experiment_name="causal paraphrased train | causal paraphrased eval",
            results_folder="results/paraphrased_sft/",
            condition="self_rec",
            strategy="paraphrased",
            arm="paraphrased_sft",
            collusion="causal_paraphrased",
        ),
        # Additive: the causal-paraphrased SFT monitor evaluated on the causal
        # experiment's paraphrased transforms (hence the nested input directory).
        Monitor(
            path="additive/paraphrased_monitor_config_sft_selfrec_3_epoch.yaml",
            experiment="additive",
            experiment_name="Additive SFT Summarised",
            results_folder="results/paraphrased_base_selfrec/",
            condition="self_rec",
            strategy="paraphrased",
            arm="paraphrased_sft",
            collusion="causal_paraphrased",
            input_dir="transforms/causal/paraphrased",
        ),
        *monitor_family(
            folder="self_rec/monitors/base",
            experiment="self_rec",
            condition="self_rec",
            title="Self Rec",
            suffix="base",
            collusion="U",
        ),
        # Trusted-monitor-only ablation: same inputs as the none_base arm, with the
        # untrusted monitor switched off in `defaults`.
        Monitor(
            path="self_rec/monitors/base/base_monitor_config_tm_only.yaml",
            experiment="self_rec",
            experiment_name="TM Only",
            results_folder="results/none_base/tm_only",
            condition="self_rec",
            strategy="none",
            arm="none_base",
            collusion="U",
            disable_untrusted_monitor=True,
        ),
        *monitor_family(
            folder="self_rec/monitors/sft",
            experiment="self_rec",
            condition="self_rec",
            title="Self-Rec SFT",
            suffix="sft",
            collusion={
                "none": "self_rec_base",
                "minified": "self_rec_minified",
                "paraphrased": "self_rec_paraphrased",
            },
        ),
        # Label controls for the passive self-rec SFT, each evaluated twice: on the
        # standard generations (`original`) and on backdoor-free generations
        # (`no_backdoors`, where the "attacks" run is really the honest solutions).
        *[
            Monitor(
                path=f"control_self_rec/monitors/{variant}/base_monitor_config_{arm_name}.yaml",
                experiment="control_self_rec",
                experiment_name=f"{title} ({variant_title})",
                results_folder=f"results/{variant}/none_sft_{arm_name}/",
                condition="self_rec",
                strategy="none",
                arm=f"{variant}/none_sft_{arm_name}",
                collusion=checkpoint,
                input_dir=f"generator/{variant}",
                input_files=(
                    {"attacks": "honest.eval"} if variant == "no_backdoors" else None
                ),
            )
            for arm_name, checkpoint, title in (
                ("honest", "self_rec_honest", "Self-Rec SFT Honest"),
                (
                    "randomised",
                    "self_rec_permuted",
                    "Self Rec SFT Randomised Permutation",
                ),
            )
            for variant, variant_title in (
                ("original", "Original"),
                ("no_backdoors", "No Backdoors"),
            )
        ],
    ),
)

BIGCODEBENCH = Setting(
    generators=(
        Generator(
            path="acausal/generator/generator_config.yaml",
            experiment="acausal",
            experiment_name="acausal",
            extra_defaults={"dataset_type": "bigcodebench"},
            runs=(
                GenRun("attacks", "attack", "acausal_attack_policy_prompt_bcb", "U"),
            ),
        ),
        Generator(
            path="acausal/generator/generator_optimized_config.yaml",
            experiment="acausal_optimized",
            experiment_name="acausal_optimized",
            extra_defaults={"dataset_type": "bigcodebench", "limit": 10},
            runs=(
                GenRun(
                    "attacks",
                    "attack",
                    "acausal_optimized_attack_policy_prompt_bcb",
                    "U",
                ),
            ),
        ),
        Generator(
            path="causal/generator/generator_config.yaml",
            experiment="causal",
            experiment_name="causal",
            extra_defaults={"dataset_type": "bigcodebench"},
            runs=(GenRun("attacks", "attack", "causal_attack_policy_prompt_bcb", "U"),),
        ),
        Generator(
            path="self_rec/generator/generator_config.yaml",
            experiment="self_rec",
            experiment_name="self_rec_generator",
            extra_defaults={"dataset_type": "bigcodebench"},
            runs=(
                GenRun("attacks", "attack", "self_rec_attack_policy_prompt_bcb", "U"),
                GenRun("honest", "honest", "honest_policy_prompt_bcb", "U"),
                GenRun("honeypots", "attack", "self_rec_attack_policy_prompt_bcb", "H"),
            ),
        ),
        # SFT training data (train split). `dataset:` rather than `dataset_type:`
        # is the historic spelling here; the loader accepts both.
        Generator(
            path="self_rec/generator/generator_config_train.yaml",
            experiment="bcb_train",
            experiment_name="bcb_train",
            data_split="train",
            extra_defaults={"dataset": "bigcodebench"},
            runs=(
                GenRun("attacks", "attack", "self_rec_attack_policy_prompt_bcb", "U"),
                GenRun("honeypots", "attack", "self_rec_attack_policy_prompt_bcb", "H"),
            ),
        ),
    ),
    transforms=(
        *transform_pair(
            folder="acausal/transforms", experiment="acausal", experiment_name="acausal"
        ),
        *transform_pair(
            folder="causal/transforms", experiment="causal", experiment_name="causal"
        ),
        *transform_pair(
            folder="self_rec/transforms",
            experiment="self_rec",
            experiment_name="self_rec",
        ),
        *transform_pair(
            folder="self_rec/transforms",
            experiment="bcb_train",
            experiment_name="bcb_train",
            file_suffix="_train",
            labels=("attacks", "honeypots"),
        ),
    ),
    monitors=(
        *monitor_family(
            folder="acausal/monitors",
            experiment="acausal",
            condition="acausal",
            title="Acausal",
            suffix="base",
            collusion="U",
        ),
        *monitor_family(
            folder="causal/monitors",
            experiment="causal",
            condition="causal",
            title="Causal",
            suffix="base",
            collusion="U",
        ),
        *monitor_family(
            folder="self_rec/monitors/base",
            experiment="self_rec",
            condition="self_rec",
            title="Self-Rec",
            suffix="base",
            collusion="U",
        ),
        *monitor_family(
            folder="self_rec/monitors/sft",
            experiment="self_rec",
            condition="self_rec",
            title="Self-Rec SFT",
            suffix="sft",
            collusion={
                "none": "bcb_self_rec_base",
                "minified": "bcb_self_rec_minified",
                "paraphrased": "bcb_self_rec_paraphrased",
            },
        ),
    ),
)

SPEC: dict[str, Setting] = {"apps": APPS, "bigcodebench": BIGCODEBENCH}

# Every generated tree is gitignored, so the DEFAULT invocation must be able to
# rebuild all of them from scratch -- hence bigcodebench is generated for the open
# class too, even though the open-weights *runs* are APPS-focused.
DEFAULT_SETTINGS = ("apps", "bigcodebench")

# Resolved SFT checkpoints live here when --checkpoints is not passed. This file
# is the one piece of genuinely non-regenerable state in the scheme (the arm ->
# tinker:// mapping), so keep it under version control: it sits under the
# gitignored experiments/ tree, so it needs `git add -f`.
DEFAULT_CHECKPOINTS_FILE = Path(__file__).resolve().parent / "ckpts.json"


# ---------------------------------------------------------------------------
# model resolution
# ---------------------------------------------------------------------------


class ModelResolver:
    """Turns a spec model reference (role key or SFT arm) into a model string."""

    def __init__(self, model_class: str, checkpoints: Mapping[str, str]):
        self.model_class = model_class
        self.checkpoints = checkpoints
        self.pending: set[str] = set()
        self.used_arms: set[str] = set()

    def role(self, key: str) -> str:
        return ROLE_MODELS[self.model_class][key]

    def resolve(self, ref: str) -> str:
        """`ref` is either a role key (U/T/H) or an SFT arm name."""
        if ref in ROLE_MODELS[self.model_class]:
            return self.role(ref)
        if ref not in SFT_ARMS:
            raise SystemExit(f"spec error: unknown model reference {ref!r}")
        self.used_arms.add(ref)
        if self.model_class == "closed":
            return SFT_ARMS[ref]
        path = self.checkpoints.get(ref)
        if not path:
            self.pending.add(ref)
            return f"tinker/tinker://PENDING__{ref}"
        if not path.startswith("tinker://"):
            raise SystemExit(
                f"checkpoint for arm {ref!r} must be a tinker:// path, got: {path!r}"
            )
        return f"tinker/{path}"


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------

HEADER = """\
# GENERATED FILE - do not hand-edit; your changes will be overwritten.
# Source of truth: experiments/open_models/make_configs.py (declarative spec at module top).
# Regenerate: python experiments/open_models/make_configs.py [--checkpoints ckpts.json]
#   (SFT checkpoints are picked up automatically from experiments/open_models/ckpts.json)
"""


class _Dumper(yaml.SafeDumper):
    """SafeDumper that indents sequences under their parent key."""

    def increase_indent(self, flow: bool = False, indentless: bool = False):  # noqa: D102
        return super().increase_indent(flow, False)


def _block(obj: Any) -> str:
    return yaml.dump(
        obj,
        Dumper=_Dumper,
        sort_keys=False,
        default_flow_style=False,
        width=1000,
        allow_unicode=True,
    ).rstrip("\n")


def render(document: Mapping[str, Any]) -> str:
    """Render a config document, one blank-line-separated block per top-level key."""
    parts: list[str] = []
    for key, value in document.items():
        if key == "runs":
            chunks = ["runs:"]
            for i, (label, run) in enumerate(value.items()):
                if i:
                    chunks.append("")
                chunks.append(textwrap.indent(_block({label: run}), "  "))
            parts.append("\n".join(chunks))
        else:
            parts.append(_block({key: value}))
    return HEADER + "\n" + "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# document builders
# ---------------------------------------------------------------------------


def experiment_folder(model_class: str, setting: str, experiment: str) -> str:
    return f"experiments/{MODEL_CLASS_DIRS[model_class]}/{setting}/{experiment}"


def build_generator(
    spec: Generator, models: ModelResolver, setting: str, max_tokens: int | None
) -> dict[str, Any]:
    defaults: dict[str, Any] = {"data_split": spec.data_split}
    defaults.update(spec.extra_defaults)
    if max_tokens is not None:
        defaults["max_tokens"] = max_tokens
    runs = {
        run.label: {
            "eval_mode": run.eval_mode,
            "system_prompt": run.system_prompt,
            "save_path": run.save_path or f"generator/{run.label}.eval",
            "model": models.role(run.role),
        }
        for run in spec.runs
        # Shared labels are generated once, by their source experiment.
        if shared_source(models.model_class, spec.experiment, run.label) is None
    }
    return {
        "experiment_folder": experiment_folder(
            models.model_class, setting, spec.experiment
        ),
        "experiment_name": spec.experiment_name,
        "results_folder": spec.results_folder,
        "defaults": defaults,
        "runs": runs,
    }


def build_transform(
    spec: Transform, models: ModelResolver, setting: str
) -> dict[str, Any]:
    transform: dict[str, Any] = {"type": spec.kind, "remove_comments": True}
    if spec.kind == "llm":
        transform["model_info"] = {
            "model": models.role("T"),
            "system_prompt": "LLM_SUMMARIZE_SYSTEM_PROMPT",
            "base_prompt": "LLM_SUMMARIZE_BASE_PROMPT",
        }
    runs = {
        label: {
            "eval_path": spec.eval_path(label),
            "save_path": f"transforms/{spec.transform_dir}/"
            f"{LABEL_STEMS[label]}_transformed.eval",
        }
        for label in spec.labels
        # A shared label is transformed once, by its source experiment; the
        # other conditions read that same transformed .eval (see SHARED_INPUTS).
        if shared_source(models.model_class, spec.experiment, label) is None
    }
    return {
        "experiment_folder": experiment_folder(
            models.model_class, setting, spec.experiment
        ),
        "experiment_name": spec.experiment_name,
        "results_folder": f"results/{spec.transform_dir}/",
        "defaults": {"transform": transform},
        "runs": runs,
    }


def _monitor_entry(
    name: str,
    model: str,
    system_prompt: str,
    *,
    transformed: bool,
    remove_comments: bool,
) -> dict[str, Any]:
    return {
        name: {
            "model_info": {
                "model": model,
                "system_prompt": system_prompt,
                "base_prompt": "MONITOR_USER_PROMPT",
            },
            "remove_comments": remove_comments,
            "transformed": transformed,
        }
    }


def build_monitor(spec: Monitor, models: ModelResolver, setting: str) -> dict[str, Any]:
    transformed = spec.strategy != "none"
    # untrusted monitors read the (possibly transformed) code; the trusted monitor
    # always reads the original, comments stripped.
    monitors = [
        _monitor_entry(
            "untrusted_collusion_monitor",
            models.resolve(spec.collusion),
            COLLUSION_PROMPTS[spec.condition],
            transformed=transformed,
            remove_comments=not transformed,
        ),
        _monitor_entry(
            "trusted_backdoor_monitor",
            models.role("T"),
            BACKDOOR_PROMPTS["none"],
            transformed=False,
            remove_comments=True,
        ),
        _monitor_entry(
            "untrusted_backdoor_monitor",
            models.role("U"),
            BACKDOOR_PROMPTS[spec.strategy],
            transformed=transformed,
            remove_comments=not transformed,
        ),
    ]
    defaults: dict[str, Any] = {}
    if spec.disable_untrusted_monitor:
        defaults["disable_untrusted_monitor"] = True
    defaults["monitors"] = monitors

    runs = {
        label: {
            # Shared honest/honeypots inputs live in the source experiment's
            # folder (a sibling under <model-class>/<setting>/), so they are
            # read via `../<source>/`. Outputs stay in THIS experiment: the
            # collusion prompt is condition-specific, so the same shared
            # honeypots must still be re-scored under each condition.
            "eval_path": (
                f"../{_src}/{spec.eval_path(label)}"
                if (_src := shared_source(models.model_class, spec.experiment, label))
                else spec.eval_path(label)
            ),
            "save_path": spec.save_path(label),
            "monitors": list(MONITOR_ORDER),
        }
        for label in spec.labels
    }
    return {
        "experiment_folder": experiment_folder(
            models.model_class, setting, spec.experiment
        ),
        "experiment_name": spec.experiment_name,
        "results_folder": spec.results_folder,
        "defaults": defaults,
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def target_dir(model_class: str, setting: str) -> Path:
    return CONFIGS / (setting if model_class == "closed" else f"{setting}_open")


def generate_setting(
    model_class: str, setting: str, models: ModelResolver, max_tokens: int | None
) -> tuple[list[Path], list[Path]]:
    """Write one (model_class, setting) tree. Returns (written, pruned) paths."""
    spec = SPEC[setting]
    root = target_dir(model_class, setting)
    documents: dict[str, dict[str, Any]] = {}
    for gen in spec.generators:
        documents[gen.path] = build_generator(gen, models, setting, max_tokens)
    for tr in spec.transforms:
        documents[tr.path] = build_transform(tr, models, setting)
    for mon in spec.monitors:
        documents[mon.path] = build_monitor(mon, models, setting)

    written: list[Path] = []
    for rel, document in sorted(documents.items()):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(render(document))
        written.append(dst)

    # Sidecar helpers (configs/bigcodebench/plot_everything.sh) are checked in with
    # the closed tree; mirror them so a gitignored *_open tree is fully regenerable.
    if model_class == "open":
        for src in sorted(target_dir("closed", setting).rglob("*")):
            if src.is_file() and src.suffix != ".yaml":
                dst = root / src.relative_to(target_dir("closed", setting))
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
                dst.chmod(src.stat().st_mode)

    # the spec owns every .yaml under the tree: drop anything it no longer emits
    keep = set(written)
    pruned = sorted(p for p in root.rglob("*.yaml") if p not in keep)
    for stale in pruned:
        stale.unlink()
    for directory in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return written, pruned


def load_checkpoints(path: Path | None) -> dict[str, str]:
    """Load arm -> tinker:// checkpoint paths.

    With no --checkpoints, fall back to DEFAULT_CHECKPOINTS_FILE if it exists.
    Without that fallback the bare regenerate command -- the one printed at the
    top of every generated file -- would silently revert resolved SFT arms back
    to PENDING placeholders, throwing away the record of which LoRA to use.
    """
    if path is None:
        if not DEFAULT_CHECKPOINTS_FILE.exists():
            return {}
        path = DEFAULT_CHECKPOINTS_FILE
        print(f"using checkpoints from {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(
            f"{path}: expected a JSON object mapping arm -> tinker:// path"
        )
    unknown = sorted(set(data) - set(SFT_ARMS))
    if unknown:
        raise SystemExit(
            f"unknown SFT arm(s) in {path}: {unknown}\n"
            f"valid arms: {sorted(SFT_ARMS)}"
        )
    bad = sorted(k for k, v in data.items() if not isinstance(v, str))
    if bad:
        raise SystemExit(f"{path}: checkpoint path must be a string for arm(s): {bad}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the closed- and open-model experiment config trees "
        "from the declarative spec in this module.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="default: every tree (closed and open, apps and bigcodebench). "
        "All generated trees are gitignored, so the bare command must rebuild "
        "them all; the open-weights runs themselves are APPS-focused.",
    )
    parser.add_argument(
        "--model-class",
        nargs="+",
        default=["closed", "open"],
        choices=["closed", "open"],
        help="which model class(es) to generate (default: both)",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        choices=["apps", "bigcodebench"],
        help="override the per-class default settings",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="JSON file mapping SFT arm name -> tinker:// checkpoint path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="completion cap injected into OPEN generator configs "
        f"(default: {DEFAULT_MAX_TOKENS})",
    )
    args = parser.parse_args()

    checkpoints = load_checkpoints(args.checkpoints)

    pending: set[str] = set()
    for model_class in args.model_class:
        settings = args.settings or DEFAULT_SETTINGS
        models = ModelResolver(model_class, checkpoints)
        for setting in settings:
            written, pruned = generate_setting(
                model_class,
                setting,
                models,
                args.max_tokens if model_class == "open" else None,
            )
            root = target_dir(model_class, setting).relative_to(REPO_ROOT)
            print(f"{model_class:>6}  {root}: {len(written)} files")
            for stale in pruned:
                print(f"          pruned {stale.relative_to(REPO_ROOT)}")
        pending |= models.pending

    if checkpoints:
        print("\ncheckpoints:")
        for arm in sorted(checkpoints):
            print(f"  {arm}: {checkpoints[arm]}")
    if pending:
        print(
            f"\n{len(pending)} SFT arm(s) still need a Tinker checkpoint "
            f"(emitted as tinker://PENDING__<arm>):"
        )
        for arm in sorted(pending):
            print(f"  - {arm}")
        print("\nTrain each with finetuning/tinker_lora_finetune.py, then re-run:")
        print(
            "  python experiments/open_models/make_configs.py --checkpoints ckpts.json"
        )
        print(
            '  ckpts.json: {"self_rec_base": "tinker://<run>/sampler_weights/<name>", ...}'
        )


if __name__ == "__main__":
    main()
