"""Configuration loader for LASR Labs 2025 Control Project.

Loads and normalizes YAML configs with validation and path resolution.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from dacite import Config, from_dict

from lasr_labs_2025_control_project.scripts.helpers.metadata_dataclasses import (
    ModelInfo,
    TransformType,
)


@dataclass
class MonitorConfig:
    """Configuration for a single monitor.

    Attributes:
        model_info: The monitor model info.
        transformed: Whether the monitor should get transformed solution.
    """

    model_info: ModelInfo
    transformed: bool


@dataclass(frozen=True, slots=True)
class RunMonitorsConfig:
    """Configuration for a single run monitors.

    Attributes:
        trusted_backdoor_monitor: The trusted backdoor monitor config.
        trusted_collusion_monitor: The trusted collusion monitor config.
        untrusted_backdoor_monitor: The untrusted backdoor monitor config.
        untrusted_collusion_monitor: The untrusted collusion monitor config.
    """

    trusted_backdoor_monitor: Optional[MonitorConfig]
    trusted_collusion_monitor: Optional[MonitorConfig]
    untrusted_backdoor_monitor: Optional[MonitorConfig]
    untrusted_collusion_monitor: Optional[MonitorConfig]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration for a single monitoring run.

    Attributes:
        label: The label of the run.
        eval_path: The path to the eval file.
        save_path: The path to the save file.
        limit: The limit of the run.
        monitors: The monitors to run.
    """

    label: Literal["attacks", "honeypots", "honest"]
    eval_path: Path
    save_path: Path
    limit: Optional[int]
    monitors: RunMonitorsConfig


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Normalized configuration.

    Attributes:
        experiment_folder: Path to the base directory for all the eval files.
        results_folder: Path to the results folder (can be relative to the experiment folder).
        experiment_name: The name of the experiment.
        defaults: The defaults section of the config.
        runs: The runs section of the config.
    """

    experiment_folder: Path
    results_folder: Path
    experiment_name: str
    defaults: dict[str, Any]
    runs: list[RunConfig]


# Prompt registry - populated at module level
_PROMPT_REGISTRY: dict[str, str] = {}


def _init_prompt_registry() -> None:
    """Initialize the prompt registry on first access."""
    if _PROMPT_REGISTRY:
        return

    from lasr_labs_2025_control_project.prompts.logprob_prompts import (
        ACASUAL_LOGPROBS_SYSTEM_PROMPT,
        ACASUAL_LOGPROBS_USER_PROMPT,
        CAUSAL_LOGPROBS_SYSTEM_PROMPT,
        CAUSAL_LOGPROBS_USER_PROMPT,
        LOGPROBS_SYSTEM_PROMPT,
        LOGPROBS_USER_PROMPT,
        SELF_REC_FT_LOGPROBS_SYSTEM_PROMPT,
        SELF_REC_FT_LOGPROBS_USER_PROMPT,
    )
    from lasr_labs_2025_control_project.prompts.summarizer_prompts import (
        LLM_SUMMARIZE_BASE_PROMPT,
        LLM_SUMMARIZE_SYSTEM_PROMPT,
    )

    _PROMPT_REGISTRY.update(
        {
            "LOGPROBS_SYSTEM_PROMPT": LOGPROBS_SYSTEM_PROMPT,
            "LOGPROBS_USER_PROMPT": LOGPROBS_USER_PROMPT,
            "SELF_REC_FT_LOGPROBS_SYSTEM_PROMPT": SELF_REC_FT_LOGPROBS_SYSTEM_PROMPT,
            "SELF_REC_FT_LOGPROBS_USER_PROMPT": SELF_REC_FT_LOGPROBS_USER_PROMPT,
            "CAUSAL_LOGPROBS_SYSTEM_PROMPT": CAUSAL_LOGPROBS_SYSTEM_PROMPT,
            "CAUSAL_LOGPROBS_USER_PROMPT": CAUSAL_LOGPROBS_USER_PROMPT,
            "ACASUAL_LOGPROBS_SYSTEM_PROMPT": ACASUAL_LOGPROBS_SYSTEM_PROMPT,
            "ACASUAL_LOGPROBS_USER_PROMPT": ACASUAL_LOGPROBS_USER_PROMPT,
            "LLM_SUMMARIZE_SYSTEM_PROMPT": LLM_SUMMARIZE_SYSTEM_PROMPT,
            "LLM_SUMMARIZE_BASE_PROMPT": LLM_SUMMARIZE_BASE_PROMPT,
        }
    )


def get_prompt_by_name(prompt_name: str) -> str:
    """Retrieve a prompt by name from the registry."""
    _init_prompt_registry()
    try:
        return _PROMPT_REGISTRY[prompt_name]
    except KeyError:
        available = ", ".join(_PROMPT_REGISTRY.keys())
        raise ValueError(
            f"Unknown prompt '{prompt_name}'. Available: {available}"
        ) from None


def load_config_dict(config_path: Path) -> dict[str, Any]:
    """Load YAML config file and return base directory and parsed data.

    Args:
        config_path: The path to the config file.

    Returns:
        A tuple of the parsed config and the experiment folder path.
    """
    cfg_path = config_path.expanduser().resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping at the top level")

    if "experiment_folder" not in data:
        raise ValueError(
            "Config must include a non-empty 'experiment_folder' top level key, which is a base directory for all the eval files"
        )

    if "results_folder" not in data:
        raise ValueError(
            "Config must include a non-empty 'results_folder' top level key. This folder will be used to store the results of the experiment"
        )
    if "experiment_name" not in data:
        raise ValueError(
            "Config must include a non-empty 'experiment_name' top level key, which is the unique name of the experiment"
        )

    return data


def normalize_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize the defaults section of the config.

    Args:
        config: The config to normalize.

    Returns:
        The normalized config with the following keys:
        - limit: The limit of the run.
        - monitors_map: A map of monitor names to their configs.
    """
    CANONICAL_MONITOR_NAMES = {
        "trusted_backdoor_monitor",
        "trusted_collusion_monitor",
        "untrusted_backdoor_monitor",
        "untrusted_collusion_monitor",
    }
    defaults = config.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be provided and must be a mapping")

    if "monitors" not in defaults:
        raise ValueError("defaults must include a 'monitors' list")

    monitors_list = defaults["monitors"]
    if not isinstance(monitors_list, list) or not monitors_list:
        raise ValueError("defaults.monitors must be a non-empty list")

    monitors_map: dict[str, dict[str, Any]] = {}

    for entry in monitors_list:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(
                "Each monitor entry must be a dict with exactly one key (the canonical monitor name)"
            )
        name, cfg = next(iter(entry.items()))
        if name not in CANONICAL_MONITOR_NAMES:
            raise ValueError(
                f"Monitor name '{name}' is not one of the allowed canonical keys: {', '.join(CANONICAL_MONITOR_NAMES)}"
            )
        if name in monitors_map:
            raise ValueError(f"Duplicate monitor name '{name}' in defaults.monitors")

        if "transformed" not in cfg:
            raise ValueError(f"Monitor '{name}' must define 'transformed'")

        monitors_map[name] = cfg

    return {
        "limit": defaults.get("limit"),
        "monitors_map": monitors_map,
    }


def _resolve_prompt_value(prompt: str) -> str:
    """Resolve a prompt name to its registered prompt text when possible.

    Args:
        prompt: The prompt to resolve. It's either a registered prompt name or a literal string (in that case it's left as is).

    Returns:
        The resolved prompt string.
    """
    if isinstance(prompt, str):
        try:
            return get_prompt_by_name(prompt)
        except ValueError:
            return prompt


def _resolve_prompts_in_raw_monitor(raw_monitor: dict[str, Any]) -> None:
    """Mutate raw_monitor to resolve model_info.system_prompt and model_info.base_prompt if they are names.

    Args:
        raw_monitor: The raw monitor config.
    """
    model_info = raw_monitor.get("model_info")
    if not isinstance(model_info, dict):
        return
    if "system_prompt" in model_info:
        model_info["system_prompt"] = _resolve_prompt_value(model_info["system_prompt"])
    if "base_prompt" in model_info:
        model_info["base_prompt"] = _resolve_prompt_value(model_info["base_prompt"])


def _resolve_eval_path(
    experiment_folder: Path, eval_path_str: str, check_exists: bool = False
) -> Path:
    eval_path = Path(eval_path_str)
    if not eval_path.is_absolute():
        eval_path = experiment_folder / eval_path
    if check_exists and not eval_path.exists():
        raise FileNotFoundError(f"eval_path {eval_path} does not exist")
    return eval_path


def _build_run_config(
    label: Literal["attacks", "honeypots", "honest"],
    run_cfg: dict[str, Any],
    experiment_folder: Path,
    defaults: dict[str, Any],
) -> RunConfig:
    eval_path_str = run_cfg.get("eval_path")
    save_path_str = run_cfg.get("save_path")
    if eval_path_str is None or save_path_str is None:
        raise ValueError("eval_path and save_path must be provided for each run")

    eval_path = _resolve_eval_path(experiment_folder, eval_path_str, check_exists=True)
    save_path = _resolve_eval_path(experiment_folder, save_path_str)

    limit = run_cfg.get("limit", defaults.get("limit"))

    if "monitors" not in run_cfg:
        raise ValueError(f"Run '{label}' must include an explicit 'monitors' list")

    selected_names = run_cfg["monitors"]
    if not isinstance(selected_names, list) or not selected_names:
        raise ValueError(
            f"Run '{label}': 'monitors' must be a non-empty list of strings"
        )

    available = defaults["monitors_map"]
    resolved_raw_monitors: dict[str, Any] = {}

    for name in selected_names:
        if not isinstance(name, str):
            raise ValueError(f"Run '{label}': monitor selectors must be strings")
        if name not in available:
            raise ValueError(f"Run '{label}' references unknown monitor '{name}'")
        # copy the config
        m = dict(available[name])
        _resolve_prompts_in_raw_monitor(m)
        resolved_raw_monitors[name] = m

    return RunConfig(
        label=label,
        eval_path=eval_path,
        save_path=save_path,
        limit=limit,
        monitors=from_dict(
            RunMonitorsConfig, resolved_raw_monitors, Config(cast=[TransformType])
        ),
    )


def normalize_runs(config: dict[str, Any], defaults: dict[str, Any]) -> list[RunConfig]:
    """Parse and normalize all run configurations."""
    experiment_folder = config["experiment_folder"]
    runs_raw = config.get("runs", {})
    if not isinstance(runs_raw, (list, dict)) or not runs_raw:
        raise ValueError("Config must include a non-empty 'runs' list or mapping")

    normalized: list[RunConfig] = []

    if isinstance(runs_raw, dict):
        for label, cfg in runs_raw.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Run '{label}' must be a mapping")
            normalized.append(
                _build_run_config(label, cfg, experiment_folder, defaults)
            )
    else:
        # Legacy list format
        for idx, run in enumerate(runs_raw, start=1):
            if not isinstance(run, dict):
                raise ValueError(f"Run #{idx} must be a mapping")
            label = run.get("label", f"run_{idx}")
            normalized.append(
                _build_run_config(label, run, experiment_folder, defaults)
            )

    return normalized


def load_and_normalize_config(config_path: str | Path) -> ExperimentConfig:
    """Load and fully normalize a configuration file.

    Returns:
        ExperimentConfig: The normalized experiment configuration.
    """
    cfg = load_config_dict(Path(config_path))

    defaults = normalize_defaults(cfg)
    runs = normalize_runs(cfg, defaults)

    return ExperimentConfig(
        experiment_folder=cfg["experiment_folder"],
        results_folder=cfg["results_folder"],
        experiment_name=cfg["experiment_name"],
        defaults=defaults,
        runs=runs,
    )
