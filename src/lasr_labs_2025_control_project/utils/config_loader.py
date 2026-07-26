"""Configuration loader for LASR Labs 2025 Control Project.

Loads and normalizes YAML configs with validation and path resolution.
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from control_arena.eval_mode import EvalMode
from dacite import Config, from_dict

from lasr_labs_2025_control_project.scripts.helpers.metadata_dataclasses import (
    ModelInfo,
    MonitorConfig,
    TransformInfo,
    TransformType,
)


@dataclass(frozen=True, slots=True)
class RunMonitors:
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


class DataSplit(StrEnum):
    """The data split to run the generator on."""

    TRAIN = "train"
    TEST = "test"


class DatasetType(StrEnum):
    """The dataset type to use for generation."""

    APPS = "apps"
    BIGCODEBENCH = "bigcodebench"


@dataclass(frozen=True, slots=True)
class RunGenerator:
    """Configuration for a single generator run.

    Attributes:
        model_info: The model info.
        eval_mode: The evaluation mode.
        data_split: The data split.
        dataset_type: The dataset type (apps or bigcodebench).
        max_tokens: Optional per-run completion token cap passed to inspect's
            eval(); None keeps the provider default (open models that reason at
            length before emitting <final_code> need more than the tinker
            provider's 4096 default).
    """

    model_info: ModelInfo
    eval_mode: EvalMode
    data_split: DataSplit
    dataset_type: DatasetType = DatasetType.APPS
    max_tokens: Optional[int] = None


class RunLabel(StrEnum):
    """The label of the run."""

    ATTACKS = "attacks"
    HONEYPOTS = "honeypots"
    HONEST = "honest"


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration for a single monitoring run.

    Attributes:
        label: The label of the run.
        eval_path: The path to the eval file.
        save_path: The path to the save file.
        limit: The limit of the run.
    """

    label: RunLabel
    save_path: Path
    limit: Optional[int]


@dataclass(frozen=True, slots=True)
class GeneratorRunConfig(RunConfig):
    """Configuration for a single generator run."""

    generator: RunGenerator


@dataclass(frozen=True, slots=True)
class TransformRunConfig(RunConfig):
    """Configuration for a single transform run."""

    eval_path: Path
    transform: TransformInfo


@dataclass(frozen=True, slots=True)
class MonitorRunConfig(RunConfig):
    """Configuration for a single monitoring run."""

    eval_path: Path
    monitors: RunMonitors


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
    """Automatically initialize the prompt registry with all string prompts defined in the prompts package."""
    if _PROMPT_REGISTRY:
        return

    import importlib
    import pkgutil

    # Find all modules in the prompts package
    prompts_pkg = "lasr_labs_2025_control_project.prompts"
    try:
        pkg = importlib.import_module(prompts_pkg)
    except ImportError:
        raise RuntimeError(f"Could not import prompts package: {prompts_pkg}")

    # Iterate through all modules in the prompts package
    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__, prompts_pkg + "."):
        if ispkg:
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue  # skip modules that fail to import

        # Add all UPPERCASE string variables to the registry
        for attr in dir(mod):
            value = getattr(mod, attr)
            if isinstance(value, str):
                _PROMPT_REGISTRY[attr] = value


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


def normalize_monitor_defaults(config: dict[str, Any]) -> dict[str, Any]:
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

        if "remove_comments" not in cfg:
            raise ValueError(f"Monitor '{name}' must define 'remove_comments'")

        monitors_map[name] = cfg

    return {
        "limit": defaults.get("limit"),
        "monitors_map": monitors_map,
        "disable_untrusted_monitor": defaults.get("disable_untrusted_monitor", False),
        "disable_trusted_monitor": defaults.get("disable_trusted_monitor", False),
    }


def normalize_transform_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize the defaults section of the config.

    Args:
        config: The config to normalize.

    Returns:
        The normalized config with the following keys:
        - limit: The limit of the run.
        - transform: The transform config.
    """
    defaults = config.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be provided and must be a mapping")

    if "transform" not in defaults:
        raise ValueError("defaults must include a 'transform' mapping")

    transform = defaults["transform"]
    if not isinstance(transform, dict):
        raise ValueError("defaults.transform must be a mapping")

    if "remove_comments" not in transform:
        raise ValueError(
            "defaults.transform must include a 'remove_comments' boolean field"
        )

    _resolve_prompts(transform)

    return {
        "limit": defaults.get("limit"),
        "transform": from_dict(
            TransformInfo, transform, Config(cast=[TransformType, bool])
        ),
    }


def _resolve_prompts(config: dict[str, Any]) -> None:
    """Mutate config to resolve model_info.system_prompt and model_info.base_prompt if they are names.

    Args:x
        config: The config with model_info key.
    """
    model_info = config.get("model_info")
    if not isinstance(model_info, dict):
        return
    if "system_prompt" in model_info:
        model_info["system_prompt"] = get_prompt_by_name(model_info["system_prompt"])
    if "base_prompt" in model_info:
        model_info["base_prompt"] = get_prompt_by_name(model_info["base_prompt"])


def _resolve_eval_path(
    experiment_folder: Path, eval_path_str: str, check_exists: bool = False
) -> Path:
    eval_path = Path(eval_path_str)
    if not eval_path.is_absolute():
        eval_path = experiment_folder / eval_path
    if check_exists and not eval_path.exists():
        raise FileNotFoundError(f"eval_path {eval_path} does not exist")
    return eval_path


def _build_base_run_config(
    label: RunLabel,
    run_cfg: dict[str, Any],
    experiment_folder: Path,
    defaults: dict[str, Any],
) -> RunConfig:
    save_path_str = run_cfg.get("save_path")
    if save_path_str is None:
        raise ValueError("save_path must be provided for each run")

    save_path = _resolve_eval_path(experiment_folder, save_path_str)

    limit = run_cfg.get("limit", defaults.get("limit"))
    return RunConfig(
        label=label,
        save_path=save_path,
        limit=limit,
    )


def _build_monitor_run_config(
    label: RunLabel,
    run_cfg: dict[str, Any],
    experiment_folder: Path,
    defaults: dict[str, Any],
) -> MonitorRunConfig:
    base_config = _build_base_run_config(label, run_cfg, experiment_folder, defaults)

    eval_path_str = run_cfg.get("eval_path")
    if eval_path_str is None:
        raise ValueError("eval_path must be provided for each monitor run")
    eval_path = _resolve_eval_path(experiment_folder, eval_path_str, check_exists=True)

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
        m = deepcopy(available[name])
        _resolve_prompts(m)
        resolved_raw_monitors[name] = m

    return MonitorRunConfig(
        label=label,
        eval_path=eval_path,
        save_path=base_config.save_path,
        limit=base_config.limit,
        monitors=from_dict(
            RunMonitors, resolved_raw_monitors, Config(cast=[TransformType, bool])
        ),
    )


def get_cfg_value(key: str, run_cfg: dict[str, Any], defaults: dict[str, Any]) -> Any:
    """Get a value from the run config or the defaults."""
    value = run_cfg.get(key, defaults.get(key))
    if value is None:
        raise ValueError(
            f"{key} must be provided either in the run config or in the defaults"
        )
    return value


def _build_generator_run_config(
    label: RunLabel,
    run_cfg: dict[str, Any],
    experiment_folder: Path,
    defaults: dict[str, Any],
) -> GeneratorRunConfig:
    """
    Build a GeneratorRunConfig from the provided run configuration.

    Args:
        label: The label for this run.
        run_cfg: The run configuration dictionary.
        experiment_folder: The base experiment folder.
        defaults: The defaults dictionary.

    Returns:
        GeneratorRunConfig: The constructed generator run configuration.
    """
    # Base config (label, save_path, limit)
    base_config = _build_base_run_config(label, run_cfg, experiment_folder, defaults)

    # Parse dataset_type (defaults to APPS for backward compatibility)
    dataset_type_raw = (
        run_cfg.get("dataset_type")
        or run_cfg.get("dataset")
        or defaults.get("dataset")
        or defaults.get("dataset_type")
        or "apps"
    )

    dataset_type = DatasetType(dataset_type_raw)

    # Parse eval_mode
    eval_mode_raw = get_cfg_value("eval_mode", run_cfg, defaults)
    eval_mode = EvalMode(eval_mode_raw)

    # Parse data_split
    data_split_raw = get_cfg_value("data_split", run_cfg, defaults)
    data_split = DataSplit(data_split_raw)

    # Resolve prompts in-place (if any)
    _resolve_prompts(run_cfg)

    # Parse model_info
    model = get_cfg_value("model", run_cfg, defaults)
    system_prompt_name = get_cfg_value("system_prompt", run_cfg, defaults)

    # Optional completion-token cap (per-run override, then defaults, else None)
    max_tokens = run_cfg.get("max_tokens", defaults.get("max_tokens"))

    # Build the RunGenerator dataclass
    run_generator = from_dict(
        RunGenerator,
        {
            "model_info": ModelInfo(
                model=model,
                system_prompt=get_prompt_by_name(system_prompt_name),
                base_prompt="{problem_statement}",
            ),
            "eval_mode": eval_mode,
            "data_split": data_split,
            "dataset_type": dataset_type,
            "max_tokens": max_tokens,
        },
    )

    # Return the GeneratorRunConfig dataclass
    return GeneratorRunConfig(
        label=label,
        save_path=base_config.save_path,
        limit=base_config.limit,
        generator=run_generator,
    )


def _build_transform_run_config(
    label: RunLabel,
    run_cfg: dict[str, Any],
    experiment_folder: Path,
    defaults: dict[str, Any],
) -> TransformRunConfig:
    base_config = _build_base_run_config(label, run_cfg, experiment_folder, defaults)

    eval_path_str = run_cfg.get("eval_path")
    if eval_path_str is None:
        raise ValueError("eval_path must be provided for each transform run")
    eval_path = _resolve_eval_path(experiment_folder, eval_path_str, check_exists=True)

    return TransformRunConfig(
        label=label,
        eval_path=eval_path,
        save_path=base_config.save_path,
        limit=base_config.limit,
        transform=defaults["transform"],
    )


def _build_run_config(
    label: RunLabel,
    run_cfg: dict[str, Any],
    experiment_folder: Path,
    defaults: dict[str, Any],
    config_type: Literal["monitor", "generator", "transform"] = "monitor",
) -> RunConfig:
    if config_type == "monitor":
        return _build_monitor_run_config(label, run_cfg, experiment_folder, defaults)
    elif config_type == "generator":
        return _build_generator_run_config(label, run_cfg, experiment_folder, defaults)
    elif config_type == "transform":
        return _build_transform_run_config(label, run_cfg, experiment_folder, defaults)
    else:
        raise ValueError(f"Invalid config type: {config_type}")


def normalize_runs(
    config: dict[str, Any],
    defaults: dict[str, Any],
    config_type: Literal["monitor", "generator", "transform"] = "monitor",
) -> list[RunConfig]:
    """Parse and normalize all run configurations."""
    experiment_folder = config["experiment_folder"]
    runs_raw = config.get("runs", {})
    if not isinstance(runs_raw, dict) or not runs_raw:
        raise ValueError("Config must include a non-empty 'runs' mapping")

    normalized: list[RunConfig] = []

    for label, cfg in runs_raw.items():
        label = RunLabel(label)
        if not isinstance(cfg, dict):
            raise ValueError(f"Run '{label}' must be a mapping")
        normalized.append(
            _build_run_config(label, cfg, experiment_folder, defaults, config_type)
        )

    return normalized


def load_and_normalize_config(
    config_path: str | Path,
    config_type: Literal["monitor", "generator", "transform"] = "monitor",
) -> ExperimentConfig:
    """Load and fully normalize a configuration file.

    Returns:
        ExperimentConfig: The normalized experiment configuration.
    """
    cfg = load_config_dict(Path(config_path))

    if config_type == "monitor":
        defaults = normalize_monitor_defaults(cfg)
    elif config_type == "transform":
        defaults = normalize_transform_defaults(cfg)
    else:
        defaults = cfg.get("defaults", {})
    runs = normalize_runs(cfg, defaults, config_type)

    experiment_folder = Path(cfg["experiment_folder"])
    results_folder = experiment_folder / cfg["results_folder"]

    return ExperimentConfig(
        experiment_folder=experiment_folder,
        results_folder=results_folder,
        experiment_name=cfg["experiment_name"],
        defaults=defaults,
        runs=runs,
    )
