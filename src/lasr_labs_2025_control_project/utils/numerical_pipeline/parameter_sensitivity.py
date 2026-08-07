"""Parameter sensitivity tools for optimization sweeps.

This module provides helpers to run the standard parameter optimization with a
fixed replacement FNR of zero and then evaluate how sensitive the blue team's
success rate is to variations in individual parameters. The output mirrors the
`OptimizationResult` structure so downstream consumers (plotting, reporting,
serialization) can process the sensitivity sweeps alongside other experiments.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, List, Mapping, MutableMapping

import numpy as np

from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (
    OptimizationConfig,
    OptimizationResult,
    build_blue_team_win_calculator,
    construct_parameter_spaces,
    optimize_parameters,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.optimizers import (
    double_oracle,  # noqa: F401
    grid_search,  # noqa: F401
    stackelberg_equilibrium,  # noqa: F401
)

OPTIMIZER = double_oracle  # Default optimizer for re-optimization


def run_parameter_sensitivity_analysis(
    base_config: OptimizationConfig,
    variation_points: int,
    max_iterations: int,
) -> Dict[str, OptimizationResult]:
    """Run a baseline optimization and one-at-a-time parameter sensitivity sweeps.

    This function performs a comprehensive sensitivity analysis by:
    1. Running baseline optimization with replace_fnr=0 to find optimal parameters
    2. For each parameter (red/blue team), sweeping its value while:
       a. Keeping other parameters fixed at baseline (fixed mode)
       b. Re-optimizing other parameters (reoptimized mode)
    3. Analyzing "green team" (simulation environment) parameter sensitivity

    Args:
        base_config: Seed `OptimizationConfig`; a fresh copy with
            `blue_team_replace_fnr = 0` will be constructed.
        variation_points: Number of sample points per parameter sweep (>= 2).
        max_iterations: Outer iteration budget for the baseline optimization.

    Returns:
        Dict mapping labels to `OptimizationResult` objects. Keys include:
            - "baseline": single-point optimization at replace FNR 0.
            - "<param>_sensitivity_fixed": sweep with other params held constant
            - "<param>_sensitivity_reoptimized": sweep with re-optimization
    """

    if variation_points < 2:
        raise ValueError("variation_points must be >= 2 for sensitivity analysis")

    # Clone config with replace_fnr=0 for baseline optimization
    config_zero = _clone_config_with_replace_fnr(base_config, replace_fnr=0.0)

    # Ensure parameter spaces are defined
    parameter_spaces = config_zero.parameter_spaces
    if not parameter_spaces:
        parameter_spaces = construct_parameter_spaces(
            config_zero.untrusted_threshold_min,
            config_zero.untrusted_threshold_max,
        )
        config_zero.parameter_spaces = parameter_spaces

    # Step 1: Run baseline optimization to find optimal parameters
    baseline_result = optimize_parameters(
        base_config=config_zero,
        replace_fnr_values=[0.0],
        max_iterations=max_iterations,
    )

    if not baseline_result.results:
        raise RuntimeError(
            "Baseline optimization returned no results; cannot run sensitivity analysis"
        )

    # Extract best parameters from baseline optimization
    baseline_params = _extract_best_parameters(baseline_result)

    sensitivity_results: Dict[str, OptimizationResult] = {"baseline": baseline_result}

    # Step 2: Perform one-at-a-time parameter sensitivity sweeps
    for param_name, base_value in baseline_params.items():
        if param_name not in parameter_spaces:
            # Skip any parameters that are not part of the standard optimization grid.
            continue

        space = parameter_spaces[param_name]
        # Build a grid of values to sweep across this parameter
        sweep_values = _build_variation_grid(space, base_value, variation_points)
        fixed_entries: List[Dict[str, Any]] = []
        reopt_entries: List[Dict[str, Any]] = []

        for candidate in sweep_values:
            # Create a copy of baseline params with the current parameter varied
            params_copy = dict(baseline_params)
            params_copy[param_name] = candidate

            # Mode 1: Fixed - evaluate with all other params held constant
            safety, outcome_distribution = _evaluate_parameters(
                config_zero, params_copy
            )

            fixed_entries.append(
                {
                    "safety": safety,
                    "usefulness": 1.0,  # replace FNR is fixed at zero
                    "blue_team_replace_fnr": 0.0,
                    "best_parameters": params_copy,
                    "outcome_distribution": outcome_distribution,
                    "varied_parameter": param_name,
                    "varied_value": candidate,
                    "analysis_mode": "fixed",
                    "converged": True,
                }
            )

            # Mode 2: Reoptimized - re-optimize all other params while fixing this one
            reopt_result = _reoptimize_with_parameter(
                config=config_zero,
                param_name=param_name,
                fixed_value=candidate,
                parameter_spaces=parameter_spaces,
                max_iterations=max_iterations,
            )

            # Extract blue mixed strategy and red parameters from optimization result
            blue_mixed_strategy = reopt_result.get("blue_mixed_strategy", [])
            # Prefer new field name, fallback to deprecated for backwards compatibility
            red_params = reopt_result.get("red_best_response") or reopt_result.get(
                "best_parameters", {}
            )

            # Combine blue and red parameters for complete parameter set
            if blue_mixed_strategy:
                first_blue_strategy = blue_mixed_strategy[0][0]
                reopt_best_params = {**first_blue_strategy, **red_params}
            else:
                reopt_best_params = dict(red_params)

            reopt_best_params = {k: float(v) for k, v in reopt_best_params.items()}

            reopt_outcomes_raw = reopt_result.get("outcome_distribution", {})
            # Convert all outcome values to floats for consistency
            reopt_outcomes: Dict[str, float] = {}
            if isinstance(reopt_outcomes_raw, Mapping):
                for k, v in reopt_outcomes_raw.items():
                    try:
                        reopt_outcomes[k] = float(v)
                    except (TypeError, ValueError):
                        continue

            reopt_entries.append(
                {
                    "safety": float(reopt_result["best_score"]),
                    "usefulness": 1.0,
                    "blue_team_replace_fnr": 0.0,
                    "best_parameters": reopt_best_params,
                    "blue_mixed_strategy": blue_mixed_strategy,
                    "red_mixed_strategy": reopt_result.get("red_mixed_strategy", []),
                    "outcome_distribution": reopt_outcomes,
                    "history": reopt_result.get("history", []),
                    "varied_parameter": param_name,
                    "varied_value": candidate,
                    "analysis_mode": "reoptimized",
                    "converged": bool(reopt_result.get("converged", True)),
                }
            )

        # Keep the order deterministic (ascending value) for downstream consumers.
        fixed_entries.sort(key=lambda entry: entry["varied_value"])
        reopt_entries.sort(key=lambda entry: entry["varied_value"])

        # Store both fixed and reoptimized results for this parameter
        sensitivity_results[f"{param_name}_sensitivity_fixed"] = OptimizationResult(
            config=config_zero,
            results=fixed_entries,
        )
        sensitivity_results[f"{param_name}_sensitivity_reoptimized"] = (
            OptimizationResult(
                config=config_zero,
                results=reopt_entries,
            )
        )

    # Step 3: Add green team (simulation environment) parameter sensitivity sweeps
    green_team_results = _run_green_team_sensitivity(
        base_config=config_zero,
        baseline_params=baseline_params,
        variation_points=variation_points,
        max_iterations=max_iterations,
    )
    sensitivity_results.update(green_team_results)

    return sensitivity_results


def _run_green_team_sensitivity(
    base_config: OptimizationConfig,
    baseline_params: Dict[str, float],
    variation_points: int,
    max_iterations: int,
) -> Dict[str, OptimizationResult]:
    """Run sensitivity analysis for green team (simulation environment) parameters.

    Unlike red/blue parameter sensitivity which varies optimization variables,
    this varies the simulation environment itself (num_tasks, budget, slack_honeypots)
    by passing override parameters to the win calculator and re-optimizes for each variation.

    This helps understand how robust the defense strategy is to changes in the
    simulation environment assumptions.

    Args:
        base_config: Base configuration
        baseline_params: Baseline red/blue parameter values
        variation_points: Number of sample points per parameter
        max_iterations: Optimization iterations per point

    Returns:
        Dict mapping green parameter names to OptimizationResult objects,
        with both fixed and reoptimized modes for each parameter.
    """
    results: Dict[str, OptimizationResult] = {}

    # Define green team parameters to sweep with their ranges (±1 order of magnitude)
    green_params = {
        "num_tasks": {
            "baseline": base_config.num_tasks,
            "min": base_config.num_tasks / 10.0,
            "max": base_config.num_tasks * 10.0,
            "scale": "log",
            "override_key": "override_num_tasks",  # Key passed to win calculator
        },
        "blue_team_budget_fraction": {
            "baseline": base_config.blue_team_budget_fraction,
            "min": base_config.blue_team_budget_fraction / 10.0,
            "max": base_config.blue_team_budget_fraction * 10.0,
            "scale": "log",
            "override_key": "override_blue_team_budget_fraction",
        },
        "slack_honeypots": {
            "baseline": base_config.slack_honeypots,
            "min": max(0, base_config.slack_honeypots / 10.0),
            "max": base_config.slack_honeypots * 10.0,
            "scale": "log",
            "override_key": "override_slack_honeypots",
        },
    }

    for param_name, param_info in green_params.items():
        baseline_val = param_info["baseline"]
        min_val = param_info["min"]
        max_val = param_info["max"]
        scale = param_info["scale"]
        override_key = param_info["override_key"]

        # Build variation grid based on scale (log or linear)
        if scale == "log":
            # Use geometric spacing for log-scale parameters
            min_c = max(min_val, 1e-12)  # Avoid log(0)
            max_c = max(max_val, min_c * 1.0001)  # Ensure valid range
            sweep_values = np.geomspace(min_c, max_c, num=variation_points)
        else:
            sweep_values = np.linspace(min_val, max_val, num=variation_points)

        # Always include baseline value for direct comparison
        sweep_values = np.append(sweep_values, baseline_val)
        sweep_values = sorted(
            set(float(np.clip(val, min_val, max_val)) for val in sweep_values)
        )

        # Remove near-duplicates (within floating point tolerance)
        unique_values: List[float] = []
        for value in sweep_values:
            if unique_values and abs(value - unique_values[-1]) < 1e-9:
                continue
            unique_values.append(value)

        fixed_entries: List[Dict[str, Any]] = []
        reopt_entries: List[Dict[str, Any]] = []

        for green_value in unique_values:
            # First, evaluate the baseline parameters at this green value (fixed case)
            fixed_result = _evaluate_green_parameter_fixed(
                config=base_config,
                override_key=override_key,
                override_value=green_value,
                baseline_params=baseline_params,
            )

            fixed_entries.append(
                {
                    "safety": fixed_result["safety"],
                    "usefulness": 1.0,
                    "blue_team_replace_fnr": 0.0,
                    "best_parameters": dict(baseline_params),
                    "outcome_distribution": fixed_result["outcome_distribution"],
                    "varied_parameter": param_name,
                    "varied_value": float(green_value),
                    "analysis_mode": "fixed",
                    "converged": True,
                }
            )

            # Run optimization with the green parameter override
            reopt_result = _reoptimize_with_green_parameter(
                config=base_config,
                override_key=override_key,
                override_value=green_value,
                baseline_params=baseline_params,
                max_iterations=max_iterations,
            )

            # Extract mixed strategies and parameters
            blue_mixed_strategy = reopt_result.get("blue_mixed_strategy", [])
            red_params = reopt_result.get("red_best_response") or reopt_result.get(
                "best_parameters", {}
            )

            if blue_mixed_strategy:
                first_blue_strategy = blue_mixed_strategy[0][0]
                reopt_best_params = {**first_blue_strategy, **red_params}
            else:
                reopt_best_params = dict(red_params)

            reopt_best_params = {k: float(v) for k, v in reopt_best_params.items()}

            reopt_outcomes_raw = reopt_result.get("outcome_distribution", {})
            reopt_outcomes: Dict[str, float] = {}
            if isinstance(reopt_outcomes_raw, Mapping):
                for k, v in reopt_outcomes_raw.items():
                    try:
                        reopt_outcomes[k] = float(v)
                    except (TypeError, ValueError):
                        continue

            reopt_entries.append(
                {
                    "safety": float(reopt_result["best_score"]),
                    "usefulness": 1.0,
                    "blue_team_replace_fnr": 0.0,
                    "best_parameters": reopt_best_params,
                    "blue_mixed_strategy": blue_mixed_strategy,
                    "red_mixed_strategy": reopt_result.get("red_mixed_strategy", []),
                    "outcome_distribution": reopt_outcomes,
                    "history": reopt_result.get("history", []),
                    "varied_parameter": param_name,
                    "varied_value": float(green_value),
                    "analysis_mode": "reoptimized",
                    "converged": bool(reopt_result.get("converged", True)),
                }
            )

        fixed_entries.sort(key=lambda entry: entry["varied_value"])
        reopt_entries.sort(key=lambda entry: entry["varied_value"])

        results[f"{param_name}_sensitivity_fixed"] = OptimizationResult(
            config=base_config,
            results=fixed_entries,
        )
        results[f"{param_name}_sensitivity_reoptimized"] = OptimizationResult(
            config=base_config,
            results=reopt_entries,
        )

    return results


def _clone_config_with_replace_fnr(
    base_config: OptimizationConfig, replace_fnr: float
) -> OptimizationConfig:
    """Return a shallow copy of the config with a fresh win calculator.

    This creates a new configuration with an updated blue team replacement FNR
    and rebuilds the win calculator to use this new value. All other config
    parameters are preserved.

    Args:
        base_config: Original optimization configuration
        replace_fnr: New blue team replacement false negative rate

    Returns:
        New OptimizationConfig with updated replace_fnr and win calculator
    """

    win_calc = build_blue_team_win_calculator(
        num_tasks=base_config.num_tasks,
        blue_team_budget_fraction=base_config.blue_team_budget_fraction,
        slack_honeypots=base_config.slack_honeypots,
        blue_team_replace_fnr=replace_fnr,
        roc_function=base_config.monitor_roc_fn,
        untrusted_honest_fpr=base_config.monitor_roc_fn_metadata.get(
            "untrusted_honest_fpr", 0
        ),
        untrusted_honest_fnr=base_config.monitor_roc_fn_metadata.get(
            "untrusted_honest_fnr", 0
        ),
        untrusted_honest_extra_fpr=base_config.monitor_roc_fn_metadata.get(
            "untrusted_honest_extra_fpr_fn"
        ),
        human_audit_success_rate=base_config.human_audit_success_rate,
    )

    metadata_copy = dict(base_config.monitor_roc_fn_metadata or {})

    return replace(
        base_config,
        blue_team_replace_fnr=replace_fnr,
        blue_team_win_calculator=win_calc,
        monitor_roc_fn_metadata=metadata_copy,
        parameter_spaces=deepcopy(base_config.parameter_spaces),
        untrusted_threshold_min=base_config.untrusted_threshold_min,
        untrusted_threshold_max=base_config.untrusted_threshold_max,
    )


def _extract_best_parameters(result: OptimizationResult) -> Dict[str, float]:
    """Return the parameter dict from the highest-safety run.

    Searches through all optimization results and extracts the parameters
    that achieved the highest safety score.

    Args:
        result: Optimization result containing multiple runs

    Returns:
        Dict of parameter names to values for the best run

    Raises:
        TypeError: If best_parameters is missing or not a dictionary
    """

    best_entry = max(
        result.results,
        key=lambda entry: entry.get("safety", float("-inf")),
    )

    best_params = best_entry.get("best_parameters", {})
    if not isinstance(best_params, MutableMapping):
        raise TypeError("Optimization result is missing best_parameters dictionary")

    return {k: float(v) for k, v in best_params.items()}


def _build_variation_grid(
    space: Mapping[str, Any],
    baseline_value: float,
    num_points: int,
) -> List[float]:
    """Construct a monotonic grid over the feasible region for a parameter.

    Creates a grid of values spanning the parameter's min/max range, respecting
    the scale (linear or log). Always includes the baseline value and removes
    near-duplicates.

    Args:
        space: Parameter space definition with 'min', 'max', and 'scale' keys
        baseline_value: Current baseline value (always included in grid)
        num_points: Target number of grid points

    Returns:
        Sorted list of unique grid values including baseline
    """

    min_val = float(space["min"])
    max_val = float(space["max"])
    scale = space["scale"]

    if scale == "log":
        # Guard against zero or negative bounds by clipping to a small epsilon.
        min_c = max(min_val, 1e-12)
        max_c = max(max_val, min_c * 1.0001)
        grid = np.geomspace(min_c, max_c, num=num_points)
    else:
        grid = np.linspace(min_val, max_val, num=num_points)

    grid = np.append(grid, baseline_value)

    # Clamp and deduplicate while preserving order.
    unique_values: List[float] = []
    for value in sorted(set(float(np.clip(val, min_val, max_val)) for val in grid)):
        if unique_values and abs(value - unique_values[-1]) < 1e-9:
            continue
        unique_values.append(value)

    return unique_values


def _evaluate_parameters(
    config: OptimizationConfig,
    params: Mapping[str, float],
) -> tuple[float, Dict[str, float]]:
    """Evaluate the blue team win calculator for a parameter assignment.

    Converts parameter dict to numpy arrays, calls the win calculator, and
    extracts the safety score and outcome distribution.

    Args:
        config: Optimization configuration containing the win calculator
        params: Parameter values to evaluate

    Returns:
        Tuple of (safety_score, outcome_distribution_dict)
    """

    calculator = config.blue_team_win_calculator
    input_params = {k: np.array([float(v)], dtype=float) for k, v in params.items()}
    safety_array, outcome_distribution = calculator(input_params)

    safety = float(np.asarray(safety_array).squeeze())

    clean_outcomes: Dict[str, float] = {}
    for key, value in outcome_distribution.items():
        clean_outcomes[key] = float(np.asarray(value).squeeze())

    return safety, clean_outcomes


def _reoptimize_with_parameter(
    config: OptimizationConfig,
    param_name: str,
    fixed_value: float,
    *,
    parameter_spaces: Mapping[str, Dict[str, Any]],
    max_iterations: int,
) -> Dict[str, Any]:
    """Run optimization with one parameter clamped to a fixed value.

    Creates a modified parameter space where the specified parameter has
    its min and max set to the fixed value, effectively constraining it.
    Then runs grid_search optimization on the remaining free parameters.

    Args:
        config: Optimization configuration
        param_name: Name of parameter to fix
        fixed_value: Value to fix the parameter at
        parameter_spaces: Original parameter space definitions
        max_iterations: Max optimization iterations

    Returns:
        Optimization result dictionary from grid_search

    Raises:
        KeyError: If param_name not in parameter_spaces
    """

    if param_name not in parameter_spaces:
        raise KeyError(f"Unknown parameter for reoptimization: {param_name}")

    # Create a copy of parameter spaces with the specified parameter fixed
    param_data: Dict[str, Dict[str, Any]] = {}
    for name, cfg in parameter_spaces.items():
        param_data[name] = {
            "min": float(cfg["min"]),
            "max": float(cfg["max"]),
            "scale": cfg["scale"],
        }

    # Constrain the target parameter to the fixed value
    param_data[param_name]["min"] = float(fixed_value)
    param_data[param_name]["max"] = float(fixed_value)

    # Run double oracle with the modified parameter space
    return OPTIMIZER(
        param_data,
        config.blue_team_win_calculator,
        max_iterations=max_iterations,
    )


def _evaluate_green_parameter_fixed(
    config: OptimizationConfig,
    override_key: str,
    override_value: float,
    baseline_params: Dict[str, float],
) -> Dict[str, Any]:
    """Evaluate baseline parameters at a green parameter value without re-optimizing.

    Args:
        config: Base optimization config
        override_key: Key to pass to the win calculator (e.g., "override_num_tasks")
        override_value: Value for the green parameter
        baseline_params: Baseline red/blue parameters to evaluate

    Returns:
        Dict with "safety" and "outcome_distribution" keys
    """
    # Create a modified win calculator that injects the green parameter override
    original_calculator = config.blue_team_win_calculator

    def modified_calculator(
        params: Dict[str, np.ndarray],
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Wrapper that injects green team parameter override."""
        params_with_override = dict(params)
        params_with_override[override_key] = np.array([override_value])
        return original_calculator(params_with_override)

    # Convert baseline params to arrays
    params_as_arrays = {k: np.array([v]) for k, v in baseline_params.items()}

    # Evaluate the calculator with the baseline parameters and green override
    safety, outcome_distribution = modified_calculator(params_as_arrays)

    # Convert arrays back to floats
    safety_value = float(safety[0]) if isinstance(safety, np.ndarray) else float(safety)
    outcomes = {
        k: float(v[0]) if isinstance(v, np.ndarray) else float(v)
        for k, v in outcome_distribution.items()
    }

    return {
        "safety": safety_value,
        "outcome_distribution": outcomes,
    }


def _reoptimize_with_green_parameter(
    config: OptimizationConfig,
    override_key: str,
    override_value: float,
    baseline_params: Dict[str, float],
    max_iterations: int,
) -> Dict[str, Any]:
    """Re-optimize parameters with a green parameter override.

    Args:
        config: Base optimization config
        override_key: Key to pass to the win calculator (e.g., "override_num_tasks")
        override_value: Value for the green parameter
        baseline_params: Baseline red/blue parameters to start optimization from
        max_iterations: Max optimization iterations

    Returns:
        Optimization result dictionary
    """

    # Create a modified win calculator that injects the green parameter override
    original_calculator = config.blue_team_win_calculator

    def modified_calculator(
        params: Dict[str, np.ndarray],
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Wrapper that injects green team parameter override."""
        params_with_override = dict(params)
        params_with_override[override_key] = np.array([override_value])
        return original_calculator(params_with_override)

    # Use the standard parameter spaces (red/blue team parameters only)
    parameter_spaces = config.parameter_spaces
    if not parameter_spaces:
        parameter_spaces = construct_parameter_spaces(
            config.untrusted_threshold_min,
            config.untrusted_threshold_max,
        )

    # Run optimization with the modified calculator
    return OPTIMIZER(
        parameter_spaces,
        modified_calculator,
        max_iterations=max_iterations,
    )


__all__ = ["run_parameter_sensitivity_analysis"]
