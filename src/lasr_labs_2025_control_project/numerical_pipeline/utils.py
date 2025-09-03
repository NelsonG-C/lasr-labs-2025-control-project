### Utility functions for numerical pipeline

from typing import Callable, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp


def optimize_parameters(
    num_tasks: float,
    blue_team_budget_fraction: float,
    red_team_tpr_values: List[float],
    red_team_fpr_values: List[float],
) -> Dict[str, Union[Dict[str, float], float]]:
    """Does the thing TODO: write better description."""
    red_team_roc_fn = build_roc_inverse_interpolator(
        tpr=jnp.array(red_team_tpr_values), fpr=jnp.array(red_team_fpr_values)
    )

    blue_team_win_calculator = build_blue_team_win_calculator(
        num_tasks=num_tasks,
        blue_team_budget_fraction=blue_team_budget_fraction,
        red_team_roc_function=red_team_roc_fn,
    )

    # Blue team ascends, red team descends
    gradient_change_rate = {
        "blue_team_honeypot_fraction": 0.1,
        "red_team_roc_threshold": -0.1,
        "red_team_p_attack": -0.1,
    }

    params = {
        "blue_team_honeypot_fraction": 0.5,
        "red_team_roc_threshold": 0.5,
        "red_team_p_attack": 0.001,
    }

    gradient_function = jax.grad(blue_team_win_calculator)

    for _ in range(1000):
        gradients = gradient_function(params)

        gradient_sum = jnp.sqrt(
            jax.tree.reduce(
                lambda acc, x: acc + jnp.sum(x**2), gradients, jnp.array(0.0)
            )
        )
        if gradient_sum < 1e-6:
            break

        params = jax.tree.map(
            lambda value, gradient, change_rate: value + change_rate * gradient,
            params,
            gradients,
            gradient_change_rate,
        )

    blue_team_win_probability = blue_team_win_calculator(params)

    return {"params": params, "blue_team_win_probability": blue_team_win_probability}


def build_blue_team_win_calculator(
    num_tasks: float,
    blue_team_budget_fraction: float,
    red_team_roc_function: Callable[[float], Tuple[float, float]],
) -> Callable[[Dict[str, float]], float]:
    """Build a win probability calculator for the blue team

    Args:
        num_tasks: Number of tasks in the simulation
        blue_team_budget_fraction: Blue team honeypot + audit budget
        red_team_roc_function: Red team ROC function
    """

    @jax.jit
    def calculate_blue_team_win(
        params: Dict[str, float],
    ) -> jnp.ndarray:
        blue_team_honeypot_fraction = params["blue_team_honeypot_fraction"]
        red_team_roc_threshold = params["red_team_roc_threshold"]
        red_team_p_attack = params["red_team_p_attack"]

        true_positive_rate, false_positive_rate = red_team_roc_function(
            red_team_roc_threshold
        )

        # Simulate honeypots
        num_honeypots = (
            blue_team_budget_fraction * num_tasks * blue_team_honeypot_fraction
        )
        # Calculate the probability of red team passing honeypots
        # Binomial 0-4 is sum
        # n_honeypots, n_honeypots-1, n_honeypots-2 ...
        fpr_powers = num_honeypots - jnp.arange(0, 5)
        tpr_powers = jnp.arange(0, 5)
        # 1, n_honeypots, n_honeypots*(n_honeypots-1), ...
        numerators = jnp.cumprod(num_honeypots - jnp.arange(-1, 4)) / (
            num_honeypots + 1
        )
        # Factorials for the denominators
        denominators = jnp.array([1, 1, 2, 6, 24])

        binomial_probs = (
            true_positive_rate**tpr_powers
            * false_positive_rate**fpr_powers
            * numerators
            / denominators
        )
        p_red_team_fails_honeypots = 1 - jnp.sum(binomial_probs)

        # Probability of returning to honeypots
        p_go_back_to_honeypots = (1 - red_team_p_attack) ** num_tasks

        # Calculate probability red team is caught
        false_negative_rate = 1 - true_positive_rate
        random_audit_rate = (
            1 - blue_team_honeypot_fraction
        ) * blue_team_budget_fraction
        p_given_attack_caught = (
            false_negative_rate
            + random_audit_rate
            - false_negative_rate * random_audit_rate
        )

        p_red_team_caught = (
            1 - (1 - p_given_attack_caught * red_team_p_attack) ** num_tasks
        )

        # Use Claude's formula from the markov chain
        p_red_team_victory = (
            (1 - p_red_team_caught - p_go_back_to_honeypots)
            * (1 - p_red_team_fails_honeypots)
            / (1 - (1 - p_red_team_fails_honeypots) * p_go_back_to_honeypots)
        )

        return 1 - p_red_team_victory

    return calculate_blue_team_win


def build_roc_inverse_interpolator(
    tpr: jnp.ndarray, fpr: jnp.ndarray
) -> Callable[[float], Tuple[float, float]]:
    """
    Build an inverse interpolator for ROC curves parameterized by (tpr+fpr)/2.

    Args:
        tpr: True positive rates (sorted)
        fpr: False positive rates (sorted)

    Returns:
        A jitted function that maps (tpr+fpr)/2 values to (tpr, fpr) pairs
    """
    # Ensure arrays are 1D
    tpr = jnp.atleast_1d(tpr)
    fpr = jnp.atleast_1d(fpr)

    # Check if we need to prepend (0,0)
    needs_origin = not (jnp.isclose(tpr[0], 0) & jnp.isclose(fpr[0], 0))
    if needs_origin:
        tpr = jnp.concatenate([jnp.array([0.0]), tpr])
        fpr = jnp.concatenate([jnp.array([0.0]), fpr])

    # Check if we need to append (1,1)
    needs_endpoint = not (jnp.isclose(tpr[-1], 1) & jnp.isclose(fpr[-1], 1))
    if needs_endpoint:
        tpr = jnp.concatenate([tpr, jnp.array([1.0])])
        fpr = jnp.concatenate([fpr, jnp.array([1.0])])

    # Compute the sum values for each point
    sum_values = (tpr + fpr) / 2.0

    # For inverse interpolation, we need sum_values to be monotonically increasing
    # In case of ties, we'll use the first occurrence (implicitly handled by searchsorted)

    @jax.jit
    def inverse_interpolate(target_sum: float) -> Tuple[float, float]:
        """
        Given target (tpr+fpr)/2 values, return corresponding (tpr, fpr) pairs.

        Args:
            target_sum: Target sum values (scalar or array)

        Returns:
            Tuple of (tpr, fpr) arrays with same shape as input
        """
        # Preserve input shape for output
        target_sum_flat = jnp.atleast_1d(target_sum).ravel()

        # Clamp to valid range [0, 1]
        target_sum_flat = jnp.clip(target_sum_flat, 0.0, 1.0)

        # Find indices for interpolation using searchsorted
        # searchsorted with 'right' gives us the index where we'd insert to maintain order
        # We want the segment [i-1, i] for interpolation
        indices = jnp.searchsorted(sum_values, target_sum_flat, side="right")

        # Clamp indices to valid range for interpolation
        indices = jnp.clip(indices, 1, len(sum_values) - 1)

        # Get the left and right points for interpolation
        i0 = indices - 1
        i1 = indices

        # Get sum values at these points
        sum0 = sum_values[i0]
        sum1 = sum_values[i1]

        # Compute interpolation weights
        # Handle case where sum0 == sum1 (flat segments)
        denom = sum1 - sum0
        # Use where to avoid division by zero
        alpha = jnp.where(
            denom > 0,
            (target_sum_flat - sum0) / denom,
            0.0,  # If denominator is 0, use left point (alpha=0)
        )

        # Linearly interpolate tpr and fpr
        tpr_interp = tpr[i0] * (1 - alpha) + tpr[i1] * alpha
        fpr_interp = fpr[i0] * (1 - alpha) + fpr[i1] * alpha

        # Reshape back to original shape
        tpr_result = tpr_interp.item()
        fpr_result = fpr_interp.item()

        return tpr_result, fpr_result

    return inverse_interpolate
