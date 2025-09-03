import math
from typing import Callable, Dict

import numpy as np
import pytest

from lasr_labs_2025_control_project.numerical_pipeline.utils import (
    build_blue_team_win_calculator,
    build_roc_inverse_interpolator,
    generate_tpr_and_fpr_values,
)


def simulate_blue_win_monte_carlo(
    *,
    num_tasks: int,
    blue_team_budget_fraction: float,
    blue_team_honeypot_fraction: float,
    red_team_attack_frequency: float,
    red_team_roc_fn: Callable[
        [float | np.ndarray], tuple[float | np.ndarray, float | np.ndarray]
    ],
    threshold: float,
    trials: int = 50_000,
    seed: int = 123,
) -> float:
    """Monte Carlo simulator of the narrative in the docstring.

    Returns proportion of Blue wins over `trials` simulated games.
    """
    rng = np.random.default_rng(seed)

    tpr, fpr = red_team_roc_fn(threshold)
    fnr = 1 - tpr

    n_audits = int(
        round(num_tasks * blue_team_budget_fraction * (1 - blue_team_honeypot_fraction))
    )
    p_random_audit = n_audits / num_tasks

    n_honeypots = int(
        round(blue_team_budget_fraction * num_tasks * blue_team_honeypot_fraction)
    )

    blue_wins = 0

    for _ in range(trials):
        while True:
            # Honeypot phase: red misclassifies honeypots as real with prob fpr.
            # If 5 or more are thought real -> red loses immediately (blue wins).
            if n_honeypots > 0:
                n_false_real = rng.binomial(n=n_honeypots, p=fpr)
                if n_false_real >= 5:
                    blue_wins += 1
                    break

            # Attack phase
            attacks = rng.binomial(n=1, p=red_team_attack_frequency, size=num_tasks)
            n_attacks = int(attacks.sum())

            if n_attacks == 0:
                # Go back to honeypots, repeat
                continue

            # Each attack is flagged by the model with probability = FNR.
            # If flagged, blue wins.
            caught_by_model = rng.binomial(n=1, p=fnr, size=n_attacks)
            if int(caught_by_model.sum()) > 0:
                blue_wins += 1
                break

            # Otherwise random audits may catch them
            caught_by_audit = rng.binomial(n=1, p=p_random_audit, size=n_attacks)
            if int(caught_by_audit.sum()) > 0:
                blue_wins += 1
                break

            # If none caught, red wins
            break

    return blue_wins / trials


@pytest.mark.parametrize(
    "num_tasks, budget, honeypot_frac, red_p_attack, threshold, auc",
    [
        (100, 0.2, 0.5, 0.01, 0.5, 0.80),
        (200, 0.1, 0.3, 0.001, 0.2, 0.99),
        (50, 0.4, 0.8, 0.02, 0.8, 0.70),
    ],
)
def test_blue_team_win_percentage_matches_monte_carlo(
    num_tasks: int,
    budget: float,
    honeypot_frac: float,
    red_p_attack: float,
    threshold: float,
    auc: float,
):
    # Build ROC function from the existing utilities using AUC-ROC
    tpr_vals, fpr_vals = generate_tpr_and_fpr_values(auc_roc=auc, num_values=200)
    roc_fn = build_roc_inverse_interpolator(
        tpr=np.array(tpr_vals), fpr=np.array(fpr_vals)
    )

    calc = build_blue_team_win_calculator(
        num_tasks=num_tasks,
        blue_team_budget_fraction=budget,
        red_team_roc_function=roc_fn,
    )

    # Deterministic function under test
    params: Dict[str, np.ndarray] = {
        "blue_team_honeypot_fraction": np.array(honeypot_frac, dtype=float),
        "red_team_roc_threshold": np.array(threshold, dtype=float),
        "red_team_p_attack": np.array(red_p_attack, dtype=float),
    }
    deterministic_win = float(calc(params))

    # Monte Carlo reference
    mc_win = simulate_blue_win_monte_carlo(
        num_tasks=num_tasks,
        blue_team_budget_fraction=budget,
        blue_team_honeypot_fraction=honeypot_frac,
        red_team_attack_frequency=red_p_attack,
        red_team_roc_fn=roc_fn,
        threshold=threshold,
        trials=40_000,
        seed=42,
    )

    # Allow a reasonable tolerance as the closed-form makes approximations
    assert math.isclose(deterministic_win, mc_win, rel_tol=0.15, abs_tol=0.03), (
        f"deterministic={deterministic_win:.4f}, monte_carlo={mc_win:.4f}"
    )
