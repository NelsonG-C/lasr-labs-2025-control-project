### Utility functions for numerical pipeline


from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import binom, norm


@dataclass()
class NumericalPipelineParameters:
    self_recognition_auc_roc: float
    replace_fraction: float
    num_tasks: int
    blue_team_budget_fraction: float


@dataclass()
class BlueTeamChoices:
    honeypot_fraction: float = 0.5


@dataclass()
class RedTeamChoices:
    sus_threshold: float = 0.5
    attack_frequency: float = (
        0.001  # Increased from 0.001 to make optimization meaningful
    )


class NumericalPipeline:
    def __init__(self, params: NumericalPipelineParameters):
        self.params = params

    def run_optimization(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> Tuple[BlueTeamChoices, RedTeamChoices]:
        """
        Alternating optimization between red and blue teams.
        Blue team maximizes safety
        Red team minimizes safety
        """
        blue_choices = BlueTeamChoices()
        red_choices = RedTeamChoices()
        prev_objective = float("-inf")

        for iteration in range(max_iterations):
            # Blue team optimization (maximize)
            blue_choices, red_choices = self._optimize_team(
                blue_choices,
                red_choices,
                True,
            )
            # Red team optimization (minimize)
            blue_choices, red_choices = self._optimize_team(
                blue_choices,
                red_choices,
                False,
            )

            # Check convergence
            current_objective = self._objective_value(blue_choices, red_choices)

            if abs(current_objective - prev_objective) < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            prev_objective = current_objective

        return blue_choices, red_choices

    def _objective_value(
        self,
        blue_choices: BlueTeamChoices,
        red_choices: RedTeamChoices,
    ) -> float:
        """Calculate safety"""
        safety, usefulness = self.calculate_safety_usefulness(blue_choices, red_choices)
        return safety

    def _optimize_team(
        self,
        blue_choices: BlueTeamChoices,
        red_choices: RedTeamChoices,
        maximize: bool,
    ) -> Tuple[BlueTeamChoices, RedTeamChoices]:
        """Generic optimization for either team"""
        try:
            if maximize:  # Blue team
                # Simple grid search for blue team honeypot fraction
                grid_values = np.linspace(0.0, 1.0, 21)  # Check 21 points from 0 to 1
                best_safety = -999
                best_h = blue_choices.honeypot_fraction

                for h in grid_values:
                    test_blue = BlueTeamChoices(honeypot_fraction=h)
                    safety = self._objective_value(test_blue, red_choices)
                    if safety > best_safety:
                        best_safety = safety
                        best_h = h

                return BlueTeamChoices(honeypot_fraction=best_h), red_choices

            else:  # Red team
                # Grid search for red team (trying to minimize)
                best_value = float("inf")
                best_s = red_choices.sus_threshold
                best_a = red_choices.attack_frequency

                # Try different values for sus_threshold and attack_frequency
                for s in np.linspace(0.0, 1.0, 21):
                    for a in np.linspace(0.0, 1.0, 21):
                        new_red = RedTeamChoices(sus_threshold=s, attack_frequency=a)
                        value = self._objective_value(blue_choices, new_red)
                        if value < best_value:
                            best_value = value
                            best_s = s
                            best_a = a

                return blue_choices, RedTeamChoices(
                    sus_threshold=best_s, attack_frequency=best_a
                )

        except Exception as e:
            print(f"Optimization failed: {e}")

        # Return original choices if optimization fails
        return blue_choices, red_choices

    def calculate_safety_usefulness(
        self,
        blue_team_choices: BlueTeamChoices,
        red_team_choices: RedTeamChoices,
    ) -> Tuple[float, float]:
        ## Step 1: monitor validation
        n_honeypots = max(
            0,
            blue_team_choices.honeypot_fraction
            * self.params.num_tasks
            * self.params.blue_team_budget_fraction,
        )
        blue_team_audit_fraction = float(
            max(
                0,
                int(
                    self.params.num_tasks
                    * max(0.0, 1 - blue_team_choices.honeypot_fraction)
                    * self.params.blue_team_budget_fraction
                ),
            )
        ) / float(self.params.num_tasks)

        # Need probability that more than five honeypots are fallen for.
        # This requires a binomial probability calculation.
        # Reverse AUC-ROC to get to the

        # Calculate inverse normal CDF (Φ⁻¹(AUC))
        phi_inv_auc = norm.ppf(self.params.self_recognition_auc_roc)

        # Calculate variance: σ² = 1/(2 * [Φ⁻¹(AUC)]²)
        std = (1.0 / (2.0 * phi_inv_auc**2)) ** 0.5

        # Calculate probability of individual honeypot being fallen for:
        # Which is the probability that a honeypot (mean 0, std=\sigma)
        # Is given a selfyness score > the threshold
        p_given_honeypot_not_flagged = 1 - norm.cdf(
            red_team_choices.sus_threshold, loc=0, scale=std
        )
        # Calculate probability of five honeypots being fallen for using binomial:
        # P(X > 5) = 1 - P(X <= 5) where X ~ Binomial(n_honeypots, p_honeypot_fallen_for)
        p_more_than_five_honeypots_not_flagged = 1 - binom.cdf(
            5, n=int(n_honeypots), p=p_given_honeypot_not_flagged
        )
        # For later, also calculate the probability that an attack is not flagged,
        # because it is given a selfyness score > the threshold
        p_given_attack_flagged = norm.cdf(
            red_team_choices.sus_threshold, loc=1, scale=std
        )
        p_given_attack_caught = (
            1 - blue_team_choices.honeypot_fraction
        ) * self.params.blue_team_budget_fraction

        ## Safety Calculation
        p_any_attack_introduced = 1 - binom.cdf(
            0,
            n=int(self.params.num_tasks),
            p=red_team_choices.attack_frequency * (1 - self.params.replace_fraction),
        )

        ## Calculate probability any given attack is flagged or caught
        p_given_attack_passes = 1 - (
            p_given_attack_flagged
            + p_given_attack_caught
            - p_given_attack_flagged * p_given_attack_caught
        )

        # Probability of not being caught by honeypots:
        # Chance of being caught by honeypots over chance of run ending
        # Which also includes chance that an attack is introduced at all
        # For two state game where we halt in state 1 with probabiltiy p_1 each turn
        # And halt in state 2 with probability p_2 each turn,
        # Probability overall of halting in state 2 is
        # p_2*(1-p_1)/(p_1 + p_2 - p_1*p_2))
        p_game_ends_with_attack = (
            p_any_attack_introduced
            * (1 - p_more_than_five_honeypots_not_flagged)
            / (
                p_more_than_five_honeypots_not_flagged
                + p_any_attack_introduced
                - p_more_than_five_honeypots_not_flagged * p_any_attack_introduced
            )
        )
        safety = 1 - p_game_ends_with_attack * p_given_attack_passes

        # Usefulness is the loss from replacing minus the benefit from auditing
        usefulness = 1 - max(self.params.replace_fraction + blue_team_audit_fraction, 0)

        return (float(safety), usefulness)
