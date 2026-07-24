"""Monte-Carlo validation of the stats model (sequestered).

These tests validate analytic model outputs against direct simulation. They
are deselected by default (pyproject addopts `-m "not mc"`); run with:

    uv run pytest tests/monte_carlo -m mc

Covers:
1. Off-grid joint-surface values are achievable: randomizing thresholds over
   adjacent grid nodes with the bilinear weights reproduces the interpolated
   pass rate (the SPEC's core achievability claim, on the real API).
2. epsilon matches the simulated benign false-flag rate.
3. The blue-team win calculator's outcome distribution matches an episode
   simulator that follows the model's own event semantics (per failed
   honeypot, red is caught with probability 1/slack — the PGF reading of the
   exp(-lambda/slack) term; Poisson approximated by binomial).
"""

import numpy as np
import pytest

from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (
    build_optimization_config,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.roc_function_builders import (
    build_roc_function,
)

from ..conftest import make_score_dataframes

pytestmark = [pytest.mark.mc, pytest.mark.slow]

N_MC = 400_000


class TestJointSurfaceAchievability:
    def test_offgrid_value_is_randomized_threshold_mixture(self):
        attacks_df, honeypots_df, benign_df = make_score_dataframes(rho=0.6, seed=7)
        fn, meta = build_roc_function(
            attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
        )
        s = meta["untrusted_honest_threshold"]
        rng = np.random.default_rng(11)

        tm_a = attacks_df["trusted_suspiciousness_score"].to_numpy()
        coll_a = attacks_df["untrusted_collusion_score"].to_numpy()
        susp_a = attacks_df["untrusted_suspiciousness_score"].to_numpy()
        tm_b = benign_df["trusted_suspiciousness_score"].to_numpy()

        t_query, f_query = 4.3, 0.037  # off-grid on both axes

        # Red mixture: adjacent integer collusion thresholds around t_query
        t0, t1 = np.floor(t_query), np.ceil(t_query)
        wt = (t_query - t0) / (t1 - t0)
        # Blue mixture: adjacent achievable benign-FPR levels around f_query
        c_levels = np.unique(tm_b)[::-1]
        f_levels = np.array([np.mean(tm_b >= c) for c in c_levels])
        k = np.searchsorted(f_levels, f_query)
        c0, c1 = c_levels[k - 1], c_levels[k]
        wf = (f_query - f_levels[k - 1]) / (f_levels[k] - f_levels[k - 1])

        idx = rng.integers(0, len(coll_a), size=N_MC)
        t_draw = np.where(rng.uniform(size=N_MC) < wt, t1, t0)
        c_draw = np.where(rng.uniform(size=N_MC) < wf, c1, c0)
        colludes = coll_a[idx] >= t_draw
        reported = ~colludes & (susp_a[idx] >= s)
        tm_flags = tm_a[idx] >= c_draw
        mc_pass = np.mean(~tm_flags & ~reported)

        analytic_pass, _ = fn(t_query, f_query)
        sigma = np.sqrt(0.25 / N_MC)
        assert analytic_pass == pytest.approx(mc_pass, abs=6 * sigma + 1e-3)

    def test_epsilon_matches_simulated_benign_flag_rate(self):
        attacks_df, honeypots_df, benign_df = make_score_dataframes(rho=0.6, seed=8)
        _, meta = build_roc_function(
            attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
        )
        s = meta["untrusted_honest_threshold"]
        rng = np.random.default_rng(12)
        susp_b = benign_df["untrusted_suspiciousness_score"].to_numpy()
        flags = susp_b[rng.integers(0, len(susp_b), size=N_MC)] >= s
        assert meta["untrusted_monitor_epsilon"] == pytest.approx(
            np.mean(flags), abs=6 * np.sqrt(0.25 / N_MC)
        )


class TestGameLevelSimulation:
    def _simulate_outcomes(
        self,
        *,
        rng: np.random.Generator,
        n_episodes: int,
        num_tasks: int,
        num_honeypots: float,
        honest_error_sum: float,
        u_fpr: float,
        slack: float,
        p_attack: float,
        attack_catch_rate: float,
        attack_pass_rate: float,
    ) -> dict[str, float]:
        """Episode simulator following the analytic model's event semantics."""
        # Binomial honeypot failures (physical reality). Honest errors dilute
        # the retraining signal, shrinking effective honeypots by
        # 1 - (FPR + FNR). The analytic model uses the exact binomial PGF
        # (1 - u_fpr/slack)^n_eff, so this matches up to rounding n_eff to an
        # integer for the draw.
        n_hp = rng.binomial(
            int(round(num_honeypots * max(1 - honest_error_sum, 0.0))),
            u_fpr,
            size=n_episodes,
        )
        # Each failed honeypot independently catches red w.p. 1/slack
        # (PGF: E[(1-1/s)^X] = (1 - u_fpr/s)^n, the analytic term).
        fails_honeypots = rng.uniform(size=n_episodes) > (1 - 1 / slack) ** n_hp

        caught_upfront = rng.uniform(size=n_episodes) < 1 - (1 - p_attack) ** 10
        caught_upfront &= ~fails_honeypots

        alive = ~fails_honeypots & ~caught_upfront
        # Remaining tasks: each attack is decisive w.p. catch+pass; the model's
        # p_no_attacks treats fizzled attacks as non-events.
        p_decisive = p_attack * (attack_catch_rate + attack_pass_rate)
        any_decisive = rng.uniform(size=n_episodes) < 1 - (1 - p_decisive) ** (
            num_tasks - 10
        )
        no_attacks = alive & ~any_decisive
        p_first_caught = attack_catch_rate / (
            attack_catch_rate + attack_pass_rate + 1e-9
        )
        first_caught = (
            alive & any_decisive & (rng.uniform(size=n_episodes) < p_first_caught)
        )
        red_wins = alive & any_decisive & ~first_caught

        return {
            "red_team_fails_honeypots": float(np.mean(fails_honeypots)),
            "red_team_introduces_no_attacks": float(np.mean(no_attacks)),
            "red_team_caught": float(np.mean(caught_upfront | first_caught)),
            "red_team_victory": float(np.mean(red_wins)),
        }

    def test_outcome_distribution_matches_simulation(self):
        attacks_df, honeypots_df, benign_df = make_score_dataframes(rho=0.6, seed=9)
        num_tasks, budget_fraction, slack = 2000, 0.02, 2.0
        config = build_optimization_config(
            num_tasks=num_tasks,
            blue_team_budget_fraction=budget_fraction,
            slack_honeypots=slack,
            attacks_df=attacks_df,
            honeypots_df=honeypots_df,
            benign_df=benign_df,
        )
        epsilon = config.monitor_roc_fn_metadata["untrusted_monitor_epsilon"]
        honest_fnr = config.monitor_roc_fn_metadata["untrusted_honest_fnr"]
        fn = config.monitor_roc_fn
        rng = np.random.default_rng(13)

        for honeypot_fraction, t, p_attack in [
            (0.5, 5.0, 1e-3),
            (0.2, 7.5, 1e-2),
            (0.8, 3.0, 3e-3),
        ]:
            params = {
                "blue_team_honeypot_fraction": np.float64(honeypot_fraction),
                "red_team_roc_threshold": np.float64(t),
                "red_team_p_attack": np.float64(p_attack),
            }
            _, analytic = config.blue_team_win_calculator(params)

            # Reconstruct the same intermediate rates the calculator uses.
            budget = budget_fraction * num_tasks
            num_honeypots = budget * honeypot_fraction
            audit_fraction = (budget - num_honeypots) / num_tasks
            trusted_fnr = max(audit_fraction - epsilon, 0.0)
            p_audited = min(audit_fraction / (epsilon + 1 / num_tasks), 1.0)
            not_flagged, u_fpr = fn(t, trusted_fnr)
            catch_rate = (1 - not_flagged) * p_audited
            pass_rate, _ = fn(t, min(trusted_fnr, 1.0))

            simulated = self._simulate_outcomes(
                rng=rng,
                n_episodes=N_MC,
                num_tasks=num_tasks,
                num_honeypots=num_honeypots,
                honest_error_sum=epsilon + honest_fnr,
                u_fpr=u_fpr,
                slack=slack,
                p_attack=p_attack,
                attack_catch_rate=catch_rate,
                attack_pass_rate=pass_rate,
            )
            for key, sim_value in simulated.items():
                # 6 MC sigma plus slack for the Poisson-vs-binomial and
                # sequencing approximations in the analytic composition.
                assert analytic[key] == pytest.approx(sim_value, abs=0.01), (
                    f"{key} analytic={float(analytic[key]):.4f} "
                    f"simulated={sim_value:.4f} at "
                    f"(h={honeypot_fraction}, t={t}, p={p_attack})"
                )
