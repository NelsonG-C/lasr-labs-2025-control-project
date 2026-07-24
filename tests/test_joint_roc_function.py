"""Unit tests for the dependence-aware joint ROC model (phase 1).

Monte-Carlo simulation-based validation lives separately in
tests/monte_carlo/ (deselected by default; run with `pytest -m mc`).
"""

import numpy as np
import pytest

from lasr_labs_2025_control_project.utils.numerical_pipeline.optimization import (
    build_optimization_config,
)
from lasr_labs_2025_control_project.utils.numerical_pipeline.roc_function_builders import (
    build_roc_function,
)

from .conftest import make_score_dataframes


def _reference_product_pass_rate(attacks_df, s, t, c):
    """Independence-model pass rate from the same raw arrays and honest rule.

    pass = P(tm < c) * (1 - P(coll < t AND susp >= s)) with the AND factorized
    per-sample on (coll, susp) (those two belong to the same model; only the
    tm x (coll, susp) independence is under test).
    """
    tm = attacks_df["trusted_suspiciousness_score"].to_numpy()
    coll = attacks_df["untrusted_collusion_score"].to_numpy()
    susp = attacks_df["untrusted_suspiciousness_score"].to_numpy()
    reported = (coll < t) & (susp >= s)
    return np.mean(tm < c) * (1 - np.mean(reported))


def _grid_points(attacks_df, honeypots_df, benign_df):
    """On-grid (t, c, fpr_t) triples so interpolation error is exactly zero."""
    t_values = np.unique(
        np.concatenate(
            [
                attacks_df["untrusted_collusion_score"].to_numpy(),
                honeypots_df["untrusted_collusion_score"].to_numpy(),
            ]
        )
    )
    tm_b = benign_df["trusted_suspiciousness_score"].to_numpy()
    c_values = np.unique(tm_b)
    return [
        (float(t), float(c), float(np.mean(tm_b >= c)))
        for t in t_values
        for c in c_values
    ]


class TestJointVsIndependent:
    def test_reduces_to_product_under_shuffling(self, correlated_dfs):
        """Shuffling the TM column breaks the cross-monitor dependence, so the
        joint model must agree with the product model within sampling noise."""
        attacks_df, honeypots_df, benign_df = correlated_dfs
        shuffled = attacks_df.copy()
        rng = np.random.default_rng(42)
        shuffled["trusted_suspiciousness_score"] = rng.permutation(
            shuffled["trusted_suspiciousness_score"].to_numpy()
        )

        fn, meta = build_roc_function(
            attacks_df=shuffled, honeypots_df=honeypots_df, benign_df=benign_df
        )
        s = meta["untrusted_honest_threshold"]
        n = len(shuffled)
        tol = 4.0 * 0.5 / np.sqrt(n)  # 4 sigma of a Bernoulli mean

        worst = max(
            abs(fn(t, f)[0] - _reference_product_pass_rate(shuffled, s, t, c))
            for t, c, f in _grid_points(shuffled, honeypots_df, benign_df)
        )
        assert worst < tol, f"joint deviates from product by {worst:.4f} > {tol:.4f}"

    def test_detects_dependence(self, correlated_dfs):
        """On correlated scores, the joint model must deviate materially from
        the product model somewhere on the grid."""
        attacks_df, honeypots_df, benign_df = correlated_dfs
        fn, meta = build_roc_function(
            attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
        )
        s = meta["untrusted_honest_threshold"]
        worst = max(
            abs(fn(t, f)[0] - _reference_product_pass_rate(attacks_df, s, t, c))
            for t, c, f in _grid_points(attacks_df, honeypots_df, benign_df)
        )
        assert worst > 0.03, f"dependence not reflected (max deviation {worst:.4f})"

    def test_legacy_flag_returns_old_model(self, correlated_dfs):
        attacks_df, honeypots_df, benign_df = correlated_dfs
        fn_new, meta_new = build_roc_function(
            attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
        )
        fn_old, meta_old = build_roc_function(
            attacks_df=attacks_df,
            honeypots_df=honeypots_df,
            benign_df=benign_df,
            assume_independent=True,
        )
        assert fn_new.type == "joint_empirical"
        assert meta_new["model_type"] == "joint_empirical"
        assert fn_old.type == "uncorrelated"
        assert meta_old["model_type"] == "independent"
        # Legacy behavior preserved exactly: epsilon key stays unset (=> 0).
        assert "untrusted_monitor_epsilon" not in meta_old

    def test_disabled_monitor_falls_back_to_legacy(self, correlated_dfs):
        attacks_df, honeypots_df, benign_df = correlated_dfs
        fn, _meta = build_roc_function(
            attacks_df=attacks_df,
            honeypots_df=honeypots_df,
            benign_df=benign_df,
            disable_untrusted_monitor=True,
        )
        assert fn.type == "uncorrelated"


class TestAsymmetricHonestErrors:
    def test_epsilon_wired_and_asymmetric(self, correlated_dfs):
        attacks_df, honeypots_df, benign_df = correlated_dfs
        _, meta = build_roc_function(
            attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
        )
        s = meta["untrusted_honest_threshold"]
        susp_b = benign_df["untrusted_suspiciousness_score"].to_numpy()
        assert meta["untrusted_monitor_epsilon"] == pytest.approx(np.mean(susp_b >= s))
        assert 0.0 < meta["untrusted_monitor_epsilon"] < 1.0
        assert 0.0 < meta["untrusted_honest_fnr"] < 1.0
        # The point of the change: the two directions are not tied together.
        assert meta["untrusted_monitor_epsilon"] != pytest.approx(
            meta["untrusted_honest_fnr"], abs=1e-6
        )

    def test_epsilon_reaches_win_calculator(self, correlated_dfs):
        attacks_df, honeypots_df, benign_df = correlated_dfs
        config = build_optimization_config(
            num_tasks=2000,
            blue_team_budget_fraction=0.02,
            slack_honeypots=2,
            attacks_df=attacks_df,
            honeypots_df=honeypots_df,
            benign_df=benign_df,
        )
        assert config.monitor_roc_fn_metadata["untrusted_monitor_epsilon"] > 0
        params = {
            "blue_team_honeypot_fraction": np.float64(0.5),
            "red_team_roc_threshold": np.float64(5.0),
            "red_team_p_attack": np.float64(1e-3),
        }
        win_prob, outcomes = config.blue_team_win_calculator(params)
        assert 0.0 <= float(win_prob) <= 1.0
        assert sum(outcomes.values()) == pytest.approx(1.0, abs=1e-6)

    def test_missing_suspiciousness_falls_back_to_perfect_honest_um(self):
        attacks_df, honeypots_df, benign_df = make_score_dataframes(rho=0.5, seed=3)
        attacks_df = attacks_df.drop(columns=["untrusted_suspiciousness_score"])
        fn, meta = build_roc_function(
            attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
        )
        assert fn.type == "joint_empirical"
        assert meta["untrusted_monitor_epsilon"] == 0.0
        assert meta["untrusted_honest_fnr"] == 0.0


class TestInterface:
    def test_broadcasting_and_scalars(self, correlated_dfs):
        attacks_df, honeypots_df, benign_df = correlated_dfs
        fn, _ = build_roc_function(
            attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
        )
        # Scalar in, scalar out
        p, f = fn(5.0, 0.02)
        assert isinstance(p, float) and isinstance(f, float)
        # Meshgrid-style paired arrays (visualization.py heatmap convention)
        t_arr = np.linspace(0.0, 10.0, 7)
        f_arr = np.linspace(0.0, 1.0, 7)
        p_arr, fpr_arr = fn(t_arr, f_arr)
        assert p_arr.shape == (7,) and fpr_arr.shape == (7,)
        # Array threshold with scalar fpr (grid search convention)
        p_arr, fpr_arr = fn(t_arr, 0.02)
        assert p_arr.shape == (7,) and fpr_arr.shape == (7,)

    def test_none_conventions(self, correlated_dfs):
        attacks_df, honeypots_df, benign_df = correlated_dfs
        fn, _ = build_roc_function(
            attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
        )
        # (t, None): raw UM collusion ROC point
        coll_rate, _hp_fpr = fn(np.array([0.0, 10.5]), None)
        assert coll_rate[0] == pytest.approx(1.0)  # everything colludes at t=0
        assert coll_rate[1] == pytest.approx(0.0, abs=0.01)
        # (None, f): TM-only attack FNR
        fnr, zeros = fn(None, np.array([0.0, 1.0]))
        assert fnr[0] == pytest.approx(1.0)
        assert fnr[1] == pytest.approx(0.0)
        assert np.all(zeros == 0.0)
        with pytest.raises(ValueError):
            fn(None, None)

    def test_monotonicity(self, correlated_dfs):
        attacks_df, honeypots_df, benign_df = correlated_dfs
        fn, _ = build_roc_function(
            attacks_df=attacks_df, honeypots_df=honeypots_df, benign_df=benign_df
        )
        f_axis = np.linspace(0.0, 1.0, 50)
        pass_rates, _ = fn(np.full_like(f_axis, 5.0), f_axis)
        assert np.all(np.diff(pass_rates) <= 1e-12)  # more auditing never helps red
        t_axis = np.linspace(-0.5, 11.0, 50)
        pass_rates, hp_fpr = fn(t_axis, np.full_like(t_axis, 0.02))
        assert np.all(
            np.diff(pass_rates) <= 1e-12
        )  # colluding less never helps red pass
        assert np.all(np.diff(hp_fpr) <= 1e-12)  # ...but does reduce honeypot risk
        # Bounds respected everywhere
        assert np.all((pass_rates >= 0) & (pass_rates <= 1))
        assert np.all((hp_fpr >= 0) & (hp_fpr <= 1))
