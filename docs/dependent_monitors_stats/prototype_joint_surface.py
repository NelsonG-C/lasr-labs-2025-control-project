"""Prototype: empirical joint attack-pass surface with multilinear interpolation.

Validates the SPEC.md design:
1. On independent scores, the joint surface reproduces the product-of-marginals
   (current `uncorrelated_roc_function`) model within sampling noise.
2. On correlated scores, the two diverge (quantifies what the reviewers asked for).
3. Multilinear interpolation exactly equals the expected pass rate under
   independent randomization over adjacent grid thresholds (achievability).

Run: uv run --no-project --with numpy,scipy python prototype_joint_surface.py
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
N = 2000  # attack samples


def make_attack_scores(rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer 0-10 TM and UM-collusion scores with Gaussian-copula dependence rho."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0, 0], cov, size=N)
    u = stats.norm.cdf(z)
    tm = np.clip(np.round(stats.norm.ppf(u[:, 0], loc=6.5, scale=2.0)), 0, 10)
    coll = np.clip(np.round(stats.norm.ppf(u[:, 1], loc=5.5, scale=2.5)), 0, 10)
    return tm, coll


def joint_pass_grid(tm: np.ndarray, coll: np.ndarray):
    """Empirical P_pass(t, c) = P(coll >= t AND tm < c) on the observed grid."""
    t_grid = np.concatenate([[coll.min() - 1], np.unique(coll), [coll.max() + 1]])
    c_grid = np.concatenate([[tm.min() - 1], np.unique(tm), [tm.max() + 1]])
    surface = np.array(
        [[np.mean((coll >= t) & (tm < c)) for c in c_grid] for t in t_grid]
    )
    return t_grid, c_grid, surface


def bilinear(t_grid, c_grid, surface, t, c):
    it = np.clip(np.searchsorted(t_grid, t) - 1, 0, len(t_grid) - 2)
    ic = np.clip(np.searchsorted(c_grid, c) - 1, 0, len(c_grid) - 2)
    wt = (t - t_grid[it]) / (t_grid[it + 1] - t_grid[it])
    wc = (c - c_grid[ic]) / (c_grid[ic + 1] - c_grid[ic])
    return (
        surface[it, ic] * (1 - wt) * (1 - wc)
        + surface[it + 1, ic] * wt * (1 - wc)
        + surface[it, ic + 1] * (1 - wt) * wc
        + surface[it + 1, ic + 1] * wt * wc
    )


def product_model(tm, coll, t, c):
    """Current independence model: P(coll >= t) * P(tm < c)."""
    return np.mean(coll >= t) * np.mean(tm < c)


print(f"{'rho':>5} {'max |joint - product|':>22} {'at (t, c)':>12}")
for rho in [0.0, 0.3, 0.6]:
    tm, coll = make_attack_scores(rho)
    t_grid, c_grid, surface = joint_pass_grid(tm, coll)
    diffs = [
        (abs(np.mean((coll >= t) & (tm < c)) - product_model(tm, coll, t, c)), t, c)
        for t in np.unique(coll)
        for c in np.unique(tm)
    ]
    worst = max(diffs)
    print(f"{rho:>5.1f} {worst[0]:>22.4f} {f'({worst[1]:.0f}, {worst[2]:.0f})':>12}")

# Achievability: bilinear interpolation == expectation under independent
# randomization over the two adjacent pure thresholds on each axis.
tm, coll = make_attack_scores(0.5)
t_grid, c_grid, surface = joint_pass_grid(tm, coll)
t_query, c_query = 4.3, 6.7  # off-grid on both axes
interp = bilinear(t_grid, c_grid, surface, t_query, c_query)
it = np.searchsorted(t_grid, t_query) - 1
ic = np.searchsorted(c_grid, c_query) - 1
wt = (t_query - t_grid[it]) / (t_grid[it + 1] - t_grid[it])
wc = (c_query - c_grid[ic]) / (c_grid[ic + 1] - c_grid[ic])
draws_t = rng.choice([t_grid[it], t_grid[it + 1]], size=200_000, p=[1 - wt, wt])
draws_c = rng.choice([c_grid[ic], c_grid[ic + 1]], size=200_000, p=[1 - wc, wc])
idx = rng.integers(0, N, size=200_000)
mc = np.mean((coll[idx] >= draws_t) & (tm[idx] < draws_c))
print(f"\nachievability check at t={t_query}, c={c_query}:")
print(f"  bilinear interpolation : {interp:.5f}")
print(f"  monte-carlo mixture    : {mc:.5f}  (agree within MC noise ~0.001)")
assert abs(interp - mc) < 0.005
print("OK")
