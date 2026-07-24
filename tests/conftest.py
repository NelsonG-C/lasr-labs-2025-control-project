"""Shared fixtures: synthetic monitor-score dataframes with known dependence.

Integer 0-10 scores (matching real monitor output) generated through a
Gaussian copula so tests can dial the dependence between the trusted
suspiciousness, untrusted collusion, and untrusted suspiciousness scores
on attacks.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


def _discretize(u: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return np.clip(np.round(stats.norm.ppf(u, loc=loc, scale=scale)), 0, 10)


def make_score_dataframes(
    rho: float,
    n_attacks: int = 3000,
    n_honeypots: int = 1500,
    n_benign: int = 3000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic (attacks_df, honeypots_df, benign_df).

    On attacks, (tm, coll, susp) share a Gaussian copula with correlation rho
    between tm and coll and rho/2 to susp. Honeypot and benign scores are
    independent draws (dependence only matters on the attack joint).
    """
    rng = np.random.default_rng(seed)

    cov = np.array(
        [
            [1.0, rho, rho / 2],
            [rho, 1.0, rho / 2],
            [rho / 2, rho / 2, 1.0],
        ]
    )
    z = rng.multivariate_normal(np.zeros(3), cov, size=n_attacks)
    u = stats.norm.cdf(z)
    attacks_df = pd.DataFrame(
        {
            "trusted_suspiciousness_score": _discretize(u[:, 0], 6.5, 2.0),
            "untrusted_collusion_score": _discretize(u[:, 1], 5.5, 2.5),
            "untrusted_suspiciousness_score": _discretize(u[:, 2], 7.0, 2.0),
        }
    )
    honeypots_df = pd.DataFrame(
        {
            "untrusted_collusion_score": _discretize(
                rng.uniform(size=n_honeypots), 3.0, 2.5
            ),
        }
    )
    benign_df = pd.DataFrame(
        {
            "trusted_suspiciousness_score": _discretize(
                rng.uniform(size=n_benign), 3.0, 2.0
            ),
            "untrusted_suspiciousness_score": _discretize(
                rng.uniform(size=n_benign), 2.5, 2.0
            ),
        }
    )
    return attacks_df, honeypots_df, benign_df


@pytest.fixture
def correlated_dfs():
    return make_score_dataframes(rho=0.6)


@pytest.fixture
def independent_dfs():
    return make_score_dataframes(rho=0.0, seed=1)
