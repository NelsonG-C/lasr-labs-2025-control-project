"""Shared test fixtures.

Two families live here:

1. Synthetic monitor-score dataframes with known dependence (Gaussian copula,
   integer 0-10 scores) for the joint-ROC / stats-model tests.
2. Stub tokenizer / sampling clients for the Tinker backend tests (offline, no
   network or API key). Live tests (``tests/test_tinker_live.py``) are skipped
   automatically when ``TINKER_API_KEY`` is unset.
"""

from __future__ import annotations

import os
import sys
import types as pytypes

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
    susp_attack: tuple[float, float] = (7.0, 2.0),
    susp_benign: tuple[float, float] = (2.5, 2.0),
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
            "untrusted_suspiciousness_score": _discretize(u[:, 2], *susp_attack),
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
                rng.uniform(size=n_benign), *susp_benign
            ),
        }
    )
    return attacks_df, honeypots_df, benign_df


def make_boolean_score_dataframes(
    rho_attack: float = 0.5,
    benign_error_corr: float = 0.0,
    n_attacks: int = 3000,
    n_honeypots: int = 1500,
    n_benign: int = 5000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic dataframes for the binary-honest-verdict schema.

    Attacks carry (tm, coll, honest-bool) with rho_attack correlating the TM
    score with both the collusion score and the latent behind the honest
    flag — so honest misses (FNR direction) co-occur with TM misses.
    Benign rows carry (tm, honest-bool) with benign_error_corr correlating
    the TM score with the honest-flag latent — so honest false flags (FPR
    direction) co-occur with TM false alarms.
    """
    rng = np.random.default_rng(seed)

    cov_a = np.array(
        [
            [1.0, rho_attack, rho_attack],
            [rho_attack, 1.0, rho_attack / 2],
            [rho_attack, rho_attack / 2, 1.0],
        ]
    )
    z = rng.multivariate_normal(np.zeros(3), cov_a, size=n_attacks)
    u = stats.norm.cdf(z)
    attacks_df = pd.DataFrame(
        {
            "trusted_suspiciousness_score": _discretize(u[:, 0], 6.5, 2.0),
            "untrusted_collusion_score": _discretize(u[:, 1], 5.5, 2.5),
            "untrusted_honest_score": stats.norm.ppf(u[:, 2], loc=7.0, scale=2.0)
            >= 5.0,
        }
    )
    honeypots_df = pd.DataFrame(
        {
            "untrusted_collusion_score": _discretize(
                rng.uniform(size=n_honeypots), 3.0, 2.5
            ),
        }
    )
    cov_b = np.array([[1.0, benign_error_corr], [benign_error_corr, 1.0]])
    zb = rng.multivariate_normal(np.zeros(2), cov_b, size=n_benign)
    ub = stats.norm.cdf(zb)
    benign_df = pd.DataFrame(
        {
            "trusted_suspiciousness_score": _discretize(ub[:, 0], 3.0, 2.0),
            "untrusted_honest_score": stats.norm.ppf(ub[:, 1], loc=2.5, scale=2.0)
            >= 5.5,
        }
    )
    return attacks_df, honeypots_df, benign_df


@pytest.fixture
def correlated_dfs():
    return make_score_dataframes(rho=0.6)


@pytest.fixture
def independent_dfs():
    return make_score_dataframes(rho=0.0, seed=1)


# --- Tinker backend stubs (offline tests for utils/tinker_provider.py) ---


# Token ids used by the stub tokenizer. "NO" is deliberately multi-token so the
# forced-decoding logprob summation is exercised.
STUB_VOCAB = {"<answer>": [900], "YES": [10], "NO": [20, 21]}
_PREFIX_IDS = [1, 2, 3]  # what the stub chat template renders to (pre-<answer>)
# logprob assigned to each token id by the stub compute_logprobs
_STUB_LOGPROBS = {10: -0.5, 20: -1.0, 21: -2.0}


class StubTokenizer:
    """Minimal transformers-tokenizer stand-in for offline tests."""

    def apply_chat_template(
        self, messages, add_generation_prompt, tokenize, return_dict=False, **kwargs
    ):
        # Return a BatchEncoding-like dict to exercise _normalize_ids.
        return {"input_ids": list(_PREFIX_IDS)}

    def encode(self, text, add_special_tokens=False):
        return list(STUB_VOCAB.get(text, [999]))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            "YES" if i == 10 else "NO" if i in (20, 21) else "x" for i in ids
        )


class _StubSeq:
    def __init__(self, tokens, stop_reason="STOP_REASON_STOP"):
        self.tokens = tokens
        self.stop_reason = stop_reason


class _StubSampleResponse:
    def __init__(self, sequences):
        self.sequences = sequences


class StubSamplingClient:
    """Records calls; returns deterministic tokens/logprobs."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.sample_calls = []
        self.logprob_calls = []

    def get_tokenizer(self):
        return self._tokenizer

    async def sample_async(self, prompt, num_samples, sampling_params, **kwargs):
        self.sample_calls.append(prompt.to_ints())
        return _StubSampleResponse([_StubSeq([10])])  # emits a "YES" token

    async def compute_logprobs_async(self, model_input):
        ids = model_input.to_ints()
        self.logprob_calls.append(ids)
        # position 0 undefined (matches real API), rest keyed by token id
        return [None] + [_STUB_LOGPROBS.get(t, -0.01) for t in ids[1:]]


class _StubModelInput:
    def __init__(self, ids):
        self._ids = list(ids)

    @classmethod
    def from_ints(cls, ids):
        return cls(ids)

    def to_ints(self):
        return self._ids


@pytest.fixture
def stub_tinker(monkeypatch):
    """Patch tinker_provider's client/tokenizer accessors with stubs.

    Returns (module, sampling_client, tokenizer) for assertions.
    """
    from lasr_labs_2025_control_project.utils import tinker_provider as tp

    tokenizer = StubTokenizer()
    sampling_client = StubSamplingClient(tokenizer)
    monkeypatch.setattr(tp, "_tokenizer", lambda base_model: tokenizer)
    monkeypatch.setattr(tp, "_sampling_client", lambda base_model: sampling_client)

    # Provide a fake `tinker` module so `from tinker import types` inside the
    # module resolves to our stub types.
    fake_types = pytypes.SimpleNamespace(
        ModelInput=_StubModelInput,
        SamplingParams=lambda **kw: pytypes.SimpleNamespace(**kw),
    )
    monkeypatch.setitem(
        sys.modules, "tinker", pytypes.SimpleNamespace(types=fake_types)
    )
    return tp, sampling_client, tokenizer


@pytest.fixture(scope="session")
def tinker_base_model():
    """A small live Tinker base model; skips the whole test if unavailable."""
    if not os.environ.get("TINKER_API_KEY"):
        pytest.skip("TINKER_API_KEY not set")
    import tinker

    caps = tinker.ServiceClient().get_server_capabilities()
    available = [m.model_name for m in caps.supported_models]
    for candidate in (
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-4B",
        "meta-llama/Llama-3.2-1B",
    ):
        if candidate in available:
            return candidate
    if available:
        return available[0]
    pytest.skip("no Tinker models available")
