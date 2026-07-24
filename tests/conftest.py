"""Shared fixtures for the Tinker backend tests.

Offline tests use stub tokenizer / sampling clients (no network, no API key) to
exercise the pure logic in ``utils/tinker_provider.py``. Live tests
(``tests/test_tinker_live.py``) are skipped automatically when ``TINKER_API_KEY``
is unset.
"""

from __future__ import annotations

import os
import sys
import types as pytypes

import pytest

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
