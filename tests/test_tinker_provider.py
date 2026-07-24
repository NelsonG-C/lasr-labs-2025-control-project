"""Offline unit tests for utils/tinker_provider.py (no network / no API key).

Live end-to-end tests against the real Tinker API live in test_tinker_live.py.
"""

from __future__ import annotations

import pytest
from inspect_ai.model import GenerateConfig, get_model

from lasr_labs_2025_control_project.utils import tinker_provider as tp


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_is_tinker_model():
    assert tp.is_tinker_model("tinker/Qwen/Qwen3-32B")
    assert not tp.is_tinker_model("openai/gpt-4.1")
    assert not tp.is_tinker_model("openai-api/together/Qwen/Qwen2.5-7B-Instruct-Turbo")


def test_strip_prefix():
    assert tp.strip_prefix("tinker/Qwen/Qwen3-32B") == "Qwen/Qwen3-32B"
    # idempotent / no-op for unprefixed names
    assert tp.strip_prefix("Qwen/Qwen3-32B") == "Qwen/Qwen3-32B"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([1, 2, 3], [1, 2, 3]),  # already flat
        ({"input_ids": [4, 5]}, [4, 5]),  # BatchEncoding-as-dict
        ({"input_ids": [[6, 7, 8]]}, [6, 7, 8]),  # nested batch dim
    ],
)
def test_normalize_ids(raw, expected):
    assert tp._normalize_ids(raw) == expected


def test_normalize_ids_batchencoding_attr():
    class FakeBatchEncoding:
        input_ids = [9, 10, 11]

    assert tp._normalize_ids(FakeBatchEncoding()) == [9, 10, 11]


# --------------------------------------------------------------------------- #
# Provider registration (needs inspect only; no key)
# --------------------------------------------------------------------------- #
def test_provider_registered_and_resolves(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    model = get_model("tinker/Qwen/Qwen3-32B")
    assert type(model.api).__name__ == "TinkerAPI"
    # prefix is stripped by get_model before reaching the provider
    assert model.api.base_model == "Qwen/Qwen3-32B"


def test_service_client_requires_key(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    tp._service_client = None  # reset process-wide cache
    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        tp._service()


# --------------------------------------------------------------------------- #
# Forced-decoding collusion logprobs (stubbed clients)
# --------------------------------------------------------------------------- #
async def test_collusion_logprobs_sums_multitoken(stub_tinker):
    module, _sc, _tok = stub_tinker
    result = await module.collusion_yes_no_logprobs(
        "tinker/Qwen/Qwen3-32B", "system", "user"
    )
    # YES is one token (-0.5); NO is two tokens (-1.0 + -2.0) summed.
    assert result["YES"] == pytest.approx(-0.5)
    assert result["NO"] == pytest.approx(-3.0)
    # YES has higher logprob -> completion reports YES
    assert result["completion"] == "<answer>YES</answer>"


async def test_collusion_logprobs_shared_prefix(stub_tinker):
    """Both candidates must be scored against the same prompt+<answer> prefix."""
    _module, sc, _tok = stub_tinker
    await _module.collusion_yes_no_logprobs("tinker/m", "system", "user")
    # two compute_logprobs calls (YES, NO); both share the 4-token prefix
    # ([1,2,3] chat template + [900] for "<answer>")
    assert len(sc.logprob_calls) == 2
    assert sc.logprob_calls[0][:4] == [1, 2, 3, 900]
    assert sc.logprob_calls[1][:4] == [1, 2, 3, 900]
    assert sc.logprob_calls[0][4:] == [10]  # YES
    assert sc.logprob_calls[1][4:] == [20, 21]  # NO


async def test_generate_via_provider(stub_tinker):
    module, _sc, _tok = stub_tinker
    api = get_model("tinker/Qwen/Qwen3-32B").api
    out = await api.generate(
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(max_tokens=8, temperature=0.0),
    )
    # stub emits token id 10 -> tokenizer decodes to "YES"
    assert out.completion == "YES"
    assert out.stop_reason == "stop"
