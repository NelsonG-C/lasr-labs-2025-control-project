"""Live end-to-end tests against the real Tinker API.

Skipped automatically unless ``TINKER_API_KEY`` is set (see the
``tinker_base_model`` fixture). Marked ``slow`` since they make real API calls;
deselect with ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

from lasr_labs_2025_control_project.utils import tinker_provider as tp

pytestmark = pytest.mark.slow

_ANSWER_SYSTEM = (
    "You will be given a question. Answer only in an <answer> tag, "
    "like <answer>YES</answer> or <answer>NO</answer>."
)


def test_capabilities_nonempty(tinker_base_model):
    import tinker

    caps = tinker.ServiceClient().get_server_capabilities()
    assert len(caps.supported_models) > 0


async def test_generate_via_provider_returns_text(tinker_base_model):
    model = get_model(f"tinker/{tinker_base_model}")
    out = await model.generate(
        [
            ChatMessageSystem(content="You are terse."),
            ChatMessageUser(content="Say the single word: hello"),
        ],
        config=GenerateConfig(max_tokens=16, temperature=0.0),
    )
    assert isinstance(out.completion, str) and out.completion.strip() != ""


async def test_collusion_logprobs_exact_and_discriminate(tinker_base_model):
    model_name = f"tinker/{tinker_base_model}"
    true_stmt = await tp.collusion_yes_no_logprobs(
        model_name, _ANSWER_SYSTEM, "Is 7 a prime number?"
    )
    false_stmt = await tp.collusion_yes_no_logprobs(
        model_name, _ANSWER_SYSTEM, "Is 8 a prime number?"
    )

    # exact logprobs present for both answer tokens
    for d in (true_stmt, false_stmt):
        assert d["YES"] is not None and d["NO"] is not None
        assert d["YES"] <= 0.0 and d["NO"] <= 0.0  # logprobs are non-positive

    # the collusion score (YES - NO) should be higher for the true statement
    score_true = true_stmt["YES"] - true_stmt["NO"]
    score_false = false_stmt["YES"] - false_stmt["NO"]
    assert score_true > score_false
