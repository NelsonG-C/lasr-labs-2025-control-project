"""Offline unit tests for the Tinker LoRA trainer's data adapter.

The live training loop is exercised separately (needs a key + costs tokens); here
we pin down the cross-entropy Datum construction, which is the easy thing to get
wrong (shift + answer-only loss mask).
"""

from __future__ import annotations

import sys
import types as pytypes

import pytest


class _StubTok:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict=False, **kw):
        return {"input_ids": [1, 2, 3]}  # 3-token prompt

    def encode(self, text, add_special_tokens=False):
        return [10, 11]  # 2-token answer


class _StubModelInput:
    def __init__(self, ids):
        self._ids = list(ids)

    @classmethod
    def from_ints(cls, ids):
        return cls(ids)

    def to_ints(self):
        return self._ids


class _StubDatum:
    def __init__(self, model_input, loss_fn_inputs):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


@pytest.fixture
def stub_finetune(monkeypatch):
    from lasr_labs_2025_control_project.utils import tinker_provider as tp

    monkeypatch.setattr(tp, "_tokenizer", lambda base_model: _StubTok())
    fake_types = pytypes.SimpleNamespace(
        ModelInput=_StubModelInput, Datum=_StubDatum
    )
    monkeypatch.setitem(sys.modules, "tinker", pytypes.SimpleNamespace(types=fake_types))
    from lasr_labs_2025_control_project.finetuning import tinker_lora_finetune as ft

    return ft


def test_build_datum_shift_and_answer_only_mask(stub_finetune):
    ft = stub_finetune
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "<answer>YES</answer>"},
    ]
    datum = ft.build_datum(messages, "Qwen/Qwen3-30B-A3B-Instruct-2507")

    # tokens = prompt[1,2,3] + answer[10,11] + eos[99]
    assert datum.model_input.to_ints() == [1, 2, 3, 10, 11]  # tokens[:-1]
    assert datum.loss_fn_inputs["target_tokens"] == [2, 3, 10, 11, 99]  # tokens[1:]
    # weights: 1.0 only where the target is an answer/EOS token (prompt masked)
    assert datum.loss_fn_inputs["weights"] == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_build_datum_first_answer_token_is_trained(stub_finetune):
    """The position predicting the first answer token must have weight 1."""
    ft = stub_finetune
    messages = [
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "<answer>NO</answer>"},
    ]
    datum = ft.build_datum(messages, "m")
    tgt = datum.loss_fn_inputs["target_tokens"]
    w = datum.loss_fn_inputs["weights"]
    first_answer_pos = tgt.index(10)  # first answer token id
    assert w[first_answer_pos] == 1.0
    assert all(x == 0.0 for x in w[:first_answer_pos])  # prompt not trained


def test_load_examples_roundtrip(tmp_path, stub_finetune):
    ft = stub_finetune
    p = tmp_path / "train.jsonl"
    p.write_text(
        '{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<answer>YES</answer>"}]}\n'
        "\n"  # blank line tolerated
        '{"messages": [{"role": "user", "content": "b"}, {"role": "assistant", "content": "<answer>NO</answer>"}]}\n'
    )
    examples = ft.load_examples(p)
    assert len(examples) == 2

    (tmp_path / "empty.jsonl").write_text("")
    with pytest.raises(ValueError):
        ft.load_examples(tmp_path / "empty.jsonl")


def test_default_base_model_is_u_not_t(stub_finetune):
    # Guard the instruction: SFT U (30B), never default to the small T.
    assert stub_finetune.DEFAULT_BASE_MODEL == "Qwen/Qwen3-30B-A3B-Instruct-2507"
