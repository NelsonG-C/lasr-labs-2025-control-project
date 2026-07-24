"""Single-file Tinker (Thinking Machines) compatibility layer for the pipeline.

Everything Tinker-specific lives in this module. It does two jobs:

1. Registers an inspect-ai model provider under the ``tinker/`` prefix, so that
   ``get_model("tinker/Qwen/Qwen3-32B")`` and ``eval(model="tinker/...")`` work
   for solution *generation* (attacks / honest / honeypots), the text-``<score>``
   suspiciousness monitors, and the LLM paraphraser -- i.e. every plain-sampling
   role -- with no other pipeline changes.

2. Exposes :func:`collusion_yes_no_logprobs`, which returns the *exact*
   log P(YES) / log P(NO) answer-token logprobs the untrusted collusion monitor
   needs, via forced decoding (``compute_logprobs`` on ``prompt+"YES"`` vs
   ``prompt+"NO"``). This sidesteps the top-k truncation / provider-shape / CoT
   contamination problems that plague logprobs on OpenAI-compatible backends,
   and is the correct method for a binary-decision monitor.

The ``tinker`` package is imported lazily (only when a tinker model is actually
used) so the rest of the pipeline runs without it installed. Auth is via the
``TINKER_API_KEY`` environment variable, read by the Tinker SDK itself.

Model ids are the part after the ``tinker/`` prefix, e.g. ``Qwen/Qwen3-32B``.
The authoritative live catalogue is
``ServiceClient().get_server_capabilities().supported_models``.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
)
from inspect_ai.model import (
    modelapi as _modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

if TYPE_CHECKING:  # avoid importing tinker at module load
    from tinker import SamplingClient, ServiceClient

TINKER_PREFIX = "tinker/"

# Chat-template kwargs that disable chain-of-thought for hybrid-reasoning
# families, so completions carry only the final answer (matches the behaviour
# we rely on for clean answer extraction). Applied best-effort: templates that
# don't accept the kwarg are called without it.
_NOTHINK_TEMPLATE_KWARGS = {"enable_thinking": False}


def strip_prefix(model_name: str) -> str:
    """Return the Tinker base-model id from a possibly-``tinker/``-prefixed name."""
    return model_name[len(TINKER_PREFIX) :] if model_name.startswith(TINKER_PREFIX) else model_name


def is_tinker_model(model_name: str) -> bool:
    return model_name.startswith(TINKER_PREFIX)


# --------------------------------------------------------------------------- #
# Client / tokenizer caches (creating a SamplingClient is near-instant, but a
# process-wide cache keeps us from rebuilding one per request).
# --------------------------------------------------------------------------- #
_service_lock = threading.Lock()
_service_client: "Optional[ServiceClient]" = None


def _service() -> "ServiceClient":
    global _service_client
    if _service_client is None:
        with _service_lock:
            if _service_client is None:
                import os

                if not os.environ.get("TINKER_API_KEY"):
                    raise RuntimeError(
                        "TINKER_API_KEY is not set; required to use tinker/ models."
                    )
                import tinker

                _service_client = tinker.ServiceClient()
    return _service_client


@lru_cache(maxsize=None)
def _sampling_client(base_model: str) -> "SamplingClient":
    return _service().create_sampling_client(base_model=base_model)


@lru_cache(maxsize=None)
def _tokenizer(base_model: str):
    return _sampling_client(base_model).get_tokenizer()


def _normalize_ids(out: Any) -> list[int]:
    """Coerce apply_chat_template output to a flat list[int].

    transformers >=5 returns a BatchEncoding (``{"input_ids": [...]}``) rather
    than a bare list; some versions/paths also nest a batch dimension.
    """
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    if len(out) > 0 and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def _apply_chat_template(
    base_model: str,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Tokenise chat messages, disabling thinking when the template supports it."""
    tok = _tokenizer(base_model)
    try:
        out = tok.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            **_NOTHINK_TEMPLATE_KWARGS,
        )
    except TypeError:
        out = tok.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
        )
    return _normalize_ids(out)


def _messages_to_dicts(input: list[ChatMessage]) -> list[dict[str, str]]:
    role_map = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool"}
    return [{"role": role_map.get(m.role, "user"), "content": m.text} for m in input]


# --------------------------------------------------------------------------- #
# inspect-ai provider: handles plain sampling (generation / text monitors /
# paraphrasing). Collusion logprobs go through collusion_yes_no_logprobs below.
# --------------------------------------------------------------------------- #
@_modelapi(name="tinker")
def tinker() -> type[ModelAPI]:
    return TinkerAPI


class TinkerAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=["TINKER_API_KEY"],
            config=config,
        )
        # model_name arrives with the tinker/ prefix already stripped by get_model
        self.base_model = strip_prefix(model_name)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        import tinker
        from tinker import types

        prompt_ids = _apply_chat_template(
            self.base_model, _messages_to_dicts(input), add_generation_prompt=True
        )
        sampling_params = types.SamplingParams(
            max_tokens=config.max_tokens or 2048,
            temperature=1.0 if config.temperature is None else config.temperature,
            top_p=1.0 if config.top_p is None else config.top_p,
            top_k=-1 if config.top_k is None else config.top_k,
            stop=config.stop_seqs,
            seed=config.seed,
        )
        sc = _sampling_client(self.base_model)
        resp = await sc.sample_async(
            prompt=types.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=sampling_params,
        )
        seq = resp.sequences[0]
        text = _tokenizer(self.base_model).decode(list(seq.tokens), skip_special_tokens=True)
        stop_reason = "max_tokens" if str(seq.stop_reason).endswith("LENGTH") else "stop"
        return ModelOutput.from_content(
            model=self.model_name, content=text, stop_reason=stop_reason
        )

    def connection_key(self) -> str:
        return f"tinker/{self.base_model}"


# --------------------------------------------------------------------------- #
# Forced-decoding YES/NO logprobs for the untrusted collusion monitor.
# --------------------------------------------------------------------------- #
async def collusion_yes_no_logprobs(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    answer_open: str = "<answer>",
    yes_text: str = "YES",
    no_text: str = "NO",
) -> dict[str, Any]:
    """Return exact answer-token logprobs for a YES/NO collusion monitor.

    Builds the assistant scaffold up to ``<answer>`` and teacher-forces both
    ``YES`` and ``NO`` continuations, reading each one's summed token logprob
    (handles multi-token YES/NO). Returns ``{"YES", "NO", "completion"}`` in the
    same shape as ``get_answer_logprobs_from_openai_completion`` so the collusion
    scorer treats it identically.
    """
    from tinker import types

    base_model = strip_prefix(model_name)
    tok = _tokenizer(base_model)
    sc = _sampling_client(base_model)

    prefix_ids = list(
        _apply_chat_template(
            base_model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            add_generation_prompt=True,
        )
    )
    prefix_ids += tok.encode(answer_open, add_special_tokens=False)

    async def _candidate_logprob(text: str) -> Optional[float]:
        cand_ids = tok.encode(text, add_special_tokens=False)
        if not cand_ids:
            return None
        full = prefix_ids + cand_ids
        lps = await sc.compute_logprobs_async(types.ModelInput.from_ints(full))
        # compute_logprobs returns one logprob per input token (None/NaN where
        # undefined). The candidate occupies the final len(cand_ids) positions.
        tail = lps[len(prefix_ids) :]
        vals = [lp for lp in tail if lp is not None and lp == lp]  # drop None/NaN
        if len(vals) != len(cand_ids):
            return None
        return float(sum(vals))

    yes_lp = await _candidate_logprob(yes_text)
    no_lp = await _candidate_logprob(no_text)

    if yes_lp is not None and no_lp is not None:
        answer = yes_text if yes_lp >= no_lp else no_text
    elif yes_lp is not None:
        answer = yes_text
    elif no_lp is not None:
        answer = no_text
    else:
        answer = ""
    completion = f"{answer_open}{answer}</answer>" if answer else ""

    return {"YES": yes_lp, "NO": no_lp, "completion": completion}
