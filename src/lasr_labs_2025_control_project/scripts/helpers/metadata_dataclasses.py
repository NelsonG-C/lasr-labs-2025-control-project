from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from control_arena.eval_mode import EvalMode


class TransformType(StrEnum):
    """The type of the transform.

    Possible values are:
    - LLM = "llm"
    - MINIFIER = "minifier"
    - NONE = "none"
    """

    LLM = "llm"
    MINIFIER = "minifier"
    NONE = "none"


class RunLabel(StrEnum):
    """The label of a run.

    Possible values are:
    - ATTACKS = "attacks"
    - HONEYPOTS = "honeypots"
    - HONEST = "honest"
    """

    ATTACKS = "attacks"
    HONEYPOTS = "honeypots"
    HONEST = "honest"


@dataclass
class ModelInfo:
    """
    Information about the model used for generation.

    Attributes:
        model: Name of the model.
        system_prompt: System prompt used for generation.
        base_prompt: Base (user) prompt used for generation.
    """

    model: str
    system_prompt: str
    base_prompt: Optional[str] = "{solution}"


@dataclass
class GeneratorInfo:
    """
    Information about the generator used for generation.

    Attributes:
        model_info: Information about the model used for generation.
        eval_mode: The evaluation mode.
        secret_input: The secret input.
    """

    model_info: ModelInfo
    eval_mode: EvalMode
    secret_input: Optional[str]


@dataclass
class TransformInfo:
    """
    Information about the transform used for transformation.

    Attributes:
        type: The type of the transform.
        model_info: Information about the model used for transformation.
    """

    type: TransformType
    model_info: Optional[ModelInfo] = None
