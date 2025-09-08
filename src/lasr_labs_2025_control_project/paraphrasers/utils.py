import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import python_minifier
from pydantic import Field

from lasr_labs_2025_control_project.self_rec.utils.scripts.data_models import (
    ProblemSolutions,
    Solution,
)


@dataclass()
class ParaphrasedSolution(Solution):
    solution: str = Field(description="The paraphrased solution")
    original_solution: str = Field(
        description="The original solution before paraphrasing"
    )
    paraphraser_type: str = Field(description="The type of paraphraser used")
    paraphraser_kwargs: Optional[Dict] = None

    metadata: dict = Field(
        default_factory=dict,
        description="Metadata, like 'paraphrased', 'formatted', etc",
    )

    def __init__(
        self,
        original_solution: Solution,
        paraphrased_solution: str,
        paraphraser_type: str,
        paraphraser_kwargs: Optional[Dict[Any, Any]] = None,
    ):
        """A Solution object that also stores the original solution for reference."""
        self.solution = paraphrased_solution
        self.original_solution = original_solution.solution

        for field in original_solution.__dataclass_fields__:
            if field not in ["solution"]:
                setattr(self, field, getattr(original_solution, field))

        self.metadata = self.metadata or {}
        self.metadata["paraphraser_type"] = paraphraser_type
        if paraphraser_kwargs is not None:
            self.metadata["paraphraser_kwargs"] = paraphraser_kwargs


@dataclass(frozen=True)
class ParaphraserConfig:
    paraphraser_type: Literal["llm", "minifier", "none"]
    kwargs: Optional[Dict] = None

    def __post_init__(self):
        if self.paraphraser_type == "minifier":
            assert self.kwargs is None, "minfier does not take any kwargs"
            return

        if self.paraphraser_type == "llm":
            assert self.kwargs is not None and set(self.kwargs.keys()) == {
                "model_name",
            }, "llm paraphraser requires kwargs with keys 'model_name'"

            raise NotImplementedError("LLM paraphraser not implemented yet")

        if self.paraphraser_type == "none":
            return

        raise Exception(f"Unknown paraphraser type: {self.paraphraser_type}")


class Paraphraser:
    """Base class for paraphrasers. Use Paraphraser.build(config) to create an instance of a specific paraphraser type."""

    paraphraser_type: str
    paraphraser_kwargs: Optional[Dict] = None

    @classmethod
    def build(cls, config: ParaphraserConfig):
        if config.paraphraser_type == "minifier":
            return MinifierParaphraser()

        if config.paraphraser_type == "llm":
            return LLMParaphraser(
                model_name=config.kwargs["model_name"],  # type: ignore
            )

        if config.paraphraser_type == "none":
            return NoParaphraser()

        raise Exception(f"Unknown paraphraser type: {config.paraphraser_type}")

    def __call__(self, solutions: List[ProblemSolutions]) -> List[ProblemSolutions]:
        """Paraphrase the solutions in the given list of ProblemSolutions."""

        problem_solutions_list = []
        for problem_solutions in solutions:
            solutions_list: List[Solution] = []
            for solution in problem_solutions.solutions:
                paraphrased_solution = self._paraphrase_single_solution(
                    solution.solution
                )

                solutions_list.append(
                    ParaphrasedSolution(
                        original_solution=solution,
                        paraphrased_solution=paraphrased_solution,
                        paraphraser_type=self.paraphraser_type,
                        paraphraser_kwargs=self.paraphraser_kwargs,
                    )
                )
            problem_solutions_list.append(
                ProblemSolutions(
                    apps_problem_id=problem_solutions.apps_problem_id,
                    problem_statement=problem_solutions.problem_statement,
                    solutions=solutions_list,
                )
            )

        return problem_solutions_list

    def _paraphrase_single_solution(_self, _text: str) -> str:
        raise NotImplementedError(
            "Somehow you've managed to build the base Paraphraser class without throwing an exception. Well done. I hope you're very proud of yourself."
        )

    def __init__(_self):
        raise Exception(
            "Paraphraser is an abstract base class and should not be instantiated directly."
            " Please use Paraphraser.build(config) to create an instance of a specific paraphraser type."
        )


class NoParaphraser(Paraphraser):
    """A paraphraser that does nothing, for code consistency and type checking purposes."""

    def __init__(self):
        self.paraphraser_type = "none"
        self.paraphraser_kwargs = None

    def __call__(_self, text: str) -> str:
        """Does nothing."""
        return text


class MinifierParaphraser(Paraphraser):
    """A paraphraser that minifies code using the `code_minifier` package."""

    def __init__(self):
        self.paraphraser_type = "minifier"
        self.paraphraser_kwargs = None

    def _paraphrase_single_solution(_self, text: str) -> str:
        """Minify the given text string."""

        text = text.replace("```python\n", "").replace("```", "")

        return python_minifier.minify(text)


class LLMParaphraser(Paraphraser):
    def __init__(self, model_name: str):
        self.paraphraser_type = "llm"
        self.paraphraser_kwargs = {"model_name": model_name}

        # Assert regex match for model_name = model_provider/model
        assert type(model_name) is str and re.match(r"^[^/]+/[^/]+$", model_name), (
            "model_name must be of the form 'model_provider/model'"
        )

        self.model_provider, self.model_name = model_name.split("/")

        raise NotImplementedError("LLM paraphraser not implemented yet")
