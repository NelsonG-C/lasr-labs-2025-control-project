from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import python_minifier

from lasr_labs_2025_control_project.self_rec.utils.scripts.data_models import (
    ProblemSolutions,
    Solution,
)


@dataclass()
class ParaphrasedSolution(Solution):
    model: str
    solution: str
    original_solution: str


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
                "model_provider",
                "model_name",
            }, (
                "llm paraphraser requires kwargs with keys 'model_provider' and 'model_name'"
            )

            raise NotImplementedError("LLM paraphraser not implemented yet")

        if self.paraphraser_type == "none":
            return

        raise Exception(f"Unknown paraphraser type: {self.paraphraser_type}")


class Paraphraser:
    """Base class for paraphrasers. Use Paraphraser.build(config) to create an instance of a specific paraphraser type."""

    @classmethod
    def build(cls, config: ParaphraserConfig):
        if config.paraphraser_type == "minifier":
            return MinifierParaphraser()

        if config.paraphraser_type == "llm":
            return LLMParaphraser(
                model_provider=config.kwargs["model_provider"],  # type: ignore
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
                        model=solution.model,
                        solution=paraphrased_solution,
                        original_solution=solution.solution,
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

    def __init__(_self):
        pass

    def __call__(_self, text: str) -> str:
        """Does nothing."""
        return text


class MinifierParaphraser(Paraphraser):
    """A paraphraser that minifies code using the `code_minifier` package."""

    def __init__(_self):
        pass

    def _paraphrase_single_solution(_self, text: str) -> str:
        """Minify the given text string."""

        text = text.removeprefix("```")
        text = text.removeprefix("python")
        text = text.removesuffix("```")

        return python_minifier.minify(text)


class LLMParaphraser(Paraphraser):
    def __init__(self, model_provider: str, model_name: str):
        raise NotImplementedError("LLM paraphraser not implemented yet")
