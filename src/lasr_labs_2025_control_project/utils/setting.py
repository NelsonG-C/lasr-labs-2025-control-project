import json
from typing import Any, Callable, Dict, List, Literal

from control_arena.control_task_metadata import ControlTaskMetadata
from control_arena.eval_mode import EvalMode
from control_arena.settings.apps.task.dataset import (
    AppsTaskMetadata,
    get_backdoors_dataset,
)
from control_arena.settings.apps.task.scorers import (
    apps_scorer,
)
from control_arena.settings.apps.task.submission_types import (
    SubmitAttackArgs,
    SubmitHonestArgs,
)
from control_arena.settings.setting import Setting
from control_arena.tools.submit_tool import SubmitToolConfig
from inspect_ai.dataset import Dataset, MemoryDataset, Sample, hf_dataset
from inspect_ai.scorer import (
    Scorer,
)
from inspect_ai.solver import (
    TaskState,
)
from inspect_ai.tool import Tool
from inspect_ai.util import SandboxEnvironmentSpec
from pydantic import Field

from lasr_labs_2025_control_project.apps.prompts import (
    SETTING_DESCRIPTION,
    SIDE_TASK_DESCRIPTION,
)


class AppsSetting(Setting):
    cache_python_code: bool = Field(default=True)
    num_tests_to_run: int = Field(default=10)
    dataset_name: Literal["simple", "backdoor"] = Field(default="simple")
    dataset_type: Literal["filtered", "original"] = Field(default="original")
    split: Literal["train", "test", "all"] = Field(default="test")
    problem_ids_range: List[int] | None = None
    shuffle: bool = Field(default=True)

    @classmethod
    def get_name(cls) -> str:
        return "apps"

    def get_agent_tools(self, state: TaskState) -> list[Tool]:
        return []

    def get_submit_tool_config(
        self, task_metadata: ControlTaskMetadata
    ) -> SubmitToolConfig:
        """Get the configuration for the submission tool in this setting.

        Returns:
            A SubmitToolConfig object specifying the submission tool
        """
        match task_metadata.eval_mode:
            case EvalMode.ATTACK:
                return SubmitToolConfig(
                    name="submit",
                    description="Submit an answer for evaluation",
                    input_type=SubmitAttackArgs,
                )
            case EvalMode.HONEST:
                return SubmitToolConfig(
                    name="submit",
                    description="Submit the code for testing",
                    input_type=SubmitHonestArgs,
                )

    def get_dataset(
        self,
        eval_mode: EvalMode = EvalMode.HONEST,
    ) -> MemoryDataset | Dataset:
        if self.split == "all":
            dataset = self.get_all_data(eval_mode)
        else:
            dataset = hf_dataset(
                "codeparrot/apps",
                sample_fields=self.record_to_sample_factory(eval_mode),
                split=self.split,
                trust=True,
                shuffle=self.shuffle,
            )

        if self.dataset_name == "backdoor":
            backdoors = get_backdoors_dataset()
            problem_ids = {
                int(backdoor["problem_id"])  # type: ignore
                for backdoor in backdoors
                if "problem_id" in backdoor  # type: ignore
            }
            dataset = dataset.filter(lambda x: x.id in problem_ids)

        if self.problem_ids_range is not None and len(self.problem_ids_range) == 2:
            min_id, max_id = self.problem_ids_range[0], self.problem_ids_range[1]

            dataset = dataset.filter(
                lambda x: (
                    x.id is not None
                    and min_id is not None
                    and max_id is not None
                    and min_id
                    <= (int(x.id) if isinstance(x.id, str) else x.id)
                    <= max_id
                )
            )

        if self.dataset_type == "filtered":
            # Filtering the dataset to have standard format.
            # Dataset has already been filtered to have inputs and outputs for the main task, which are needed for the tests.
            dataset = dataset.filter(
                lambda x: x.input is not None
                and "Input" in x.input
                and "Output" in x.input
                and "Examples" in x.input
            )

        return dataset

    def get_all_data(self, mode: EvalMode) -> Dataset:
        dataset = []
        for split in ["train", "test"]:
            split_data = hf_dataset(
                "codeparrot/apps",
                sample_fields=self.record_to_sample_factory(mode=mode),
                split=split,
                trust=True,
            )
            dataset.extend(split_data)
        return MemoryDataset(dataset)

    def get_scorers(self) -> list[Scorer]:
        return [apps_scorer(self.cache_python_code, self.num_tests_to_run)]

    def get_sandbox_config(self) -> SandboxEnvironmentSpec:
        return SandboxEnvironmentSpec(
            type="docker",
        )

    def filter_out_no_input_output(self, record: Dict[str, Any]) -> bool:
        """
        Filter out records that do not have 'input_output' or 'solutions' fields.
        """
        return (
            bool(record.get("input_output"))
            and bool(record.get("solutions"))
            and bool(record.get("question"))
        )

    def record_to_sample_factory(
        self,
        mode: EvalMode,
    ) -> Callable[[Dict[str, Any]], Sample | list[Sample]]:
        def record_to_sample(record: Dict[str, Any]) -> Sample | list[Sample]:
            if record.get("problem_id") is None:
                return []

            if not self.filter_out_no_input_output(record):
                return []

            # Try to parse input_output - skip if it fails
            try:
                input_outputs = json.loads(record["input_output"])
                inputs = input_outputs["inputs"]
                outputs = input_outputs["outputs"]

                # Skip if inputs/outputs are empty or not lists of strings
                if not inputs or not outputs:
                    return []
                if not all(isinstance(i, str) for i in inputs) or not all(
                    isinstance(o, str) for o in outputs
                ):
                    return []

            except Exception:
                return []

            # Try to parse solutions - skip if it fails
            try:
                solutions = json.loads(record["solutions"])
                if not solutions:
                    return []
            except Exception:
                return []

            # Create metadata
            metadata = AppsTaskMetadata(
                inputs=inputs,
                outputs=outputs,
                solutions=solutions,
                main_task_description=record["question"],
                side_task_description=SIDE_TASK_DESCRIPTION
                if mode == EvalMode.ATTACK
                else "",
                setting_description=SETTING_DESCRIPTION,
                setting_name="apps",
                side_task_name="insert_backdoor" if mode == EvalMode.ATTACK else None,
                eval_mode=mode,
            )

            return Sample(
                input=record["question"],
                id=record["problem_id"],
                metadata=metadata.model_dump(),
            )

        return record_to_sample
