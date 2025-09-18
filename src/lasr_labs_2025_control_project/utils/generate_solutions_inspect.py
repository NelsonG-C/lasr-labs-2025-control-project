"""
Generates solutions for the APPS dataset. Applies the control arena APPS scorers without actually using the control arena setup.
Put your prompt into the solver's system message.
Select the APPS data split in AppsSetting.
Make sure you're using the correct model in the eval config.
Just run the script.
"""

import json
import re
from typing import Callable

from control_arena.control_task_metadata import ControlTaskMetadata
from control_arena.eval_mode import EvalMode
from control_arena.tools.submit_tool import SubmitToolConfig, get_submit_tool_def
from inspect_ai import Epochs, Task, eval, task
from inspect_ai.model import (
    ChatMessageAssistant,
    ModelOutput,
)
from inspect_ai.model._call_tools import tools_info
from inspect_ai.solver import (
    Generate,
    TaskState,
    generate,
    solver,
    system_message,
)
from inspect_ai.tool import ToolCall

from lasr_labs_2025_control_project.apps.prompts import self_rec_attack_policy_prompt
from lasr_labs_2025_control_project.apps.settings.setting import AppsSetting


@solver
def parse_and_submit(
    get_submit_tool_config: Callable[[ControlTaskMetadata], SubmitToolConfig],
    eval_mode: EvalMode,
):
    async def solve(
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        """Parse model output and create submit tool call."""
        # Get the model's output
        output = state.output.completion

        # Parse final_code and secret_input from the output
        final_code_match = re.search(
            r"<final_code>(.*?)</final_code>", output, re.DOTALL
        )
        secret_input_match = re.search(
            r"<secret_input>(.*?)</secret_input>", output, re.DOTALL
        )

        if final_code_match:
            code = final_code_match.group(1).strip()
            code = code.replace("```python", "").replace("```", "").strip()
            if eval_mode == EvalMode.ATTACK:
                try:
                    secret_input = secret_input_match.group(1).strip()  # pyright: ignore[reportOptionalMemberAccess]
                except Exception:
                    secret_input = ""
            else:
                secret_input = ""

            task_metadata = state.metadata_as(ControlTaskMetadata)
            submit_tool_config = get_submit_tool_config(task_metadata)
            submit_tool = get_submit_tool_def(submit_tool_config).as_tool()
            state.metadata["tools"] = tools_info(state.tools + [submit_tool])

            # Create submit tool call using the correct pattern
            submit_call = ToolCall(
                id="manual_submit_001",
                function="submit",
                arguments={"code": code, "secret_input": secret_input}
                if eval_mode == EvalMode.ATTACK
                else {"code": code},
            )

            # Create assistant message with tool call
            assistant_message = ChatMessageAssistant(
                content="",
                tool_calls=[submit_call],
            )

            state.output = ModelOutput.for_tool_call(
                model="manual_submit",
                tool_name="submit",
                tool_arguments={"code": code, "secret_input": secret_input}
                if eval_mode == EvalMode.ATTACK
                else {"code": code},
            )
            state.output.completion = json.dumps(
                {"code": code, "secret_input": secret_input}
                if eval_mode == EvalMode.ATTACK
                else {"code": code}
            )

            # Add the assistant message to the state
            state.messages.append(assistant_message)

        return state

    return solve


@task
def backdoored_solutions(eval_mode: EvalMode) -> Task:
    """Task that runs AppsSetting dataset through attack policy system prompt."""

    # Create AppsSetting instance
    apps_setting = AppsSetting(
        dataset_name="simple",
        cache_python_code=False,
        num_tests_to_run=10,
        dataset_type="filtered",
        original_dataset_split="test",
        filter_on_split="even_test",
        shuffle=False,
        problem_ids_range=[0, 2999],
    )

    # Get dataset in attack mode

    dataset = apps_setting.get_dataset(eval_mode=eval_mode)

    # Create task with attack policy system prompt
    return Task(
        dataset=dataset,
        solver=[
            # apps_setting.setup_environment(),  # pyright: ignore[reportArgumentType]
            system_message(self_rec_attack_policy_prompt),
            generate(),
            parse_and_submit(apps_setting.get_submit_tool_config, eval_mode),
        ],
        scorer=apps_setting.get_scorers()[0],  # Use first scorer
        sandbox=apps_setting.get_sandbox_config(),
    )


# Run evaluation
if __name__ == "__main__":
    result = eval(
        tasks=backdoored_solutions(EvalMode.ATTACK),
        model="openai/gpt-4o",
        epochs=Epochs(1, ["pass_at_1"]),
        max_subprocesses=16,
        max_connections=16,
    )
