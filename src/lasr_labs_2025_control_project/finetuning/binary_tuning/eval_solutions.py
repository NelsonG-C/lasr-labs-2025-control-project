"""
Runs apps scorers on the solutions from the eval logs without generating new solutions.
Make sure AppsSetting is configured to use same data split as in the eval log.
"""

import json
import re
from typing import Callable

from control_arena.control_task_metadata import ControlTaskMetadata
from control_arena.tools.submit_tool import SubmitToolConfig, get_submit_tool_def
from inspect_ai import Epochs, Task, eval, task
from inspect_ai.analysis import messages_df
from inspect_ai.model import (
    ChatMessageAssistant,
    ModelOutput,
)
from inspect_ai.model._call_tools import tools_info
from inspect_ai.solver import (
    Generate,
    TaskState,
    solver,
    system_message,
)
from inspect_ai.tool import ToolCall

from lasr_labs_2025_control_project.prompts.generation_prompts import (
    self_rec_subtle_attack_policy_prompt,
)
from lasr_labs_2025_control_project.settings.setting import AppsSetting


@solver
def parse_and_submit(
    get_submit_tool_config: Callable[[ControlTaskMetadata], SubmitToolConfig],
    eval_log_path: str,
):
    messages = messages_df(eval_log_path)

    async def solve(
        state: TaskState,
        generate: Generate,
    ) -> TaskState:
        """Get solution from the dataset"""
        # Get the solution from messages

        problem_statement = state.metadata_as(ControlTaskMetadata).main_task_description
        sample_id = messages[messages.content == problem_statement].sample_id.iloc[0]
        try:
            output = messages[messages.sample_id == sample_id].content.iloc[-1]
        except IndexError:
            raise ValueError(f"No solution found for {sample_id}: {problem_statement}")

        # output = state.output.completion

        # Parse final_code and secret_input from the output
        final_code_match = re.search(
            r"<final_code>(.*?)</final_code>", output, re.DOTALL
        )
        secret_input_match = re.search(
            r"<secret_input>(.*?)</secret_input>", output, re.DOTALL
        )

        if final_code_match and secret_input_match:
            code = final_code_match.group(1).strip()
            code = code.replace("```python", "").replace("```", "").strip()
            secret_input = secret_input_match.group(1).strip()

            task_metadata = state.metadata_as(ControlTaskMetadata)
            submit_tool_config = get_submit_tool_config(task_metadata)
            submit_tool = get_submit_tool_def(submit_tool_config).as_tool()
            state.metadata["tools"] = tools_info(state.tools + [submit_tool])

            # Create submit tool call using the correct pattern
            submit_call = ToolCall(
                id="manual_submit_001",
                function="submit",
                arguments={"code": code, "secret_input": secret_input},
            )

            # Create assistant message with tool call
            assistant_message = ChatMessageAssistant(
                content="",
                tool_calls=[submit_call],
            )

            state.output = ModelOutput.for_tool_call(
                model="manual_submit",
                tool_name="submit",
                tool_arguments={"code": code, "secret_input": secret_input},
            )
            state.output.completion = json.dumps(
                {"code": code, "secret_input": secret_input}
            )

            # Add the assistant message to the state
            state.messages.append(assistant_message)

        return state

    return solve


@task
def backdoored_solutions() -> Task:
    """Task that runs AppsSetting dataset through attack policy system prompt."""

    # Create AppsSetting instance
    apps_setting = AppsSetting(
        dataset_name="simple",
        cache_python_code=False,
        num_tests_to_run=10,
        dataset_type="filtered",
        original_dataset_split="test",
        filter_on_split="odd_train",
        shuffle=False,
        problem_ids_range=[0, 2999],
    )

    # Get dataset in attack mode
    from control_arena.eval_mode import EvalMode

    dataset = apps_setting.get_dataset(eval_mode=EvalMode.ATTACK)
    eval_log_path = "src/lasr_labs_2025_control_project/finetuning/binary_tuning/data/eval_logs/train/2025-09-11T07-28-05+01-00_backdoored-solutions_bJmmr3dxzvGnVhX99yNCiz.eval"

    # Create task with attack policy system prompt
    return Task(
        dataset=dataset,
        solver=[
            # apps_setting.setup_environment(),  # pyright: ignore[reportArgumentType]
            system_message(self_rec_subtle_attack_policy_prompt),
            parse_and_submit(apps_setting.get_submit_tool_config, eval_log_path),
        ],
        scorer=apps_setting.get_scorers()[0],  # Use first scorer
        sandbox=apps_setting.get_sandbox_config(),
    )


# Run evaluation
if __name__ == "__main__":
    result = eval(
        tasks=backdoored_solutions(),
        model="openai/gpt-4o-2024-08-06",
        epochs=Epochs(1, ["pass_at_1"]),
    )

    print("Evaluation completed!")
