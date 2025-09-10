from typing import Any

from control_arena.agent_builder import AgentBuilderType, control_agent  # type: ignore
from control_arena.control_task_metadata import ControlTaskMetadata  # type: ignore
from control_arena.protocol_arg_defaults import get_untrusted_model  # type: ignore
from inspect_ai.agent import Agent, AgentState  # type: ignore
from inspect_ai.model import (  # type: ignore
    ChatMessageSystem,
    GenerateConfig,
    get_model,
)
from inspect_ai.tool import Tool  # type: ignore
from pydantic import Field  # type: ignore

from lasr_labs_2025_control_project.apps.prompts import (
    honest_policy_prompt,
)


@control_agent  # type: ignore
def honest_policy(
    model: str = Field(default_factory=get_untrusted_model),
    config: Any = Field(default_factory=GenerateConfig),  # GenerateConfig
    prompt: str = Field(default=honest_policy_prompt),
    cache: bool = False,
) -> Any:  # AgentBuilderType
    llm_model = get_model(model, config=config)

    def honest(task_metadata: Any) -> Any:  # ControlTaskMetadata -> Agent # type: ignore
        """Honest policy for Untrusted Monitoring.

        Args:
            task_metadata: The task metadata containing descriptions
        """

        async def impl(state: Any, tools: list[Any]) -> Any:  # AgentState, list[Tool] -> AgentState # type: ignore
            # Format the system prompt
            system_message = ChatMessageSystem(
                content=prompt.format(
                    side_task_description=task_metadata.side_task_description,
                    setting_description=task_metadata.setting_description,
                )
            )

            # Set up messages if this is the first call
            messages = [system_message, *state._messages[1:]]

            state.output = await llm_model.generate(
                input=messages,
                tools=tools,
                cache=cache,
            )
            state._messages.append(state.output.message)
            return state

        return impl

    return honest
