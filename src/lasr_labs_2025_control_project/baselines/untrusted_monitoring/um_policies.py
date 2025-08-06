from control_arena.agent_builder import AgentBuilderType, control_agent
from control_arena.control_task_metadata import ControlTaskMetadata
from control_arena.protocol_arg_defaults import (
    get_untrusted_model,
)
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)
from inspect_ai.tool import Tool
from pydantic import Field

from .prompts import honest_system_prompt, honest_user_prompt


@control_agent
def prompted_policy(
    model: str = Field(default_factory=get_untrusted_model),
    system_prompt: str = Field(default=honest_system_prompt),
    user_prompt: str = Field(default=honest_user_prompt),
    is_trusted: bool = Field(default=False),
) -> AgentBuilderType:
    """This policy generates a response, which must include a tool call, based on a provided system and user prompt. If no tool call is present the model will be reminded to respond with a tool call.

    We do the following:
    1. Get the model using the provided model name.
    2. Define an inner function `impl` that takes the current state and a list of tools.
    3. Create system message and user messages using the provided prompts, filling in task metadata.

    Args:
        model (str): The model to use for the policy.
        system_prompt (str): The system prompt to use for the policy.
        user_prompt (str): The user prompt to use for the policy.
        is_trusted (bool): Whether this policy uses a trusted model. If True, don't add messages to state._messages.
    """
    _model = get_model(model)

    def build_agent(task_metadata: ControlTaskMetadata) -> Agent:
        async def impl(state: AgentState, tools: list[Tool]) -> AgentState:
            system_message = ChatMessageSystem(
                content=system_prompt.format(
                    main_task_description=task_metadata.main_task_description,
                    side_task_description=task_metadata.side_task_description,
                    setting_description=task_metadata.setting_description,
                )
            )

            user_message = ChatMessageUser(
                content=user_prompt.format(
                    main_task_description=task_metadata.main_task_description,
                    side_task_description=task_metadata.side_task_description,
                    setting_description=task_metadata.setting_description,
                )
            )

            chat: list[ChatMessage] = [
                system_message,
                user_message,
                *state._messages,
            ]

            # Try three times to get a response with tool calls
            # If it fails, it will raise an error
            for _ in range(3):
                # get the response from the policy model
                res = await _model.generate(
                    input=chat,
                    tools=tools,
                    cache=True,
                )
                # append the result to the full chat message history
                chat.append(res.message)
                print("Generated response:", res)
                # check if the response contains tool calls,
                # which it needs for stopping, and also for the monitor to work
                if res.message.tool_calls:
                    print("Found tool calls in response")
                    state.output = res
                    # Only add messages to state._messages for untrusted models
                    if not is_trusted:
                        state._messages += (res.message,)
                    return state
                else:
                    print("No tool calls, retrying...")
                    retry_message = ChatMessageUser(
                        content="You didn't use any of the tools. Please try again making sure to use at least one tool."
                    )
                    chat.append(retry_message)
                    state._messages += (retry_message,)

            raise ValueError("No tool calls found, exiting...")

        return impl

    return build_agent
