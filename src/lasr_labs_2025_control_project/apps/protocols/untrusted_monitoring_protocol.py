from control_arena.agent_builder import (  # type: ignore
    AgentBuilder,
    AgentBuilderType,
    control_agent,
)
from control_arena.control_task_metadata import ControlTaskMetadata  # type: ignore
from inspect_ai.agent import Agent, AgentState  # type: ignore
from inspect_ai.tool import Tool  # type: ignore
from pydantic import Field  # type: ignore

from lasr_labs_2025_control_project.apps.monitors.basic_monitor import basic_monitor
from lasr_labs_2025_control_project.apps.policies.policy_gate import policy_gate
from typing import Any


@control_agent  # type: ignore
def untrusted_monitoring_protocol(
    policy_builder: Any = Field(default_factory=policy_gate),  # AgentBuilder
    monitor_builder: Any = Field(default_factory=basic_monitor),  # AgentBuilder
) -> Any:  # AgentBuilderType
    """Build an untrusted monitor agent that combines policy and monitor.

    Args:
        policy_builder: The builder for the policy agent.
        monitor_builder: The builder for the monitor agent.

    Returns:
        An agent that combines the policy and monitor.
    """

    def untrusted_monitor(task_metadata: Any) -> Any:  # ControlTaskMetadata -> Agent
        """A protocol that applies monitoring to policy outputs.

        Args:
            task_metadata: ControlTaskMetadata containing specific information about the current task such as main and side task instructions.
            tools: list of Tool objects that will be available to the agent.
        """
        policy = policy_builder(task_metadata)
        monitor = monitor_builder(task_metadata)

        async def impl(state: Any, tools: list[Any]) -> Any:  # AgentState, list[Tool] -> AgentState
            state = await policy(state, tools)
            # Check if there are tool calls and specifically if "submit" is among them
            if state.output.message and state.output.message.tool_calls:
                state = await monitor(state, tools)
            return state

        return impl

    return untrusted_monitor
