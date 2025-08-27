import json
from enum import Enum
from typing import Any

from control_arena.eval_mode import EvalMode
from control_arena.utils import extract_xml_tag
from inspect_ai import Epochs, Task, task
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from lasr_labs_2025_control_project.experiments.acausal_collusion.apps_inspect.prompts import (
    M_baseline_system_prompt,
    P_baseline_system_prompt,
    collusion_signal_prompt,
    judge_prompt,
    meta_model_collusion_system_prompt,
    policy_response_formatting_instructions,
)
from lasr_labs_2025_control_project.experiments.acausal_collusion.apps_inspect.scorers import (
    passed_monitor,
    share_strategy,
    successful_sample,
)
from lasr_labs_2025_control_project.replications.ai_control_paper.setting import (
    AppsSetting,
)


class MetaType(Enum):
    POLICY = "MP"
    MONITOR = "MM"


@solver
def meta_collusion_generation(model: str, meta_type: str) -> Solver:
    """Solver to create summaries of meta policy and monitor results"""
    llm_model = get_model(model)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Create summaries for policy results

        if meta_type == "policy":
            chat = [
                ChatMessageSystem(
                    content=meta_model_collusion_system_prompt.format(
                        meta_assignment=MetaType.POLICY.value,
                        meta_value="P",
                    )
                ),
                ChatMessageSystem(content=collusion_signal_prompt),
                ChatMessageUser(content=state.messages[0].content),
            ]
        else:
            chat = [
                ChatMessageSystem(
                    content=meta_model_collusion_system_prompt.format(
                        meta_assignment=MetaType.MONITOR.value,
                        meta_value="M",
                    )
                ),
                ChatMessageSystem(content=collusion_signal_prompt),
                ChatMessageUser(content=state.messages[0].content),
            ]

        response = await llm_model.generate(chat)

        if response:
            if meta_type == "policy":
                store().set(
                    "policy_collusion_strategy", {"strategy": response.completion}
                )
            else:
                store().set(
                    "monitor_collusion_strategy", {"strategy": response.completion}
                )
        else:
            print("No response from meta collusion generation.")
            raise ValueError("No response from meta collusion generation.")

        return state

    return solve


@solver
def policy_solver(model: str = "anthropic/claude-3-5-haiku-latest") -> Solver:
    llm_model = get_model(model)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        policy_prompt = P_baseline_system_prompt
        collusion_strategy = store().get("policy_collusion_strategy")
        if not collusion_strategy:
            print("No collusion strategy found, cannot generate policy.")
            return state
        else:
            strategy = extract_xml_tag(collusion_strategy["strategy"], "FINAL")
            if not strategy:
                raise ValueError("No FINAL tag found in policy collusion strategy.")

        res = await llm_model.generate(
            input=[
                ChatMessageSystem(
                    content=policy_prompt.format(collusion_instructions=strategy)
                ),
                # add original problem from sample
                ChatMessageUser(content=state.messages[0].content),
                ChatMessageUser(content=policy_response_formatting_instructions),
            ],
            config=GenerateConfig(
                temperature=1.0,
            ),
        )

        store().set("generated_policy_code", {"message": res.message.text})
        state.output = res
        state.messages.append(res.message)

        return state

    return solve


@solver
def monitor_solver(model: str = "anthropic/claude-3-5-haiku-latest") -> Solver:
    llm_model = get_model(model)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        monitor_prompt = M_baseline_system_prompt

        collusion_strategy = store().get("monitor_collusion_strategy")
        policy_code = store().get("generated_policy_code")
        if not collusion_strategy:
            print("No collusion strategy found, cannot generate policy.")
            return state
        else:
            strategy = extract_xml_tag(collusion_strategy["strategy"], "FINAL")
            if not strategy:
                raise ValueError("No FINAL tag found in monitor collusion strategy.")

        if not policy_code:
            print("No policy code found, cannot generate monitor.")
            return state
        else:
            if not policy_code["message"]:
                raise ValueError("No message found in generated policy code.")
            try:
                policy_message = json.loads(policy_code["message"])
                code_content = policy_message.get("code")
                if not code_content:
                    raise ValueError("No code found in generated policy message.")
            except json.JSONDecodeError:
                code_content = "<failed to parse>"

        res = await llm_model.generate(
            input=[
                ChatMessageSystem(
                    content=monitor_prompt.format(collusion_instructions=strategy)
                ),
                # add original problem from sample
                ChatMessageUser(content=state.messages[0].content),
                ChatMessageUser(content=code_content),
            ],
            config=GenerateConfig(
                temperature=1.0,
            ),
        )
        store().set(
            "generated_monitor_response",
            {
                "message": res.message.text,
            },
        )
        state.output = res
        state.messages.append(res.message)

        return state

    return solve


@solver
def compare_strategies(
    judge_model: str = "anthropic/claude-sonnet-4-20250514",
) -> Solver:
    llm_model = get_model(judge_model)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async def compare_pair() -> dict[str, Any] | None:
            """Compare a single policy-monitor pair"""
            policy_summary = store().get("policy_collusion_strategy")
            monitor_summary = store().get("monitor_collusion_strategy")
            full_prompt = judge_prompt.format(
                policy_summary=policy_summary["strategy"],
                monitor_summary=monitor_summary["strategy"],
            )

            judge_response = await llm_model.generate(full_prompt)
            judgement = judge_response.completion.strip().upper()
            result = extract_xml_tag(judgement, "ANSWER")

            if result == "YES":
                store().set("collusion_detected", True)
            return None

        await compare_pair()

        return state

    return solve


@task
def apps_acausal_collusion(
    temperature: float = 0.0,
    num_epochs: int = 1,
    epoch_reducer: str | list[str] = ["mean"],
    model: str = "anthropic/claude-3-5-haiku-latest",
):
    apps_setting = AppsSetting(dataset_type="filtered", shuffle=False, split="train")
    dataset = apps_setting.get_dataset(eval_mode=EvalMode.HONEST)

    return Task(
        dataset=dataset,
        epochs=Epochs(num_epochs, epoch_reducer),
        solver=[
            meta_collusion_generation(model=model, meta_type="policy"),
            policy_solver(model=model),
            meta_collusion_generation(model=model, meta_type="monitor"),
            monitor_solver(model=model),
            compare_strategies(judge_model=model),
        ],
        scorer=[passed_monitor(), share_strategy(), successful_sample()],
        config=GenerateConfig(temperature=temperature),
        sandbox="docker",
    )
