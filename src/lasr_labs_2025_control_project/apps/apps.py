from control_arena.eval.control_eval_def import get_control_task_product
from control_arena.eval_mode import EvalMode
from control_arena.suspicion_scorers import max_suspicion_scorer
from inspect_ai import eval

from lasr_labs_2025_control_project.apps.monitors.basic_monitor import (
    basic_monitor,
)
from lasr_labs_2025_control_project.apps.policies.attack_policy import (
    attack_policy,
)
from lasr_labs_2025_control_project.apps.policies.honest_policy import (
    honest_policy,
)
from lasr_labs_2025_control_project.apps.prompts import (
    collusion_strategy,
    honest_monitor_prompt,
    honest_policy_prompt,
    monitor_self_rec_collusion_strategy,
    self_rec_attack_monitor_prompt,
    self_rec_attack_policy_prompt,
)
from lasr_labs_2025_control_project.apps.protocols.untrusted_monitoring_protocol import (
    untrusted_monitoring_protocol,
)
from lasr_labs_2025_control_project.apps.utils.policy_gate import (
    policy_gate,
)
from lasr_labs_2025_control_project.utils.setting import (
    AppsSetting,
)

untrusted_model = "openai/gpt-4o"
attack_policy_prompt = self_rec_attack_policy_prompt
attack_monitor_prompt = self_rec_attack_monitor_prompt

apps_setting = AppsSetting(
    dataset_name="simple",
    cache_python_code=True,
    num_tests_to_run=10,
    dataset_type="filtered",
    split="test",
    shuffle=False,
)

untrusted_policy_builder = policy_gate(
    honest=honest_policy(model=untrusted_model, prompt=honest_policy_prompt),
    attack=attack_policy(
        model=untrusted_model,
        prompt=attack_policy_prompt,
        collusion_strategy=collusion_strategy,
        ephemeral_reminder=False,
    ),
)

untrusted_monitor_builder = policy_gate(
    honest=basic_monitor(model=untrusted_model, prompt=honest_monitor_prompt),
    attack=basic_monitor(
        model=untrusted_model,
        prompt=attack_monitor_prompt,
        collusion_strategy=monitor_self_rec_collusion_strategy,
    ),
)

apps_task = get_control_task_product(
    settings=apps_setting,
    protocols=untrusted_monitoring_protocol(
        policy_builder=untrusted_policy_builder,
        monitor_builder=untrusted_monitor_builder,
    ),
    modes=[EvalMode.ATTACK],
    monitoring_scorers=[max_suspicion_scorer()],
)

eval = eval(tasks=apps_task, limit=1)
