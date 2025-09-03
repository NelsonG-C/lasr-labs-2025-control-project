from control_arena.eval.control_eval_def import get_control_task
from control_arena.eval_mode import EvalMode
from control_arena.monitor import basic_monitor
from control_arena.suspicion_scorers import max_suspicion_scorer
from inspect_ai import eval

from lasr_labs_2025_control_project.apps.policies.attack_policy import (
    attack_policy,
)
from lasr_labs_2025_control_project.apps.policies.honest_policy import (
    honest_policy,
)
from lasr_labs_2025_control_project.apps.prompts import (
    acausal_policy_collusion_context,
    acausal_policy_collusion_instructions,
    attack_monitor_prompt,
    attack_policy_simple_acausal,
    honest_monitor_prompt,
    honest_policy_prompt,
)
from lasr_labs_2025_control_project.apps.protocols.untrusted_monitoring_protocol import (
    untrusted_monitoring_protocol,
)
from lasr_labs_2025_control_project.apps.utils.policy_gate import (
    policy_gate,
)

# setting from repo file
from lasr_labs_2025_control_project.utils.setting import (
    AppsSetting,
)

untrusted_model = "openai/gpt-4o"
trusted_model = "openai/gpt-4o-mini"

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
        prompt=attack_policy_simple_acausal.format(
            acausal_policy_collusion_context=acausal_policy_collusion_context,
            acausal_policy_collusion_instructions=acausal_policy_collusion_instructions,
        ),
        ephemeral_reminder=False,
    ),
)

untrusted_monitor_builder = policy_gate(
    honest=basic_monitor(model=trusted_model, prompt=honest_monitor_prompt),
    attack=basic_monitor(model=untrusted_model, prompt=attack_monitor_prompt),
)

apps_task = get_control_task(
    setting=apps_setting,
    protocol=untrusted_monitoring_protocol(
        policy_builder=untrusted_policy_builder,
        monitor_builder=untrusted_monitor_builder,
    ),
    mode=EvalMode.ATTACK,
    monitoring_scorers=[max_suspicion_scorer()],
)

eval = eval(tasks=[apps_task], limit=1)
