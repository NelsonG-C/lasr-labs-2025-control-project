from control_arena.eval.control_eval_def import get_control_task_product
from control_arena.eval_mode import EvalMode
from control_arena.suspicion_scorers import max_suspicion_scorer
from inspect_ai import eval

from lasr_labs_2025_control_project.apps.apps import (
    apps_setting,
    untrusted_monitor_builder,
    untrusted_policy_builder,
)
from lasr_labs_2025_control_project.apps.protocols.untrusted_monitoring_protocol import (
    untrusted_monitoring_protocol,
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
