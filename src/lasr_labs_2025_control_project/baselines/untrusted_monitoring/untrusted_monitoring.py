from control_arena.eval.control_eval_def import get_control_task_product
from control_arena.eval_mode import EvalMode
from control_arena.settings.apps.apps_setting import AppsSetting
from inspect_ai import eval

from lasr_labs_2025_control_project.baselines.untrusted_monitoring.prompts import (
    attack_system_prompt,
    attack_user_prompt,
)
from lasr_labs_2025_control_project.baselines.untrusted_monitoring.um_monitor import (
    code_monitor,
)
from lasr_labs_2025_control_project.baselines.untrusted_monitoring.um_policies import (
    prompted_policy,
)
from lasr_labs_2025_control_project.baselines.untrusted_monitoring.um_protocols import (
    monitoring_protocol,
)

setting = AppsSetting(cache_python_code=False)
protocol = monitoring_protocol(
    policy_builder=prompted_policy(
        system_prompt=attack_system_prompt,
        user_prompt=attack_user_prompt,
    ),
    monitor_builder=code_monitor(),
)
eval_tasks = get_control_task_product(
    setting,
    protocol,
    [EvalMode.ATTACK],
)
eval(tasks=eval_tasks, limit=1)
