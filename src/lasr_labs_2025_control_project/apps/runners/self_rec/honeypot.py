from control_arena.eval.control_eval_config import ControlEvalConfig
from control_arena.eval.control_eval_def import get_control_task_product
from control_arena.eval_mode import EvalMode
from inspect_ai import eval

from lasr_labs_2025_control_project.apps.apps import (
    apps_setting,
    honeypot_policy_builder,
    untrusted_monitor_builder,
)
from lasr_labs_2025_control_project.apps.protocols.untrusted_monitoring_protocol import (
    untrusted_monitoring_protocol,
)
from lasr_labs_2025_control_project.apps.scorers import max_confidence_scorer

untrusted_model = "openai/gpt-4.1"
trusted_model = "openai/gpt-4o-mini"
score_field = "confidence_score"

apps_task = get_control_task_product(
    settings=apps_setting,
    protocols=untrusted_monitoring_protocol(
        policy_builder=honeypot_policy_builder(model=trusted_model),
        monitor_builder=untrusted_monitor_builder(
            model=untrusted_model, score_field=score_field
        ),
    ),
    modes=[EvalMode.ATTACK],
    monitoring_scorers=[max_confidence_scorer()],
    eval_config=ControlEvalConfig(
        untrusted_model=untrusted_model, trusted_model=trusted_model
    ),
)

eval = eval(tasks=apps_task, limit=10)
