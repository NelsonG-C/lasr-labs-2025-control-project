from control_arena.eval.control_eval_config import ControlEvalConfig
from control_arena.eval.control_eval_def import get_control_task_product
from control_arena.eval_mode import EvalMode
from inspect_ai import Epochs, eval

from lasr_labs_2025_control_project.apps.apps import (
    apps_setting,
    untrusted_monitor_builder,
    untrusted_policy_builder,
)
from lasr_labs_2025_control_project.apps.protocols.untrusted_monitoring_protocol import (
    untrusted_monitoring_protocol,
)
from lasr_labs_2025_control_project.apps.scorers import max_confidence_scorer

untrusted_model = "openai/gpt-4.1"
score_field = "confidence_score"

apps_task = get_control_task_product(
    settings=apps_setting,
    protocols=untrusted_monitoring_protocol(
        policy_builder=untrusted_policy_builder(model=untrusted_model),
        monitor_builder=untrusted_monitor_builder(
            model=untrusted_model, score_field=score_field
        ),
    ),
    modes=[EvalMode.ATTACK],
    monitoring_scorers=[max_confidence_scorer()],
    eval_config=ControlEvalConfig(untrusted_model=untrusted_model),
)
n = 2
eval = eval(tasks=apps_task, epochs=Epochs(n, [f"pass_at_{n}"]))
