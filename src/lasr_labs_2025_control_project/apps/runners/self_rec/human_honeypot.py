from control_arena.eval.control_eval_config import ControlEvalConfig
from control_arena.eval.control_eval_def import get_control_task_product
from control_arena.eval_mode import EvalMode
from inspect_ai import Epochs, eval

from lasr_labs_2025_control_project.apps.apps import (
    human_honeypot_policy_builder,
    untrusted_monitor_builder,
)
from lasr_labs_2025_control_project.apps.protocols.untrusted_monitoring_protocol import (
    untrusted_monitoring_protocol,
)
from lasr_labs_2025_control_project.apps.scorers import max_confidence_scorer
from lasr_labs_2025_control_project.utils.human_solution_setting import AppsSetting

untrusted_model = "openai/gpt-4.1"
trusted_model = "openai/gpt-4o"
score_field = "confidence_score"

apps_setting = AppsSetting(
    dataset_name="simple",
    cache_python_code=False,
    num_tests_to_run=10,
    dataset_type="filtered",
    split="test",
    shuffle=False,
    problem_ids_range=[4000, 4999],
)

apps_task = get_control_task_product(
    settings=apps_setting,
    protocols=untrusted_monitoring_protocol(
        policy_builder=human_honeypot_policy_builder(model=trusted_model),
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

n = 2
eval = eval(tasks=apps_task, epochs=Epochs(n, [f"pass_at_{n}"]), limit=10)
