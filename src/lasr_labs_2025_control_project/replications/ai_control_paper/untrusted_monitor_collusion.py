from pathlib import Path

from control_arena.eval.control_eval_def import get_control_task
from control_arena.eval_mode import EvalMode
from control_arena.monitor import basic_monitor
from control_arena.policy import attack_policy, honest_policy, untrusted_policy
from control_arena.protocol import tm_protocol
from control_arena.suspicion_scorers import max_suspicion_scorer
from control_arena.utils import extract_prompts_from_dir

# setting from repo file
from lasr_labs_2025_control_project.replications.ai_control_paper.setting import (
    AppsSetting,
)

untrusted_model = "openai/o4-mini"
trusted_model = "openai/o3-mini"

prompts = extract_prompts_from_dir(Path(__file__).parent / "prompts")

collusion_strategy_text = prompts["for_while_always"]

# Pre-inject the collusion strategy into the improved prompts while leaving
# other placeholders (e.g., {setting_description}, {task_description}, {tool_calls})
# untouched for the framework to populate at runtime.
attack_policy_prompt = prompts["attack_generator_improved"].replace(
    "{collusion_strategy}", collusion_strategy_text
)
attack_monitor_prompt = prompts["attack_monitor_improved"].replace(
    "{collusion_strategy}", collusion_strategy_text
)


apps_setting = AppsSetting(
    dataset_name="simple",
    cache_python_code=True,
    num_tests_to_run=10,
    dataset_type="filtered",
    split="test",
    shuffle=True,
)

# Set up generator policies: honest, attack, and honeypot

honest_policy_builder = honest_policy(model=untrusted_model)
attack_policy_builder = attack_policy(
    model=untrusted_model, prompt=attack_policy_prompt
)
honeypot_policy_builder = attack_policy(
    model=trusted_model, prompt=prompts["honeypot_generation_improved"]
)

untrusted_policy_builder = untrusted_policy(
    honest=honest_policy_builder, attack=attack_policy_builder
)
honeypot_policy_builder = untrusted_policy(
    honest=honeypot_policy_builder, attack=honeypot_policy_builder
)

# Set up monitor

honest_monitor_builder = basic_monitor(model=trusted_model)
attack_monitor_builder = basic_monitor(
    model=untrusted_model, prompt=attack_monitor_prompt
)
untrusted_monitor_builder = untrusted_policy(
    honest=honest_monitor_builder, attack=attack_monitor_builder
)


# Build our monitoring protocols

monitored_protocol = tm_protocol(
    policy_builder=untrusted_policy_builder,
    monitor_builder=untrusted_monitor_builder,
)

honest_gen_with_attack_monitor_protocol = tm_protocol(
    policy_builder=untrusted_policy_builder,
    monitor_builder=attack_monitor_builder,
)

honeypot_protocol = tm_protocol(
    policy_builder=honeypot_policy_builder, monitor_builder=untrusted_monitor_builder
)

attack_task = get_control_task(
    setting=apps_setting,
    protocol=monitored_protocol,
    mode=EvalMode.HONEST,
    monitoring_scorers=[max_suspicion_scorer()],
)
