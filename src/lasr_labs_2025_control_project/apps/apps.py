from typing import Any

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
    attack_self_rec_collusion_strategy,
    backdoor_injection_prompt,
    honest_monitor_prompt,
    honest_policy_prompt,
    monitor_self_rec_collusion_strategy,
    self_rec_attack_monitor_prompt,
    self_rec_attack_policy_prompt,
    self_rec_honest_policy_prompt,
)
from lasr_labs_2025_control_project.apps.utils.policy_gate import (
    policy_gate,
)
from lasr_labs_2025_control_project.utils.setting import (
    AppsSetting,
)

# definitions of the prompts we use
# Assign the prompts from the file here to pass into their respective builders
attack_policy_prompt = self_rec_attack_policy_prompt
attack_monitor_prompt = self_rec_attack_monitor_prompt
collusion_strategy_prompt = monitor_self_rec_collusion_strategy

# human honeypot prompt - base human solution prompting to add backdoor
human_honeypot_prompt = backdoor_injection_prompt

# Setting for the experiments
# Use train set for quick iteration, switch to validation or test for final runs
apps_setting = AppsSetting(
    dataset_name="simple",
    cache_python_code=False,
    num_tests_to_run=10,
    dataset_type="filtered",
    split="test",
    shuffle=False,
    problem_ids_range=[4000, 4999],
)


# Policies
def untrusted_policy_builder(model: str) -> Any:
    return policy_gate(
        honest=honest_policy(model=model, prompt=self_rec_honest_policy_prompt),
        attack=attack_policy(
            model=model,
            prompt=attack_policy_prompt,
            collusion_strategy=attack_self_rec_collusion_strategy,
            ephemeral_reminder=False,
        ),
    )


def honeypot_policy_builder(model: str) -> Any:
    return policy_gate(
        honest=honest_policy(model=model, prompt=honest_policy_prompt),
        attack=attack_policy(
            model=model,
            prompt=attack_policy_prompt,
            collusion_strategy=attack_self_rec_collusion_strategy,
            ephemeral_reminder=False,
        ),
    )


def human_honeypot_policy_builder(model: str) -> Any:
    return policy_gate(
        honest=honest_policy(model=model, prompt=honest_policy_prompt),
        attack=attack_policy(
            model=model,
            prompt=human_honeypot_prompt,
            collusion_strategy=attack_self_rec_collusion_strategy,
            ephemeral_reminder=False,
        ),
    )


# Monitors
def untrusted_monitor_builder(model: str, score_field: str) -> Any:
    return policy_gate(
        honest=basic_monitor(
            model=model, prompt=honest_monitor_prompt, score_field=score_field
        ),
        attack=basic_monitor(
            model=model,
            prompt=attack_monitor_prompt,
            collusion_strategy=collusion_strategy_prompt,
            score_field=score_field,
        ),
    )
