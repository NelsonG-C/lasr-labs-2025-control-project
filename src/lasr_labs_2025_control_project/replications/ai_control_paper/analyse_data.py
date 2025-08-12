# %%
import pathlib

from lasr_labs_2025_control_project.utils import analysis

# %%
data_path = pathlib.Path(__file__).parent / "data" / "200_samples"

keystrings = {
    "honest_honeypot_malign": "honest_honeypot",
    "attack_honeypot_malign": "attack_honeypot",
    "honest_monitor_benign": "honest_monitor",
    "attack_monitor_malign": "attack_monitor",
}

analysis.analyse_protocol(data_path, **keystrings)

# %%
