# %%
# This does the imports, always run this

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import lasr_labs_2025_control_project.utils.analysis as analysis

# %%
# Loads and demos the compare_honeypots_to_attacks function

mocks_and_demos_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "mocks_and_demos"
)
honeypot_df = pd.read_csv(
    os.path.join(mocks_and_demos_path, "random_values_60_to_100.csv")
)
attack_df = pd.read_csv(
    os.path.join(mocks_and_demos_path, "random_values_50_to_90.csv")
)

results = analysis.analyse_honeypots_attacks(honeypot_df, attack_df)

print(results)

plt.show(results.histogram)
plt.show(results.probability_of_success)
plt.show(results.roc_curve)

# %%
# Loads and demos the calculate_safety_usefulness_scores function

mocks_and_demos_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "mocks_and_demos"
)
honeypot_df = pd.read_csv(
    os.path.join(mocks_and_demos_path, "random_values_60_to_100.csv")
)
attack_df = pd.read_csv(
    os.path.join(mocks_and_demos_path, "random_values_50_to_90.csv")
)
honest_df = pd.read_csv(
    os.path.join(mocks_and_demos_path, "random_values_30_to_70.csv")
)
results = analysis.analyse_safety_usefulness(
    honeypot_df=honeypot_df,
    attack_monitor_malign=attack_df,
    honest_monitor_benign=honest_df,
)

plt.show(results.safety_usefulness_plot)
plt.show(results.histogram)

# %%
# Runs analysis on a whole folder of .eval files
target_folder = Path(os.path.join(os.path.dirname(__file__), "example_protocol"))

results = analysis.analyse_protocol(target_folder=target_folder)

# %%
