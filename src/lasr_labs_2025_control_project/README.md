# Untrusted Monitoring Project

## Setup

### Download Data
First, download the APPS and BigCodeBench data from the Google Drive FINAL_RESULTS folder.

**⚠️ Important:** Do NOT update these data folders in Drive unless discussing with the whole team, otherwise people will end up working with different data.

### Output Location
All generated plot results should be uploaded to the `plot_results` subfolder.

---

## Configuration

### Config Files
All plotting configs are located in `src/lasr_labs_2025_control_project/configs`

**⚠️ Important:** These configs should never change unless a bug is found. If you find a bug, notify the team before committing.

### Available Parameters
The only parameter you may want to change:
- `disabled_trusted_monitor`: Set to `True` to disable (default is `False`)
  - **⚠️ Warning:** Be careful about overriding results—file paths will be the same

---

## Generating Plots

### Option 1: Run All Grouped Configs
Generate plots for all specified group configs across both datasets:
```bash
src/lasr_labs_2025_control_project/scripts/run_all_grouped_plotting.sh
```

By default, this generates plots for both APPS and BigCodeBench datasets.

For custom groupings, use the commands in the "Plotting Multiple Results" section with:
- `--config`: Specify which YAML files to use
- `--grouped-path`: Specify where to save results (⚠️ if not specified, uses path from final config file)
- `--disable-trusted-monitor`: Set to `True` to disable

### Option 2: Run All Configs for Single Dataset
Plot all monitor configs recursively for a single dataset:
```bash
./src/lasr_labs_2025_control_project/scripts/plot_everything.sh [base_folder]
```

**Example:**
```bash
./plot_everything.sh src/lasr_labs_2025_control_project/configs/bigcodebench
```

### Option 3: Run Single Config
Generate plots for a specific config:
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/base_monitor_config.yaml
```

For parameter sensitivity analysis:
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_parameter_sensitivity.py \
  --config src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/base_monitor_config.yaml
```

If you want to disable the trusted monitor:
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_parameter_sensitivity.py \
  --config src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/base_monitor_config.yaml \
  --disable-trusted-monitor True
```

### Option 4: Plotting Multiple Results
Combine results from multiple configs:
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/base_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/minified_monitor_config.yaml
```

**⚠️ Note:** Results will be saved to the file path of the **last config**. In the example above, base_minified results will be overridden.

---

## Grouped Plotting Commands

### No Mitigation
Compares Base Model (backdoors), Control SFT (randomized labels, no backdoors), and SFT (backdoors):

**Um Only:**
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/base_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/control_self_rec/monitors/no_backdoors/base_monitor_config_randomised.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/base_monitor_config.yaml \
  --grouped-path no_mitigation \
  --disable-trusted-monitor True
```

**All Monitors:**
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/base_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/control_self_rec/monitors/no_backdoors/base_monitor_config_randomised.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/base_monitor_config.yaml \
  --grouped-path no_mitigation
```

### Self-Recognition Mitigations
Compares None Minifier, SFT Minifier, None Paraphraser, and SFT Paraphraser:

**Um Only:**
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/paraphrased_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/paraphrased_monitor_config.yaml \
  --grouped-path self_rec_mitigations \
  --disable-trusted-monitor True
```

**All Monitors:**
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/paraphrased_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/paraphrased_monitor_config.yaml \
  --grouped-path self_rec_mitigations
```

### Acausal Mitigations
Tests acausal approach with all mitigations:

**Um Only:**
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/acausal/monitors/base_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/acausal/monitors/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/acausal/monitors/paraphrased_monitor_config.yaml \
  --grouped-path acausal_mitigations \
  --disable-trusted-monitor True
```

**All Monitors:**
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/acausal/monitors/base_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/acausal/monitors/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/acausal/monitors/paraphrased_monitor_config.yaml \
  --grouped-path acausal_mitigations
```

### Causal Mitigations
Tests causal approach with all mitigations:

**Um Only:**
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/causal/monitors/base_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/causal/monitors/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/causal/monitors/paraphrased_monitor_config.yaml \
  --grouped-path causal_mitigations \
  --disable-trusted-monitor True
```

**All Monitors:**
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/causal/monitors/base_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/causal/monitors/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/causal/monitors/paraphrased_monitor_config.yaml \
  --grouped-path causal_mitigations
```


## Key Figures

### Main Figure
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/causal/monitors/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/paraphrased_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/base_monitor_config_tm_only.yaml \
  --grouped-path key_figure
```

### Self Rec without SFT
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/base_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/paraphrased_monitor_config.yaml \
  --grouped-path self_rec_none
```

### Self Rec with SFT
```bash
uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/base_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/paraphrased_monitor_config.yaml \
  --grouped-path self_rec_SFT
```

### Additive
TBC
