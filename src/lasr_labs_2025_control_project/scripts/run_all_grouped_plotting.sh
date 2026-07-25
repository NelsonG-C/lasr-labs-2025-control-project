#!/bin/bash

# Shell script to run plot_data.py commands for apps and bigcodebench datasets
# Exit on error
set -e

echo "Starting plot generation for apps and bigcodebench datasets..."
echo "================================================================"

# Function to run commands for a given dataset
run_dataset_commands() {
    local dataset=$1
    echo ""
    echo "Processing dataset: $dataset"
    echo "----------------------------------------"

    # Self-rec no paraphrasing: Base Model(backdoors), Control SFT (Randomised Labels, No backdoors), SFT (backdoors)

    echo "Running Self-rec no mitigations - UM Only..."
    if [ "$dataset" == "apps" ]; then
        uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
            --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/base_monitor_config.yaml \
                     src/lasr_labs_2025_control_project/configs/$dataset/control_self_rec/monitors/no_backdoors/base_monitor_config_randomised.yaml \
                     src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/base_monitor_config.yaml \
            --grouped-path no_mitigation \
            --disable-trusted-monitor True
    else
        uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
            --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/base_monitor_config.yaml \
                     src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/base_monitor_config.yaml \
            --grouped-path no_mitigation \
            --disable-trusted-monitor True
    fi

    echo "Running Self-rec no mitigations - All Monitors..."
    if [ "$dataset" == "apps" ]; then
        uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
            --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/base_monitor_config.yaml \
                     src/lasr_labs_2025_control_project/configs/$dataset/control_self_rec/monitors/no_backdoors/base_monitor_config_randomised.yaml \
                     src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/base_monitor_config.yaml \
            --grouped-path no_mitigation
    else
        uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
            --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/base_monitor_config.yaml \
                     src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/base_monitor_config.yaml \
            --grouped-path no_mitigation
    fi

    # Self-rec paraphrasing: None Minifier, SFT Minifier, None Paraphraser, SFT Paraphraser

    echo "Running Self-rec mitigations - UM Only..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/paraphrased_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/paraphrased_monitor_config.yaml \
        --grouped-path self_rec_mitigations \
        --disable-trusted-monitor True

    echo "Running Self-rec mitigations - All Monitors..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/paraphrased_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/paraphrased_monitor_config.yaml \
        --grouped-path self_rec_mitigations

    # Self Rec without SFT

    echo "Running Self Rec without SFT - UM Only..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/base_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/paraphrased_monitor_config.yaml \
        --grouped-path self_rec_none \
        --disable-trusted-monitor True

    echo "Running Self Rec without SFT - All Monitors..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/base_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/base/paraphrased_monitor_config.yaml \
        --grouped-path self_rec_none

    # Self Rec with SFT

    echo "Running Self Rec with SFT - UM Only..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/base_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/paraphrased_monitor_config.yaml \
        --grouped-path self_rec_SFT \
        --disable-trusted-monitor True

    echo "Running Self Rec with SFT - All Monitors..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/base_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/self_rec/monitors/sft/paraphrased_monitor_config.yaml \
        --grouped-path self_rec_SFT

    # Acausal + all paraphrasers

    echo "Running Acausal + all mitigations - UM Only..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/acausal/monitors/base_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/acausal/monitors/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/acausal/monitors/paraphrased_monitor_config.yaml \
        --grouped-path acausal_mitigations \
        --disable-trusted-monitor True

    echo "Running Acausal + all paraphrasers - All Monitors..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/acausal/monitors/base_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/acausal/monitors/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/acausal/monitors/paraphrased_monitor_config.yaml \
        --grouped-path acausal_mitigations

    # Causal + all paraphrasers

    echo "Running Causal + all mitigations - UM Only..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/causal/monitors/base_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/causal/monitors/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/causal/monitors/paraphrased_monitor_config.yaml \
        --grouped-path causal_mitigations \
        --disable-trusted-monitor True

    echo "Running Causal + all mitigations - All Monitors..."
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
        --config src/lasr_labs_2025_control_project/configs/$dataset/causal/monitors/base_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/causal/monitors/minified_monitor_config.yaml \
                 src/lasr_labs_2025_control_project/configs/$dataset/causal/monitors/paraphrased_monitor_config.yaml \
        --grouped-path causal_mitigations

    echo "Produce Key Figure Plot"
    uv run src/lasr_labs_2025_control_project/scripts/plot_data.py \
  --config \
    src/lasr_labs_2025_control_project/configs/apps/causal/monitors/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/minified_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/sft/paraphrased_monitor_config.yaml \
    src/lasr_labs_2025_control_project/configs/apps/self_rec/monitors/base/base_monitor_config_tm_only.yaml \
  --grouped-path key_figure

    echo "Completed processing for dataset: $dataset"
}

# Run commands for apps dataset
run_dataset_commands "apps"

# Run commands for bigcodebench dataset
run_dataset_commands "bigcodebench"

echo ""
echo "================================================================"
echo "All plot generation commands completed successfully!"
