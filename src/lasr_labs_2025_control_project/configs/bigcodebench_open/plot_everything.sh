#!/bin/bash

# Script to run plot_data.py for all monitor configs recursively
# Usage: ./plot_everything.sh [base_folder]
# Example: ./plot_everything.sh src/lasr_labs_2025_control_project/configs/bigcodebench

# Use current working directory as root
ROOT_DIR="$(pwd)"

# Use provided base folder or default to bigcodebench configs
BASE_FOLDER="${1:-src/lasr_labs_2025_control_project/configs/bigcodebench}"

# Convert to absolute path if it's relative
if [[ "$BASE_FOLDER" != /* ]]; then
    BASE_FOLDER="$ROOT_DIR/$BASE_FOLDER"
fi

echo "Running plot_data.py for all monitor configs..."
echo "Root directory: $ROOT_DIR"
echo "Base folder: $BASE_FOLDER"
echo ""

# Check if base folder exists
if [ ! -d "$BASE_FOLDER" ]; then
    echo "Error: Base folder '$BASE_FOLDER' does not exist"
    exit 1
fi

# Change to root directory for running commands
cd "$ROOT_DIR"

# Find all .yaml files in subdirectories named "monitors" recursively
# Sort them for predictable ordering
monitor_configs=$(find "$BASE_FOLDER" -type d -name "monitors" -exec find {} -maxdepth 10 -type f -name "*.yaml" \; | sort)

# Count total configs
total=$(echo "$monitor_configs" | grep -c .)
echo "Found $total monitor config(s)"
echo ""

# Process each config
count=0
while IFS= read -r config; do
    if [ -n "$config" ]; then
        count=$((count + 1))
        # Get relative path from root for display
        rel_path="${config#$ROOT_DIR/}"
        echo "[$count/$total] Running: $rel_path"
        uv run src/lasr_labs_2025_control_project/scripts/plot_data.py --config "$config"
        echo ""
    fi
done <<< "$monitor_configs"

echo "=== All $count plotting jobs completed ==="
