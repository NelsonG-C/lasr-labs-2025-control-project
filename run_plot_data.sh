#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") <config_directory>" >&2
  exit 1
fi

config_dir="$1"
if [[ ! -d "$config_dir" ]]; then
  echo "Error: '$config_dir' is not a directory" >&2
  exit 1
fi

readarray -d '' -t cfg_files < <(find "$config_dir" -type f -name '*.yaml' -print0 | sort -z)

if [[ ${#cfg_files[@]} -eq 0 ]]; then
  echo "No .yaml files found in '$config_dir'." >&2
  exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

uv_bin="${UV_BIN:-uv}"
if ! command -v "$uv_bin" >/dev/null 2>&1; then
  echo "Error: '$uv_bin' command not found. Install uv or set UV_BIN to the executable path." >&2
  exit 1
fi

"$uv_bin" run python -m lasr_labs_2025_control_project.scripts.plot_data -cfg "${cfg_files[@]}"
