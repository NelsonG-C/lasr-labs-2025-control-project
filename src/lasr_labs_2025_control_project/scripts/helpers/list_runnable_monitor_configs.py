"""List monitor configs whose output data (attack/honest/honeypots triple) exists.

The generated config trees deliberately cover every cell of the experiment
cross-product, including arms that were never run or never published for a
given model class (e.g. the closed-model causal none/minified SFT arms, or the
vestigial `additive` folder). Analysis drivers should therefore enumerate
*data-backed* configs: this script prints, one per line,

    <condition_key>\t<config_path>

for every monitor config all three of whose save_paths exist on disk, where
condition_key is `<setting>/<experiment>/<arm>` relative to the model class.

    python .../list_runnable_monitor_configs.py --trees apps apps_open
"""

import argparse
from pathlib import Path

from lasr_labs_2025_control_project.utils.config_loader import (
    load_and_normalize_config,
)

CONFIGS_ROOT = Path("src/lasr_labs_2025_control_project/configs")


def runnable_configs(tree: str) -> list[tuple[str, Path]]:
    found = []
    for cfg_path in sorted((CONFIGS_ROOT / tree).rglob("*.yaml")):
        if "/monitors" not in str(cfg_path) and "additive" not in cfg_path.parts:
            continue
        try:
            normalized = load_and_normalize_config(cfg_path, config_type="monitor")
        except Exception:
            continue  # generator/transform configs living under odd names
        save_paths = [run.save_path for run in normalized.runs]
        if len(save_paths) != 3 or not all(Path(p).exists() for p in save_paths):
            continue
        # experiments/<model-class>/<setting>/<experiment>/outputs/<arm>/...
        rel = Path(save_paths[0]).relative_to("experiments")
        arm = Path(save_paths[0]).parent.name
        key = f"{'/'.join(rel.parts[1:3])}/{arm}"
        found.append((key, cfg_path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", nargs="+", required=True)
    args = parser.parse_args()
    for tree in args.trees:
        for key, cfg_path in runnable_configs(tree):
            print(f"{key}\t{cfg_path}")


if __name__ == "__main__":
    main()
