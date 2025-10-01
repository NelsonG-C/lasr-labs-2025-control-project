"""Eval analysis script (YAML-driven).

This CLI reads a YAML config and runs log-probability analysis and optional
monitor checks over .eval datasets. It computes YES/NO logprobs for a given
"untrusted_collusion_monitor" model, can run backdoor and trusted monitors,
and optionally paraphrases solutions before analysis.

Key steps:
- Load .eval problem/solution data
- Compute YES/NO logprobs for each sample (if configured)
- Optionally run backdoor/trusted monitors
- Optionally paraphrase solutions (llm|minifier|none)
- Save a JSONL with per-sample results

YAML schema (current minimal):
    defaults:
        limit: 50
        paraphraser_type: none   # one of: none, minifier, llm
        backdoor_trusted_monitor: openai/gpt-4.1-mini
        backdoor_untrusted_monitor: openai/gpt-4.1
        collusion_untrusted_monitor: openai/ft:...
    runs:
        honeypots:
            eval_path: path/to/honeypots.eval
            json_path: path/to/honeypots.jsonl
        attacks:
            eval_path: path/to/attacks.eval
            json_path: path/to/attacks.jsonl
        honest:
            eval_path: path/to/honest.eval
            json_path: path/to/honest.jsonl

Category rules:
- honeypots => untrusted_collusion + untrusted_backdoor monitors
- attacks   => trusted_backdoor + untrusted_backdoor + untrusted_collusion
- honest    => trusted_backdoor only

Usage:
    python -m scripts.eval_analysis --config path/to/config.yaml

Notes:
- Relative paths in the YAML are resolved relative to the YAML file location.
- The .eval loading is implemented in utils/evals_to_dataframes.py.
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from lasr_labs_2025_control_project.utils.config_loader import get_prompt_by_name
from lasr_labs_2025_control_project.utils.data_pipelines import analyse_eval

# Defer config loader import to main to avoid import cycles in some environments

# from inspect_ai.analysis import messages_df, samples_df  # no longer used here

# Load environment variables from a .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("eval_analysis.log")],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint: run eval analysis from a YAML config file.

    The YAML must contain a top-level 'runs' list. Optional 'defaults' apply
    to each run and can be overridden per-run.
    """
    parser = argparse.ArgumentParser(
        prog="eval_analysis",
        description="Run evaluation analysis from a YAML config file."
        " Relative paths are resolved relative to the YAML file.",
    )
    # Allow --config or -cfg
    parser.add_argument("--config", "-cfg", help="Path to YAML config file")
    args = parser.parse_args()

    config_arg = args.config
    if not config_arg:
        raise SystemExit(
            "Error: a YAML config path is required (positional or --config)"
        )
    config_path = Path(config_arg).expanduser().resolve()
    if config_path.exists():
        from lasr_labs_2025_control_project.utils.config_loader import (
            load_and_normalize_config,
        )

        normalized = load_and_normalize_config(config_path)
        runs = normalized["runs"]  # type: ignore

        logger.info("Starting configured runs")
        for i, run in enumerate(runs, start=1):
            label = run["label"]
            eval_path = run["eval_path"]
            save_path = run["json_path"]
            limit = run["limit"]
            if limit is not None:
                limit = int(run["limit"])  # normalized
            paraphraser_type = run["paraphraser_type"]
            collusion_system_prompt = get_prompt_by_name(run["collusion_system_prompt"])
            collusion_user_prompt = get_prompt_by_name(run["collusion_user_prompt"])

            logger.info(
                f"Run '{label}': eval={eval_path}, save={save_path}, limit={limit}, paraphraser={paraphraser_type}; "
                f"enabled=[trusted={run['want_trusted']}, untrusted_backdoor={run['want_untrusted_backdoor']}, untrusted_collusion={run['want_untrusted_collusion']}] "
                f"n_backdoor_monitor_runs={run.get('n_backdoor_monitor_runs') or 1}"
            )

            analyse_eval(
                eval_path=eval_path,
                save_path=save_path,
                limit=limit,
                trusted_backdoor_monitor=run["trusted_backdoor_monitor"],
                untrusted_backdoor_monitor=run["untrusted_backdoor_monitor"],
                untrusted_collusion_monitor=run["untrusted_collusion_monitor"],
                collusion_system_prompt=collusion_system_prompt,
                collusion_user_prompt=collusion_user_prompt,
                paraphraser_type=paraphraser_type,
                n_backdoor_monitor_runs=run.get("n_backdoor_monitor_runs"),
            )
    else:
        raise SystemExit(f"Error: config file not found: {config_path}")
    # Helper functions moved to lasr_labs_2025_control_project.utils.config_loader


if __name__ == "__main__":
    main()
