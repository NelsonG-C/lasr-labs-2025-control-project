"""Render house-style safety/usefulness band plots from bootstrap replicates.

One figure per (model class, setting, strategy panel), each with the three
transform arms (none / minified / paraphrased) as labelled curves: bootstrap
median plus a percentile band (80% by default). Controls and the tm_only
ablation are excluded. Uses plot_safety_usefulness_bands, which mirrors the
layout of the point-estimate plot_safety_usefulness_curves.

    PYTHONPATH=src python .../plot_bootstrap_curves.py \
        --bootstrap-root /workspace/data/bootstrap_results --ci 0.80
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from lasr_labs_2025_control_project.utils.numerical_pipeline.visualization import (
    PLOT_FILE_FORMAT,
    plot_safety_usefulness_bands,
    plot_safety_usefulness_bands_grid,
)


PANELS = [
    ("acausal", "Acausal", "acausal", "base"),
    ("causal", "Causal", "causal", "base"),
    ("causal_sft", "Causal + SFT", "causal", "sft"),
    ("self_rec", "Passive Self-Rec", "self_rec", "base"),
    ("self_rec_sft", "Passive Self-Rec + SFT", "self_rec", "sft"),
]
TRANSFORMS = [
    ("none", "None"),
    ("minified", "Minified"),
    ("paraphrased", "Paraphrased"),
]


def load_band(path: Path, ci: float) -> dict[str, list[float]]:
    replicates = [json.loads(line) for line in gzip.open(path, "rt")]
    sweeps = [r["sweep"] for r in replicates if "sweep" in r]
    usefulness = [entry["usefulness"] for entry in sweeps[0]]
    safety = np.array([[entry["safety"] for entry in sweep] for sweep in sweeps])
    tail = 100.0 * (1.0 - ci) / 2.0
    lo, med, hi = np.percentile(safety, [tail, 50.0, 100.0 - tail], axis=0)
    return {
        "usefulness": usefulness,
        "safety_median": med.tolist(),
        "safety_lo": lo.tolist(),
        "safety_hi": hi.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-root", type=Path, required=True)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Default: <bootstrap-root>/<class>/<setting>/plots",
    )
    parser.add_argument("--ci", type=float, default=0.80)
    parser.add_argument(
        "--audit",
        type=Path,
        default=None,
        help=(
            "event_count_audit.json (helpers/event_count_audit.py). When "
            "given, ALSO emit *_diagnostic figures with X markers on cells "
            "the audit marked SATURATED."
        ),
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text()) if args.audit else None

    def saturated_us(class_name: str, setting: str, group: str, arm: str) -> dict:
        if audit is None:
            return {}
        rep = audit.get(f"{class_name}/{setting}/{group}/{arm}")
        if not rep:
            return {}
        return {
            c["usefulness"]: c["verdict"]
            for c in rep["cells"]
            if c["verdict"] in ("SATURATED", "CRITICAL")
        }

    ci_label = f"{args.ci:.0%} CI, bootstrap"
    for class_dir in sorted(args.bootstrap_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for setting_dir in sorted(class_dir.iterdir()):
            if not setting_dir.is_dir():
                continue
            panels: dict[str, dict] = {}
            panel_saturation: dict[str, dict[str, set]] = {}
            for panel_slug, panel_title, group, suffix in PANELS:
                bands = {}
                sat: dict[str, set] = {}
                for transform, label in TRANSFORMS:
                    path = (
                        setting_dir
                        / group
                        / f"{transform}_{suffix}"
                        / "replicates.jsonl.gz"
                    )
                    if path.exists():
                        bands[label] = load_band(path, args.ci)
                        sat[label] = saturated_us(
                            class_dir.name,
                            setting_dir.name,
                            group,
                            f"{transform}_{suffix}",
                        )
                if not bands:
                    continue
                panels[panel_title] = bands
                panel_saturation[panel_title] = sat
                out_dir = args.out_root or (setting_dir / "plots")
                plot_safety_usefulness_bands(
                    bands,
                    save_path=str(
                        out_dir
                        / f"{panel_slug}_safety_usefulness_bands.{PLOT_FILE_FORMAT}"
                    ),
                    ci_label=f"{panel_title}, {ci_label}",
                )
                if audit is not None:
                    plot_safety_usefulness_bands(
                        bands,
                        save_path=str(
                            out_dir
                            / f"{panel_slug}_safety_usefulness_bands_diagnostic.{PLOT_FILE_FORMAT}"
                        ),
                        ci_label=f"{panel_title}, {ci_label}",
                        saturated=sat,
                    )
            if panels:
                out_dir = args.out_root or (setting_dir / "plots")
                plot_safety_usefulness_bands_grid(
                    panels,
                    save_path=str(
                        out_dir
                        / f"assembled_safety_usefulness_bands.{PLOT_FILE_FORMAT}"
                    ),
                    ci_label=f"{args.ci:.0%} CI",
                    suptitle=f"{class_dir.name} / {setting_dir.name}",
                )
                if audit is not None:
                    plot_safety_usefulness_bands_grid(
                        panels,
                        save_path=str(
                            out_dir
                            / f"assembled_safety_usefulness_bands_diagnostic.{PLOT_FILE_FORMAT}"
                        ),
                        ci_label=f"{args.ci:.0%} CI",
                        suptitle=f"{class_dir.name} / {setting_dir.name}",
                        saturated=panel_saturation,
                    )


if __name__ == "__main__":
    main()
