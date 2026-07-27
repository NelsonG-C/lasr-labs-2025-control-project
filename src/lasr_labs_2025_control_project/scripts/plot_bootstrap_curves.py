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
)

PANELS = [
    ("acausal", "Acausal", "acausal", "base"),
    ("causal", "Causal", "causal", "base"),
    ("causal_sft", "Causal + SFT", "causal", "sft"),
    ("self_rec", "Passive Self-Rec", "self_rec", "base"),
    ("self_rec_sft", "Passive Self-Rec + SFT", "self_rec", "sft"),
]
TRANSFORMS = [("none", "None"), ("minified", "Minified"), ("paraphrased", "Paraphrased")]


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
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Default: <bootstrap-root>/<class>/<setting>/plots")
    parser.add_argument("--ci", type=float, default=0.80)
    args = parser.parse_args()

    ci_label = f"{args.ci:.0%} CI, bootstrap"
    for class_dir in sorted(args.bootstrap_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for setting_dir in sorted(class_dir.iterdir()):
            if not setting_dir.is_dir():
                continue
            for panel_slug, panel_title, group, suffix in PANELS:
                bands = {}
                for transform, label in TRANSFORMS:
                    path = (setting_dir / group / f"{transform}_{suffix}"
                            / "replicates.jsonl.gz")
                    if path.exists():
                        bands[label] = load_band(path, args.ci)
                if not bands:
                    continue
                out_dir = args.out_root or (setting_dir / "plots")
                plot_safety_usefulness_bands(
                    bands,
                    save_path=str(
                        out_dir
                        / f"{panel_slug}_safety_usefulness_bands.{PLOT_FILE_FORMAT}"
                    ),
                    ci_label=f"{panel_title}, {ci_label}",
                )


if __name__ == "__main__":
    main()
