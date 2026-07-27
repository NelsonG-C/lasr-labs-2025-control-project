"""Five-panel safety-usefulness bootstrap band plot (median + 95% CI).

Panels: acausal / causal base / causal SFT / self-rec base / self-rec SFT.
Lines: none / minified / paraphrased. One figure per model class (open has
all five panels; closed lacks causal none/minified SFT so gets four).
"""
import gzip, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="paper")
ROOT = Path("/workspace/data/bootstrap_results")
OUT = Path("docs/dependent_monitors_stats/results_real_paper_data")

PANELS = [
    ("Acausal", "apps/acausal", "base"),
    ("Causal", "apps/causal", "base"),
    ("Causal + SFT", "apps/causal", "sft"),
    ("Passive self-rec", "apps/self_rec", "base"),
    ("Passive self-rec + SFT", "apps/self_rec", "sft"),
]
TRANSFORMS = ["none", "minified", "paraphrased"]
PALETTE = dict(zip(TRANSFORMS, sns.color_palette("colorblind", 3)))

def bands(path):
    reps = [json.loads(l) for l in gzip.open(path, "rt")]
    reps = [r for r in reps if "sweep" in r]
    us = [e["usefulness"] for e in reps[0]["sweep"]]
    S = np.array([[e["safety"] for e in r["sweep"]] for r in reps])
    lo, med, hi = np.percentile(S, [2.5, 50, 97.5], axis=0)
    order = np.argsort(us)
    return np.array(us)[order], med[order], lo[order], hi[order]

for cls in ["open-source-model", "closed-source-model"]:
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True, sharex=True)
    for ax, (title, group, suffix) in zip(axes, PANELS):
        drawn = False
        for t in TRANSFORMS:
            p = ROOT / cls / group / f"{t}_{suffix}" / "replicates.jsonl.gz"
            if not p.exists():
                continue
            u, med, lo, hi = bands(p)
            ax.plot(u, med, label=t, color=PALETTE[t])
            ax.fill_between(u, lo, hi, color=PALETTE[t], alpha=0.2, linewidth=0)
            drawn = True
        ax.set_title(title)
        ax.set_xlabel("Usefulness")
        if not drawn:
            ax.text(0.5, 0.5, "no data\n(unpublished runs)", ha="center",
                    va="center", transform=ax.transAxes, color="grey")
    axes[0].set_ylabel("Safety")
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(title="Transform", loc="lower left", frameon=True)
    fig.suptitle(f"{cls} / APPS — bootstrap median and 95% CI (B=1000), joint stats model", y=1.04)
    fig.tight_layout()
    dest = OUT / f"safety_usefulness_bootstrap_panels_{cls.split('-')[0]}.pdf"
    fig.savefig(dest, bbox_inches="tight")
    print("saved", dest)
