"""Full Nash (double-oracle) sweep over human_audit_success_rate.

Closed/APPS self_rec paraphrased_sft. Both teams re-optimized per rate with
the rate common knowledge; full usefulness sweep per rate.
"""
import json
from joblib import Parallel, delayed
from lasr_labs_2025_control_project.scripts.plot_data import build_jsonl_dataframe_from_eval_log
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    build_optimization_config, optimize_parameters,
)

D = "experiments/closed-source-model/apps/self_rec/outputs/paraphrased_sft"
attacks = build_jsonl_dataframe_from_eval_log(f"{D}/attack_monitor_results.eval")
honeypots = build_jsonl_dataframe_from_eval_log(f"{D}/honeypots_monitor_results.eval")
benign = build_jsonl_dataframe_from_eval_log(f"{D}/honest_monitor_results.eval")

def run(rate):
    cfg = build_optimization_config(
        num_tasks=5000, blue_team_budget_fraction=0.02, slack_honeypots=5.0,
        attacks_df=attacks, honeypots_df=honeypots, benign_df=benign,
        human_audit_success_rate=rate,
    )
    res = optimize_parameters(base_config=cfg, max_iterations=100)
    return {"rate": rate, "converged": res.converged, "sweep": [
        {k: e[k] for k in ("usefulness", "safety", "blue_team_replace_fnr",
                           "best_parameters", "blue_mixed_strategy",
                           "red_mixed_strategy", "converged")}
        for e in res.results]}

rows = Parallel(n_jobs=3)(delayed(run)(round(0.1*i,1)) for i in range(1, 11))
out = "docs/dependent_monitors_stats/results_real_paper_data/audit_success_rate_nash_sweep_paraphrased_sft.json"
json.dump({"condition": "closed-source-model/apps/self_rec/paraphrased_sft",
           "note": "full double-oracle Nash per rate; both teams know the rate",
           "rows": rows}, open(out, "w"), indent=1, default=float)
print("saved", out)
for r in rows:
    line = " ".join(f"{e['usefulness']:.2f}:{e['safety']:.3f}" for e in r["sweep"][:4])
    print(f"rate {r['rate']:.1f} converged={r['converged']} | {line}")
