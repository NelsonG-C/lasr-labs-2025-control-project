"""human_audit_success_rate sweep, closed/APPS self_rec paraphrased_sft.

Blue's strategy is held at the rate=1.0 Nash mixture (per Jonathan: no
re-optimization of blue); red best-responds at each rate. A fixed-red column
separates the mechanical catch-rate effect from red's adaptation.
"""
import gzip, json
import numpy as np
from scipy import optimize as sopt
from lasr_labs_2025_control_project.scripts.plot_data import build_jsonl_dataframe_from_eval_log
from lasr_labs_2025_control_project.utils.numerical_pipeline import (
    build_optimization_config,
)

D = "experiments/closed-source-model/apps/self_rec/outputs/paraphrased_sft"
attacks = build_jsonl_dataframe_from_eval_log(f"{D}/attack_monitor_results.eval")
honeypots = build_jsonl_dataframe_from_eval_log(f"{D}/honeypots_monitor_results.eval")
benign = build_jsonl_dataframe_from_eval_log(f"{D}/honest_monitor_results.eval")

stored = json.load(gzip.open(
    "/workspace/data/untrusted-monitoring-2026-paper/closed-source-model/apps/self_rec/results/paraphrased_sft/analysis_data.json.gz", "rt"))
entry = stored["sweep"][0]  # usefulness 1.0
blue_mix = [(s["blue_team_honeypot_fraction"], p) for s, p in entry["blue_mixed_strategy"]]
red_mix = [((s["red_team_roc_threshold"], s["red_team_p_attack"]), p)
           for s, p in entry["red_mixed_strategy"]]

def make_expected_win(rate):
    cfg = build_optimization_config(
        num_tasks=5000, blue_team_budget_fraction=0.02, slack_honeypots=5.0,
        attacks_df=attacks, honeypots_df=honeypots, benign_df=benign,
        human_audit_success_rate=rate,
    )
    calc = cfg.blue_team_win_calculator
    tmin = cfg.untrusted_threshold_min
    tmax = cfg.untrusted_threshold_max
    def expected_win(t, logp):
        return sum(pb * float(calc({
            "blue_team_honeypot_fraction": np.float64(f),
            "red_team_roc_threshold": np.float64(t),
            "red_team_p_attack": np.float64(10.0 ** logp),
        })[0]) for f, pb in blue_mix)
    return expected_win, tmin, tmax

rows = []
for rate in [round(0.1 * i, 1) for i in range(1, 11)]:
    ew, tmin, tmax = make_expected_win(rate)
    # fixed red: evaluate stored red mixture
    fixed = sum(pr * ew(t, np.log10(p)) for (t, p), pr in red_mix)
    # red best response: coarse grid then Powell polish from the best cells
    tg = np.linspace(tmin, tmax, 41)
    pg = np.linspace(-6, -1, 26)
    grid = [(ew(t, lp), t, lp) for t in tg for lp in pg]
    grid.sort(key=lambda x: x[0])
    best = min(
        (sopt.minimize(lambda x: ew(x[0], x[1]), [t, lp], method="Powell",
                       bounds=[(tmin, tmax), (-6, -1)]) for _, t, lp in grid[:5]),
        key=lambda r: r.fun,
    )
    rows.append({"rate": rate, "safety_red_adapts": float(best.fun),
                 "safety_red_fixed": float(fixed),
                 "red_br_threshold": float(best.x[0]),
                 "red_br_p_attack": float(10 ** best.x[1])})
    r = rows[-1]
    print(f"rate {rate:.1f}: S(red adapts)={r['safety_red_adapts']:.3f}  "
          f"S(red fixed)={r['safety_red_fixed']:.3f}  "
          f"BR: t={r['red_br_threshold']:.2f} p_attack={r['red_br_p_attack']:.2e}", flush=True)

out = "docs/dependent_monitors_stats/results_real_paper_data/audit_success_rate_sweep_paraphrased_sft.json"
json.dump({"condition": "closed-source-model/apps/self_rec/paraphrased_sft",
           "usefulness": 1.0, "blue_mixture_fixed_at_rate1_nash": blue_mix,
           "rows": rows}, open(out, "w"), indent=1)
print("saved", out)
