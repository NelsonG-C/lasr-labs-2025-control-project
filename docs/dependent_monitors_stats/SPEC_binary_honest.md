# SPEC: Binary honest verdicts with honest-error × TM-error correlation

**Branch:** `feature/dependent-monitors-stats-model` (extends the joint model
in [SPEC.md](SPEC.md))
**Status:** implemented
**Date:** 2026-07-24

## Input schema

Three dataframes — attacks, benign (non-attacks), honeypots — whose rows carry:

| column | type | meaning |
|---|---|---|
| `untrusted_honest_score` | bool | the honest UM's binary flag on this sample |
| `untrusted_collusion_score` | number | UM collusion-recognition score |
| `trusted_suspiciousness_score` | number | TM suspiciousness score |

The honest verdict is already binary, so there is **no operating point `s` to
choose** — the Youden's J step from SPEC.md §2 disappears, and the honest
error rates are direct empirical frequencies:

- honest FNR = `P(¬honest_flag | attack)`
- honest FPR = `P(honest_flag | benign)`

When `untrusted_honest_score` is absent, the builder falls back to the
numeric `untrusted_suspiciousness_score` + Youden path (SPEC.md), then to the
perfect-honest-UM fallback, unchanged.

## What correlation with TM errors changes

### Attack side (coverage overlap)

Attack `i` is caught iff `tm_i ≥ c` OR (`coll_i < t` AND `honest_i`). This is
the same per-sample joint as SPEC.md with `1[susp_i ≥ s]` replaced by the
boolean, so honest-FNR × TM-FNR correlation is captured automatically: if the
UM honestly misses exactly the attacks the TM also misses (positive error
correlation), the OR-coverage is worse than the independence product would
claim. Nothing new to build — the pass surface already conditions on the
row.

### Benign side (audit-demand overlap) — the new object

Every UM flag consumes an audit. The legacy budget accounting charges the
full marginal honest FPR against the audit budget:

```
trusted_monitor_fnr = max(audit_fraction − FPR_h, 0)
```

But a benign task the UM flags **that the TM already sends to audit** costs
nothing extra. The genuinely new audits forced by the honest UM at TM benign
FPR `f` (threshold `c(f)`) are

```
ε_extra(f) = P(honest_flag ∧ tm < c(f) | benign)
```

computed row-aligned on the benign dataframe and interpolated over the
achievable benign-FPR grid (same randomization-achievability argument as the
attack surface: linear interpolation in `f` = mixing adjacent TM thresholds).
Limits: `ε_extra(0) = FPR_h`, `ε_extra(1) = 0`. Positive honest-error × TM
false-alarm correlation pushes `ε_extra` below the independence product
`FPR_h · P(tm < c(f))` — overlapping errors waste less budget.

### Budget fixed point

The TM's benign FPR `f` and the UM's extra demand must jointly fit the audit
budget:

```
f* solves  f + ε_extra(f) = audit_fraction
```

`f + ε_extra(f)` is non-decreasing in `f` (auditing one more benign task adds
1 and removes at most 1 from the UM's extra demand), so the fixed point is
found by bisection on `[0, audit_fraction]`; if `ε_extra(0) ≥ audit_fraction`
the budget is exhausted and `f* = 0`. `f*` replaces the legacy
`max(audit_fraction − FPR_h, 0)` and is ≥ it, with equality when the honest
flags are disjoint from the TM's audited set.

Kept as-is (marginal quantities, per earlier decisions):

- `p_audited = min(audit_fraction / (FPR_h + 1/N), 1)` — a report is a
  report; over-demand rationing uses the marginal flag rate.
- Honeypot retraining dilution `n_eff = n · (1 − (FPR_h + FNR_h))` — scalar
  marginals at the (now trivial) operating point.
- Honeypot collusion failure = marginal `P(coll ≥ t | honeypot)`.

## Implementation

- `build_joint_roc_function` (roc_function_builders.py): boolean path takes
  precedence over the numeric-susp path. Attack surface uses the boolean
  per-sample; benign rows are jointly dropna'd on
  (`trusted_suspiciousness_score`, `untrusted_honest_score`) so the TM grid
  and `ε_extra` come from the same rows. New metadata:
  `untrusted_honest_extra_fpr_fn` (callable `f → ε_extra(f)`);
  `untrusted_honest_threshold` is NaN in this path.
- `build_blue_team_win_calculator` (optimization.py): optional
  `untrusted_honest_extra_fpr` callable; when present, `trusted_monitor_fnr`
  comes from the bisection fixed point (vectorized over parameter arrays).
  All three call sites forward the metadata key.
- Tests: `tests/test_binary_honest_roc.py` (surface correctness, ε_extra
  correctness, correlation direction on both sides, fixed-point consistency,
  fallback); fixtures in `tests/conftest.py`
  (`make_boolean_score_dataframes`, with tunable benign honest×TM
  correlation).
