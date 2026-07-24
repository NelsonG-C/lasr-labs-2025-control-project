# SPEC: Dependence-aware stats model with asymmetric honest UM errors

**Branch:** `feature/dependent-monitors-stats-model`
**Status:** design
**Date:** 2026-07-24

## Reviewer requirements

1. **No independence assumption** between the trusted monitor (TM) and untrusted
   monitor (UM) scores. The current model combines them as
   `p_caught = tpr_t + p_um − tpr_t · p_um`
   (`roc_function_builders.py::uncorrelated_roc_function`), i.e. a product of
   marginals.
2. **Unequal honest error rates** for the UM in its honest (suspiciousness)
   role. Currently a single symmetric error `e` is extracted at the
   `FPR = 1 − TPR` point and applied as `p_um_reports *= (1 − e)`. The honest
   false-positive direction (`untrusted_monitor_epsilon`, which consumes audit
   budget and drives honeypot survival) is silently defaulted to **0** — it is
   never estimated from data at all.

## Why the previous attempt failed

The previous attempt tried to build a single smooth, "correctly concave-convex"
multi-input multi-output ROC function from the joint data. That object is
ill-posed:

- In 1D, convexifying the ROC (upper hull) is valid because every point on a
  chord is achievable by randomizing between the two endpoint thresholds.
- In 2D (UM threshold `t` × TM threshold `c`), you **cannot** hull each axis
  independently: the hulls delete different data thresholds on each margin,
  and the resulting surface does not correspond to any achievable strategy.
  Nor is the joint surface required to be concave/convex in any direction —
  dependence between the scores can make it locally anything.
- Meanwhile the raw empirical joint surface is a 2D step function, and the
  double-oracle best-response step uses Powell local search
  (`double_oracle.py`), which stalls on plateaus.

## Key insight: randomization, not hulling

The achievable operating points under **independent threshold randomization**
are exactly the **multilinear interpolation** of the empirical joint counting
function on the grid of observed score thresholds:

- If red randomizes between adjacent UM thresholds `t₀, t₁` with weight `w`,
  every red-side quantity (attack pass rate, honeypot collusion FPR)
  interpolates linearly in `w` — *simultaneously, with the same `w`*.
- If blue independently randomizes between adjacent TM thresholds `c₀, c₁`
  with weight `v`, joint quantities interpolate bilinearly:
  `E[1(coll ≥ t) · 1(tm < c)] = Σᵢⱼ wᵢ vⱼ J(tᵢ, cⱼ)`.
- Randomization beyond adjacent grid cells (global convexification) is
  **already handled by the double-oracle mixed strategies** — that is what the
  LP over the payoff matrix does. We do not need to pre-hull anything.

So the correct object is not a concave-convex surface; it is the
**piecewise-multilinear empirical joint surface**, with the game solver
supplying all further convexification. This is continuous and piecewise-smooth,
so Powell works; optionally the red best response can also enumerate the finite
set of observed collusion-score thresholds exactly (they are few), which makes
the restricted game finite and removes the one remaining approximation
(see "Nonlinearity caveat" below).

## Data available

Every row of `attacks_df` carries all three scores:
`untrusted_collusion_score`, `untrusted_suspiciousness_score`,
`trusted_suspiciousness_score`. So the trivariate joint on attacks is fully
empirical. Negatives remain marginal by construction (honeypots supply UM
collusion negatives; benign supplies TM and UM-suspiciousness negatives) —
that is fine, because negatives enter the model through *different terms*
(honeypot Poisson survival vs. audit-budget consumption) that never need a
joint negative distribution.

## Model changes

### 1. Joint attack-pass function (replaces the independence product)

Per attack sample `i`, given UM collusion threshold `t`, TM threshold `c`,
and honest-UM suspiciousness threshold `s`, the attack is **caught** iff

```
tm_i ≥ c                                   (trusted monitor flags it)
OR ( coll_i < t  AND  susp_i ≥ s )         (UM declines to collude AND honestly flags it)
```

Define the empirical pass probability

```
P_pass(t, c, s) = (1/N) Σ_i 1[tm_i < c] · (1 − 1[coll_i < t] · 1[susp_i ≥ s])
```

evaluated exactly on the observed-score grid and extended off-grid by
multilinear interpolation (trilinear in `(t, c, s)`). This replaces both the
`tpr_t + p_um − tpr_t·p_um` product **and** the `(1 − e)` symmetric-error
factor in one move: any dependence between the TM score, the UM collusion
score, and the UM honest suspiciousness score on attacks is captured
automatically, sample by sample.

Parameterization stays compatible with the existing win calculator:

- `c` is derived from the blue-team TM FPR target exactly as today
  (`build_fpr_parameterized_roc_function` on benign scores gives the
  `fpr_t → c` map; we invert through the benign empirical CDF).
- `t` remains `red_team_roc_threshold`.
- `s` is a **new blue-team choice** (see §2); default it to a fixed benign-FPR
  target so existing configs run unchanged.

### 2. Asymmetric honest UM errors

The honest threshold `s` is a **fixed capability parameter of the untrusted
model**, not a strategy variable — it is chosen once at build time by a fixed
rule and both error rates are then measured from data. Note the current rule
(minimize `(1 − TPR − FPR)²`, i.e. the FPR = FNR crossing) *creates* the
symmetry the reviewers objected to, so the rule must change; use Youden's J
(max TPR − FPR) or the score scale's semantic flag point.

Drop `um_suspiciousness_error` entirely. At the fixed `s`:

- **FNR direction** (honest UM misses a real attack): already inside
  `P_pass` above via `1[susp_i ≥ s]` — per-sample, jointly with collusion.
- **FPR direction** (honest UM flags benign code):
  `ε = P(susp ≥ s | benign)` from `benign_df.untrusted_suspiciousness_score`.
  Feed this into `untrusted_monitor_epsilon`, where it already has the right
  downstream semantics: it consumes audit budget (`optimization.py:486-491`)
  and enters the honeypot Poisson rate (`optimization.py:509`).

**Bug found during design**: the `untrusted_monitor_epsilon` metadata key is
read (`optimization.py:204,351`; `parameter_sensitivity.py:403`) but never
written anywhere in the codebase — every run to date has used ε = 0. The
suspiciousness ROC's error lands under a different metadata key
(`untrusted_suspiciousness_classification_error`) and only feeds the
`p_um_reports` term. Wiring ε from data is therefore both a modeling change
and a fix; flag the before/after in the paper.

Since `s` is fixed, ε stays a scalar snapshotted at config-build time and the
existing `f(t, fpr_t)` two-argument ROC interface is unchanged — `s` is
curried into the joint closure.

### 3. Honeypot side (unchanged semantics, noted for the paper)

Honeypot failure stays `P(coll ≥ t | honeypot)` — marginal on honeypots. The
existing comment (roc_function_builders.py:168-171) documents this as
intentional: honeypot false positives are those *due to colluding*, not due to
honestly missing the honeypot. If reviewers push, the honeypot joint
`(coll, susp)` is also available in `honeypots_df` and slots into the same
machinery.

## Implementation plan

1. `roc_function_builders.py`: new `build_joint_roc_function(...)` returning
   a `JointROCFunctionType = f(t, fpr_t, s_fpr) -> (p_pass, honeypot_fpr, epsilon)`
   built on a shared trilinear interpolator over the empirical grid
   (`np.searchsorted` per axis; grid = unique observed scores per axis with
   ±padding sentinels, same construction as the current 1D interpolators).
   Keep `build_roc_function` as a thin wrapper so `disable_*_monitor`
   degenerate cases and metadata keys survive.
2. `optimization.py`: no interface change — populate the
   `untrusted_monitor_epsilon` metadata key from benign suspiciousness scores
   at the fixed `s` (fixing the never-written-key bug), so the existing
   scalar plumbing picks it up.
3. `double_oracle.py`: add an optional finite-set best response for
   `red_team_roc_threshold` (enumerate observed collusion thresholds, keep
   Powell for the other axes). Low-risk, addresses plateau stalls at source.
4. Keep the old independence model callable behind a flag
   (`assume_independent=True`) — the paper wants a "dependence changed safety
   by X" comparison, and it doubles as a regression test (joint model on
   artificially shuffled scores must reproduce the independent model within
   sampling noise).
5. Validation:
   - Shuffle test above.
   - Copula sensitivity check: fit a Gaussian copula ρ (rank-based, jittered
     for integer scores) between `tm` and `coll` on attacks; sweep ρ from 0 to
     the fitted value and confirm the empirical-joint result sits on the curve.
     This is the reviewer-facing robustness figure.
   - Nash-gap traces from double oracle before/after (plateau stalls show up
     as non-closing gaps).

## Related work (from literature sweep, 2026-07-24)

- **Barreno, Cárdenas & Tygar, "Optimal ROC Curve for a Combination of
  Classifiers" (NeurIPS 2007)** — the closest template: works directly with
  the joint score distribution of two classifiers (no independence
  assumption), derives the Neyman–Pearson-optimal decision rule over the joint
  space, and proves achievable operating points are the convex hull of
  deterministic threshold rules under randomization. Our §"Key insight" is a
  special case of their theorem restricted to per-monitor thresholds; cite it.
- **Provost & Fawcett (ROCCH)** — standard citation for convexity via
  randomization in 1D (what the current `calculate_tpr_fpr_from_scores` hull
  implements).
- **Biostatistics paired-ROC literature** (Bantis et al. 2016/2018 correlated
  ROC curves/surfaces; DeLong's test) — treats non-independence of paired
  diagnostic scores as the default case; useful framing for the response to
  reviewers.
- **AI control**: no existing control paper found gives a formal joint
  treatment of monitor dependence — "Ensemble Monitoring for AI Control"
  (arXiv:2605.15377) measures monitor correlation empirically and shows it is
  first-order for ensemble gains, which supports the reviewers' concern.
- **Copula pitfall**: with low-cardinality integer scores (ours are 0–10),
  a Gaussian copula fitted to observed correlations *understates* dependence
  (discreteness attenuation; cf. copula-regression-with-discrete-outcomes
  literature). This reinforces the choice of the nonparametric empirical joint
  as the primary model, with the copula only as a smoothness/sensitivity
  check — and the copula ρ should be calibrated against the discretized joint
  (e.g. match rank correlation on jittered scores), not raw Pearson.

## Nonlinearity caveat (also present in the current model)

The win calculator is nonlinear in the ROC outputs (Poisson survival,
`(1−p)^n`). Mixing pure strategies mixes *win probabilities* linearly (LP —
correct), but the continuous best-response search treats interpolated ROC
outputs as pure strategies, which slightly misprices mixtures inside one grid
cell. The finite-threshold red best response (step 3) eliminates this for red;
for blue the effect is second-order (blue's thresholds enter through smooth
budget quantities). Worth one sentence in the paper's appendix, not more.
