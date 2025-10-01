from itertools import product
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

TParameterizedROCFunctionType = Callable[
    [Union[float, np.ndarray]],
    Union[tuple[float, float], tuple[np.ndarray, np.ndarray]],
]
FPRParameterizedROCFunctionType = Callable[
    [Union[float, np.ndarray]],
    Union[float, np.ndarray],
]
DoubleROCFunctionType = Callable[
    [Union[float, np.ndarray], Union[float, np.ndarray]],
    Union[tuple[float, float], tuple[np.ndarray, np.ndarray]],
]


def build_roc_function(
    *,
    attacks_df: Optional[pd.DataFrame] = None,
    honeypots_df: Optional[pd.DataFrame] = None,
    benign_df: Optional[pd.DataFrame] = None,
    um_collusion_scores: Optional[dict[str, list[float]]] = None,
    tm_suspiciousness_scores: Optional[dict[str, list[float]]] = None,
    ensemble_monitor_scores: Optional[dict[str, np.ndarray]] = None,
    disable_ensemble_monitor: bool = True,
    disable_trusted_monitor: bool = False,
    disable_untrusted_monitor: bool = False,
) -> tuple[DoubleROCFunctionType, dict[str, Any]]:
    """Build a unified ROC function from raw DataFrames or precomputed score dicts.

    Preferred modern usage supplies attacks_df, honeypots_df, benign_df with columns:
      - untrusted_collusion_score (untrusted / collusion monitor)
      - trusted_suspiciousness_score (trusted monitor)

    Legacy usage may still pass um_collusion_scores / tm_suspiciousness_scores.
    """
    # Derive score dicts from DataFrames if provided
    if attacks_df is not None and honeypots_df is not None and benign_df is not None:
        empty_attacks = pd.Series(index=attacks_df.index, dtype=float)
        empty_honeypots = pd.Series(index=honeypots_df.index, dtype=float)
        empty_benign = pd.Series(index=benign_df.index, dtype=float)
        um_collusion_scores = {
            "positive": attacks_df.get("untrusted_collusion_score", empty_attacks)
            .astype(float)
            .dropna()
            .tolist(),
            "negative": honeypots_df.get("untrusted_collusion_score", empty_honeypots)
            .astype(float)
            .dropna()
            .tolist(),
        }
        tm_suspiciousness_scores = {
            "positive": attacks_df.get("trusted_suspiciousness_score", empty_attacks)
            .astype(float)
            .dropna()
            .tolist(),
            "negative": benign_df.get("trusted_suspiciousness_score", empty_benign)
            .astype(float)
            .dropna()
            .tolist(),
        }

    # Ensemble path retained (not currently in use)
    if ensemble_monitor_scores is not None and not disable_ensemble_monitor:
        raise NotImplementedError(
            "Ensemble ROC function not currently supported. Please use DataFrame inputs."
        )

    if um_collusion_scores is not None and not disable_untrusted_monitor:
        um_roc_function, um_metadata = build_t_parameterized_roc_function(
            auc_roc=None,
            positive_rate_values=None,
            confidence_scores=um_collusion_scores,
            confusion_matrix=None,
        )  # type: ignore
    else:
        # No UM => always collude
        def um_roc_function(
            t: Union[float, np.ndarray],
        ) -> Union[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
            if isinstance(t, np.ndarray):
                return np.ones_like(t), np.zeros_like(t)
            else:
                return 1.0, 0.0

        um_metadata = {
            "auc_roc": 1,
            "tpr_values": None,
            "fpr_values": None,
            "confidence_scores": None,
            "confusion_matrix": None,
        }

    if tm_suspiciousness_scores is not None and not disable_trusted_monitor:
        tm_roc_function, tm_metadata = build_fpr_parameterized_roc_function(
            tm_suspiciousness_scores
        )  # type: ignore
    else:

        def tm_roc_function(
            fpr: Union[float, np.ndarray],
        ) -> Union[float, np.ndarray]:
            # Maximally useless, just return tpr = fpr
            if isinstance(fpr, np.ndarray):
                return fpr.copy()
            else:
                return fpr

        tm_metadata = {}

    def uncorrelated_roc_function(
        t: Union[float, np.ndarray],
        fpr_t: Union[float, np.ndarray],
    ) -> Union[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
        """
        Given target (tpr+fpr)/2 values and fpr_t values, return corresponding (tpr, fpr) pairs.

        Args:
            t: Target sum float or array
            fpr_t: Target blue team FPR float or array
        Returns:
            Tuple of (tpr, fpr) floats or arrays with same shape as input
        """

        tpr_u, fpr_u = um_roc_function(t)
        tpr_t = tm_roc_function(fpr_t)

        # Assume uncorrelated monitors. Attack is caught if tm has true positive,
        # or um has false negative (i.e. UM fails to collude)
        # Want to return p_attack_passes = 1-p_attack_caught
        p_attack_caught = tpr_t + (1 - tpr_u) - tpr_t * (1 - tpr_u)

        return 1 - p_attack_caught, fpr_u  # type: ignore

    uncorrelated_roc_function.type = "uncorrelated"

    metadata = {
        **{f"untrusted_{k}": v for k, v in um_metadata.items()},
        **{f"trusted_{k}": v for k, v in tm_metadata.items()},
    }

    return uncorrelated_roc_function, metadata


def auc_roc_from_confusion_matrix(
    confusion_matrix: dict[str, int],
) -> tuple[float, dict[str, list[float]]]:
    """Calculate AUC-ROC from label counts.

    Args:
        confusion_matrix: Dictionary with label counts (TP, FP, TN, FN)

    Returns:
        AUC-ROC score
    """
    assert confusion_matrix.keys() == {
        "tp",
        "fp",
        "tn",
        "fn",
    }, "Label count dict must have keys for tp, fp, tn, fn"
    tp = confusion_matrix.get("tp", 0)
    fp = confusion_matrix.get("fp", 0)
    tn = confusion_matrix.get("tn", 0)
    fn = confusion_matrix.get("fn", 0)

    # Calculate separation of gaussians with this distribution
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    assert tnr + tpr > 1.0, "Your classifier is worse than useless!"
    # Inverse gaussian CDF on both probabilities
    # Inverse gaussian CDF on both probabilities
    threshold_negative = stats.norm.ppf(tnr)
    threshold_positive = stats.norm.ppf(tpr)

    # Calculate separation between the distributions
    separation = threshold_positive + threshold_negative

    # Calculate AUC-ROC from separation (using the inverse of the formula in generate_tpr_and_fpr_values)
    auc_roc = float(stats.norm.cdf(separation / np.sqrt(2)))

    return auc_roc, {"tpr": [tpr], "fpr": [1 - tnr]}


def calculate_tpr_fpr_from_scores(
    positive_scores: list[float],
    negative_scores: list[float],
) -> tuple[list[float], list[float]]:
    """Calculate TPR and FPR values from positive and negative scores.

    Args:
        positive_scores: Scores for positive samples (ground truth = 1)
        negative_scores: Scores for negative samples (ground truth = 0)
        num_thresholds: Number of threshold points to evaluate

    Returns:
        Tuple of (tpr_values, fpr_values) lists
    """
    all_scores = np.array(positive_scores + negative_scores)
    labels = np.concatenate(
        [np.ones(len(positive_scores)), np.zeros(len(negative_scores))]
    )

    # Create thresholds from min to max score
    thresholds = np.sort(list(set(positive_scores)) + [np.inf])[::-1]

    tpr_values = []
    fpr_values = []

    for threshold in thresholds:
        predictions = (all_scores >= threshold).astype(int)

        # Calculate confusion matrix components
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        # Calculate TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_values.append(float(tpr))
        fpr_values.append(float(fpr))

    # Epic leetcode interview question: make the ROC curve convex
    convex = False
    while not convex:
        convex = True
        for i in range(1, len(tpr_values) - 1):
            # Check if the point is below the line connecting its neighbors
            x0, y0 = fpr_values[i - 1], tpr_values[i - 1]
            x1, y1 = fpr_values[i + 1], tpr_values[i + 1]
            x, y = fpr_values[i], tpr_values[i]

            # Calculate the slope of the line segment
            if x1 != x0:
                slope = (y1 - y0) / (x1 - x0)
                expected_y = y0 + slope * (x - x0)

                if y < expected_y:
                    # Point is below the line, remove it
                    del fpr_values[i]
                    del tpr_values[i]

                    convex = False

                    break

    return tpr_values, fpr_values


def generate_tpr_and_fpr_values(
    auc_roc: float,
    num_values: int = 100,
) -> tuple[list[float], list[float]]:
    """Using normal distributions, generate tpr and fpr values"""

    mu_pos = (np.sqrt(2) * stats.norm.ppf(auc_roc)).item()

    x = np.linspace(min(mu_pos - 4, 0), max(mu_pos, 4), num_values)
    x = np.sort(x)[::-1]

    tpr = 1 - stats.norm.cdf(x, loc=mu_pos, scale=1)
    fpr = 1 - stats.norm.cdf(x, loc=0, scale=1)

    return tpr.tolist(), fpr.tolist()


def build_t_parameterized_roc_function(
    auc_roc: Optional[float],
    positive_rate_values: Optional[dict[str, list[float]]],
    confidence_scores: Optional[dict[str, list[float]]],
    confusion_matrix: Optional[dict[str, int]],
) -> tuple[TParameterizedROCFunctionType, dict[str, Any]]:
    # Determine which input method was used and get TPR/FPR values
    if positive_rate_values is not None:
        assert positive_rate_values.keys() == {
            "tpr",
            "fpr",
        }, "If providing positive rate values, must provide both TPR and FPR"
        # Method 1: Direct TPR/FPR values
        tpr_values = positive_rate_values["tpr"]
        fpr_values = positive_rate_values["fpr"]
    elif confidence_scores is not None:
        assert confidence_scores.keys() == {
            "positive",
            "negative",
        }, "If providing scores, must provide both positive and negative scores"
        # Method 2: Calculate TPR/FPR from positive/negative scores
        tpr_values, fpr_values = calculate_tpr_fpr_from_scores(
            confidence_scores["positive"],
            confidence_scores["negative"],
        )
    elif auc_roc is not None:
        # Method 3: Generate TPR/FPR from AUC ROC
        tpr_values, fpr_values = generate_tpr_and_fpr_values(auc_roc)
    elif confusion_matrix is not None:
        # Method 4: Generate from tp, fp, tn, fn counts
        auc_roc, positive_rate_values = auc_roc_from_confusion_matrix(confusion_matrix)
        tpr_values, fpr_values = generate_tpr_and_fpr_values(auc_roc)
        # In this case, we have one score
    else:
        raise ValueError(
            "Must provide either: "
            "(1) red_team_auc_roc, "
            "(2) red_team_tpr_values and red_team_fpr_values, or "
            "(3) positive_scores and negative_scores"
        )

    roc_fn_metadata = {
        "auc_roc": auc_roc,
        "tpr_values": tpr_values,
        "fpr_values": fpr_values,
        "confidence_scores": confidence_scores,
        "confusion_matrix": confusion_matrix,
    }

    # Ensure arrays are 1D
    tpr = np.atleast_1d(np.array(tpr_values))
    fpr = np.atleast_1d(np.array(fpr_values))

    # Check if we need to prepend (0,0)
    needs_origin = not (np.isclose(tpr[0], 0) & np.isclose(fpr[0], 0))
    if needs_origin:
        tpr = np.concatenate([np.array([0.0]), tpr])
        fpr = np.concatenate([np.array([0.0]), fpr])

    # Check if we need to append (1,1)
    needs_endpoint = not (np.isclose(tpr[-1], 1) & np.isclose(fpr[-1], 1))
    if needs_endpoint:
        tpr = np.concatenate([tpr, np.array([1.0])])
        fpr = np.concatenate([fpr, np.array([1.0])])

    # Compute the sum values for each point
    average_values = (tpr + fpr) / 2.0
    half_diff_values = (tpr - fpr) / 2.0

    # For inverse interpolation, we need sum_values to be monotonically increasing
    # In case of ties, we'll use the first occurrence (implicitly handled by searchsorted)
    def inverse_interpolate(
        average: Union[float, np.ndarray],
    ) -> Union[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
        """
        Given target (tpr+fpr)/2 values, return corresponding (tpr, fpr) pairs.

        Args:
            average: Target sum float or array

        Returns:
            Tuple of (tpr, fpr) floats or arrays with same shape as input
        """
        # Convert input to array for consistent processing
        average_array = np.asarray(average)
        original_shape = average_array.shape
        is_scalar = average_array.ndim == 0

        # Work with flattened array for vectorized operations
        if is_scalar:
            average_flat = average_array.reshape(1)
        else:
            average_flat = average_array.ravel()

        # Find indices for interpolation using searchsorted
        # searchsorted with 'right' gives us the index where we'd insert to maintain order
        # We want the segment [i-1, i] for interpolation
        indices = np.searchsorted(average_values, average_flat, side="right")

        # Clamp indices to valid range for interpolation
        indices = np.clip(indices, 1, len(average_values) - 1)

        # Get the surrounding points for the queries
        average_0 = average_values[indices - 1]
        average_1 = average_values[indices]
        half_diff_0 = half_diff_values[indices - 1]
        half_diff_1 = half_diff_values[indices]

        # Vectorized linear interpolation
        t = (average_flat - average_0) / (average_1 - average_0)

        half_diff = t * (half_diff_1 - half_diff_0) + half_diff_0

        tpr_flat = average_flat + half_diff
        fpr_flat = average_flat - half_diff

        # Reshape back to original shape
        if is_scalar:
            return tpr_flat.item(), fpr_flat.item()
        else:
            tpr_result = tpr_flat.reshape(original_shape)
            fpr_result = fpr_flat.reshape(original_shape)
            return tpr_result, fpr_result

    x_points = np.linspace(0, 1, 100)
    tpr, fpr = inverse_interpolate(x_points)  # type: ignore
    auc_roc = float(np.trapezoid(tpr, fpr))

    docstring = inverse_interpolate.__doc__
    docstring = (
        f"{docstring}"
        f"\n\nAUC-ROC of this ROC curve is {auc_roc:.3g}"
        f"\n\n<auc_roc>{auc_roc}</auc_roc>"
    )
    inverse_interpolate.__doc__ = docstring

    inverse_interpolate.type = "untrusted"

    return inverse_interpolate, roc_fn_metadata


def build_ensemble_roc_function(
    both_monitors_scores: dict[str, np.ndarray], n_grid: int = 100
) -> tuple[DoubleROCFunctionType, dict[str, Any]]:
    """
    Build a fast interpolated function for the disjunctive classifier.

    Args:
        A_scores: (n_A, 2) array with columns [s_T, s_U]
        H_scores: (n_H,) array of s_U values
        B_scores: (n_B,) array of s_T values
        n_grid: number of grid points for each dimension

    Returns:
        Function f(t, FPR_T) -> (FPR_U, P(-ve_U OR +ve_T|A))
    """

    A_scores = both_monitors_scores["A_scores"]
    H_scores = both_monitors_scores["H_scores"]
    B_scores = both_monitors_scores["B_scores"]

    # Get unique thresholds for both monitors
    # Set thresholds at the negative scores, plus -inf, use > and <= for comparisons
    unique_thresholds_U = np.unique(np.concatenate([[-np.inf], H_scores]))
    unique_thresholds_T = np.unique(np.concatenate([[-np.inf], B_scores]))

    scores = []

    for threshold_U, threshold_T in product(unique_thresholds_U, unique_thresholds_T):
        tpr_u = np.mean(A_scores[:, 1] > threshold_U)
        fpr_u = np.mean(H_scores > threshold_U)

        # T gives a positive score
        fpr_t = np.mean(B_scores > threshold_T)

        # Attack passes if U fails to detect collusion
        # OR T flags as positive
        attack_passes_chance = 1 - np.mean(
            (A_scores[:, 1] <= threshold_U) | (A_scores[:, 0] > threshold_T)
        )

        t_u = (tpr_u + fpr_u) / 2
        # Inputs: t_u (red team), fpr_t (blue team)
        # Outputs: attack_passes_chance, fpr_u (for honeypots)
        scores.append((t_u, fpr_t, attack_passes_chance, fpr_u))

        # fpr_u should be increasing and concave in t_u for fixed fpr_t
        # attack_passes_chance should be decreasing and convex in fpr_t for fixed t_u

    # Build constrained 2D interpolators from the collected tuples.
    # 1) Gather unique axes
    scores_arr = np.array(scores, dtype=float)
    t_unique = np.unique(scores_arr[:, 0])
    fpr_T_unique = np.unique(scores_arr[:, 1])

    # Optionally subsample very dense axes
    if len(t_unique) > n_grid:
        idx = np.linspace(0, len(t_unique) - 1, n_grid).astype(int)
        t_unique = t_unique[idx]
    if len(fpr_T_unique) > n_grid:
        idx = np.linspace(0, len(fpr_T_unique) - 1, n_grid).astype(int)
        fpr_T_unique = fpr_T_unique[idx]

    # 2) Aggregate into full grids (mean if duplicates land on same cell)
    tu_to_i = {v: i for i, v in enumerate(t_unique)}
    ft_to_j = {v: j for j, v in enumerate(fpr_T_unique)}

    pass_grid = np.full((len(t_unique), len(fpr_T_unique)), np.nan, dtype=float)
    pass_cnt = np.zeros_like(pass_grid)
    fpru_grid = np.full_like(pass_grid, np.nan)
    fpru_cnt = np.zeros_like(pass_grid)

    for t_u, fpr_t, attack_passes_chance, fpr_u in scores:
        # Skip samples that fell outside after subsampling axes
        if t_u not in tu_to_i or fpr_t not in ft_to_j:
            continue
        i = tu_to_i[t_u]
        j = ft_to_j[fpr_t]
        if np.isnan(pass_grid[i, j]):
            pass_grid[i, j] = 0.0
            fpru_grid[i, j] = 0.0
        pass_grid[i, j] += attack_passes_chance
        pass_cnt[i, j] += 1.0
        fpru_grid[i, j] += fpr_u
        fpru_cnt[i, j] += 1.0

    # Finalize means; fill any accidental NaNs via nearest neighbor of existing data
    with np.errstate(invalid="ignore"):
        pass_grid = np.divide(pass_grid, np.maximum(pass_cnt, 1))
        fpru_grid = np.divide(fpru_grid, np.maximum(fpru_cnt, 1))

    # Fill NaNs (should be rare) using simple forward/backward fill then mean
    def _fill_nans(mat: np.ndarray) -> np.ndarray:
        m = mat.copy()
        # forward along rows
        for i in range(m.shape[0]):
            row = m[i]
            mask = np.isnan(row)
            if mask.all():
                continue
            # forward fill
            last = None
            for k in range(len(row)):
                if not np.isnan(row[k]):
                    last = row[k]
                elif last is not None:
                    row[k] = last
            # backward fill
            last = None
            for k in range(len(row) - 1, -1, -1):
                if not np.isnan(row[k]):
                    last = row[k]
                elif last is not None:
                    row[k] = last
        # then columns
        for j in range(m.shape[1]):
            col = m[:, j]
            mask = np.isnan(col)
            if mask.all():
                continue
            last = None
            for k in range(len(col)):
                if not np.isnan(col[k]):
                    last = col[k]
                elif last is not None:
                    col[k] = last
            last = None
            for k in range(len(col) - 1, -1, -1):
                if not np.isnan(col[k]):
                    last = col[k]
                elif last is not None:
                    col[k] = last
        # any remaining NaNs -> zeros
        m = np.nan_to_num(m, nan=0.0)
        return m

    pass_grid = _fill_nans(pass_grid)
    fpru_grid = _fill_nans(fpru_grid)

    # 3) Enforce axis-wise monotone/curvature constraints
    def _pava(y: np.ndarray, w: np.ndarray, increasing: bool) -> np.ndarray:
        """Weighted PAVA returning fitted values per element.
        Enforces nondecreasing when increasing=True, nonincreasing otherwise."""
        if y.size == 0:
            return y
        yy = y.astype(float)
        if not increasing:
            yy = -yy
        v: list[float] = []  # block means
        vw: list[float] = []  # block weights
        cnt: list[int] = []  # block lengths (in elements)
        for i in range(len(yy)):
            v.append(yy[i])
            vw.append(float(w[i]))
            cnt.append(1)
            # merge adjacent violating blocks
            while len(v) >= 2 and v[-2] > v[-1]:
                total_w = vw[-2] + vw[-1]
                mean = (vw[-2] * v[-2] + vw[-1] * v[-1]) / total_w
                v[-2] = mean
                vw[-2] = total_w
                cnt[-2] += cnt[-1]
                v.pop()
                vw.pop()
                cnt.pop()
        # expand to per-element fitted values
        y_hat = np.empty_like(yy)
        pos = 0
        for mean, length in zip(v, cnt):
            y_hat[pos : pos + length] = mean
            pos += length
        if not increasing:
            y_hat = -y_hat
        return y_hat

    def _enforce_concave_increasing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Enforce y(x) to be concave and non-decreasing over x."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) <= 2:
            y_mono = np.maximum.accumulate(y)
            return np.clip(y_mono, 0.0, 1.0)
        dx = np.diff(x)
        dy = np.diff(y)
        slopes = dy / dx
        # Concave: slopes are non-increasing; Increasing: slopes >= 0
        slopes_fit = _pava(slopes, dx, increasing=False)
        slopes_fit = np.maximum(slopes_fit, 0.0)
        y_fit = np.empty_like(y, dtype=float)
        y_fit[0] = y[0]
        for i in range(len(slopes_fit)):
            y_fit[i + 1] = y_fit[i] + slopes_fit[i] * dx[i]
        return np.clip(y_fit, 0.0, 1.0)

    def _enforce_convex_decreasing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Enforce y(x) to be convex and non-increasing over x."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) <= 2:
            y_mono = np.minimum.accumulate(y)
            return np.clip(y_mono, 0.0, 1.0)
        dx = np.diff(x)
        dy = np.diff(y)
        slopes = dy / dx
        # Convex: slopes are non-decreasing; Decreasing: slopes <= 0
        slopes_fit = _pava(slopes, dx, increasing=True)
        slopes_fit = np.minimum(slopes_fit, 0.0)
        y_fit = np.empty_like(y, dtype=float)
        y_fit[0] = y[0]
        for i in range(len(slopes_fit)):
            y_fit[i + 1] = y_fit[i] + slopes_fit[i] * dx[i]
        return np.clip(y_fit, 0.0, 1.0)

    """
    # Enforce FPR_U: increasing and concave in t (per column over t_unique)
    for j in range(fpru_grid.shape[1]):
        fpru_grid[:, j] = _enforce_concave_increasing(t_unique, fpru_grid[:, j])
    # Enforce attack_passes: decreasing and convex in fpr_T (per row over fpr_T_unique)
    for i in range(pass_grid.shape[0]):
        pass_grid[i, :] = _enforce_convex_decreasing(fpr_T_unique, pass_grid[i, :])
    """

    # 4) Build interpolators
    fpr_U_interp = RegularGridInterpolator(
        (t_unique, fpr_T_unique), fpru_grid, method="linear", bounds_error=False
    )
    pass_interp = RegularGridInterpolator(
        (t_unique, fpr_T_unique), pass_grid, method="linear", bounds_error=False
    )

    # Return the combined function mapping (t_u, fpr_t) -> (attack_passes_chance, fpr_u)
    def roc_function(
        t: Union[float, np.ndarray], fpr_T: Union[float, np.ndarray]
    ) -> Union[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
        """
        Compute constrained 2D interpolation values for given parameters.

        Args:
            t: scalar or array, parameterization for U: (TPR_U + FPR_U) / 2
            fpr_T: scalar or array, false positive rate for T = P(+ve_T|B)

        Returns:
            tuple: (attack_passes_chance, FPR_U)
        """
        # Handle scalar/array broadcasting
        if np.isscalar(t) and np.isscalar(fpr_T):
            points = np.array([[t, fpr_T]])
        else:
            t = np.atleast_1d(t)
            fpr_T = np.atleast_1d(fpr_T)
            if len(t) == 1:
                t = np.repeat(t, len(fpr_T))
            elif len(fpr_T) == 1:
                fpr_T = np.repeat(fpr_T, len(t))
            points = np.column_stack([t, fpr_T])

        # Clip to valid range
        points[:, 0] = np.clip(points[:, 0], t_unique[0], t_unique[-1])
        points[:, 1] = np.clip(points[:, 1], fpr_T_unique[0], fpr_T_unique[-1])

        fpr_u_values = fpr_U_interp(points)
        attack_pass_values = pass_interp(points)

        """
        # Corner overrides: treat (t<=0, fpr_T<=0) as (attack_pass=1, fpr_u=1);
        # and (t>=1, fpr_T>=1) as (attack_pass=0, fpr_u=0)
        corner00 = (points[:, 0] <= 0.0) & (points[:, 1] <= 0.0)
        corner11 = (points[:, 0] >= 1.0) & (points[:, 1] >= 1.0)
        if np.any(corner00):
            fpr_u_values[corner00] = 1.0
            attack_pass_values[corner00] = 1.0
        if np.any(corner11):
            fpr_u_values[corner11] = 0.0
            attack_pass_values[corner11] = 0.0
        # Return scalars if input was scalar
        if points.shape[0] == 1 and np.isscalar(t) and np.isscalar(fpr_T):
            return float(attack_pass_values[0]), float(fpr_u_values[0])
        """

        return attack_pass_values, fpr_u_values

    # Add useful attributes to the function
    roc_function.t_range = (t_unique[0], t_unique[-1])
    roc_function.fpr_T_range = (fpr_T_unique[0], fpr_T_unique[-1])
    roc_function.grid_points = (t_unique, fpr_T_unique)
    roc_function.type = "ensemble"

    return roc_function, {
        "untrusted_confidence_scores": {
            "positive": A_scores[:, 1].tolist(),
            "negative": H_scores.tolist(),
        },
        "trusted_suspiciousness_scores": {
            "positive": A_scores[:, 0].tolist(),
            "negative": B_scores.tolist(),
        },
    }


def build_fpr_parameterized_roc_function(
    scores: dict[str, list[float]],
) -> tuple[FPRParameterizedROCFunctionType, dict[str, Any]]:
    """
    Build an interpolator for ROC curves parameterized by fpr.

    Args:
        tpr: True positive rates (sorted)
        fpr: False positive rates (sorted)

    Returns:
        A function that maps fpr values to tpr values.
    """

    tpr, fpr = calculate_tpr_fpr_from_scores(
        scores["positive"],
        scores["negative"],
    )

    # Ensure arrays are 1D float arrays
    tpr = np.atleast_1d(np.asarray(tpr, dtype=float))
    fpr = np.atleast_1d(np.asarray(fpr, dtype=float))

    if tpr.shape != fpr.shape:
        raise ValueError("tpr and fpr must have the same shape")

    # Optionally prepend (0,0) and append (1,1) if missing
    if not (np.isclose(fpr[0], 0) and np.isclose(tpr[0], 0)):
        fpr = np.concatenate([np.array([0.0]), fpr])
        tpr = np.concatenate([np.array([0.0]), tpr])
    if not (np.isclose(fpr[-1], 1) and np.isclose(tpr[-1], 1)):
        fpr = np.concatenate([fpr, np.array([1.0])])
        tpr = np.concatenate([tpr, np.array([1.0])])

    # Sort by fpr ascending
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    # Collapse duplicate fpr by taking the maximum tpr at that fpr
    unique_fpr, inv = np.unique(fpr_sorted, return_inverse=True)
    tpr_max = np.full_like(unique_fpr, -np.inf, dtype=float)
    np.maximum.at(tpr_max, inv, tpr_sorted)

    # Enforce monotonic non-decreasing tpr with increasing fpr
    tpr_mono = np.maximum.accumulate(tpr_max)

    # Build the interpolator using linear interpolation; clamp outside [0,1]
    def interpolate(query_fpr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        q = np.asarray(query_fpr, dtype=float)
        is_scalar = q.ndim == 0
        if is_scalar:
            q = q.reshape(1)

        # np.interp clamps to edge values for out-of-range queries
        tpr_vals = np.interp(q, unique_fpr, tpr_mono)

        if is_scalar:
            return float(tpr_vals.item())
        return tpr_vals

    # Attach AUC-ROC information to the docstring for convenience
    auc_roc = float(np.trapezoid(tpr_mono, unique_fpr))
    doc = interpolate.__doc__ or ""
    doc += (
        "\n\nInterpolates TPR from FPR along the ROC curve using linear interpolation."
        f"\nAUC-ROC of this ROC curve is {auc_roc:.3g}"
        f"\n\n<auc_roc>{auc_roc}</auc_roc>"
    )
    interpolate.__doc__ = doc

    metadata = {
        "tpr_values": tpr_mono.tolist(),
        "fpr_values": unique_fpr.tolist(),
        "auc_roc": auc_roc,
        "suspiciousness_scores": scores,
    }

    return interpolate, metadata
