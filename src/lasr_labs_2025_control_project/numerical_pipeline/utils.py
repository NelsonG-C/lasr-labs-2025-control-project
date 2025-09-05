### Utility functions for numerical pipeline

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass
class OptimizationConfig:
    """Configuration class for optimize_parameters runs.

    Contains all necessary information for running parameter optimization,
    including blue team budget, number of tasks, and red team ROC function.
    """

    num_tasks: float
    blue_team_budget_fraction: float
    red_team_roc_fn: Callable[
        [Union[float, np.ndarray]],
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]],
    ]
    blue_team_win_calculator: Callable[[Dict[str, np.ndarray]], np.ndarray]

    # Optional data for reference
    red_team_positive_rate_values: Optional[Dict[str, List[float]]] = None
    red_team_confidence_scores: Optional[Dict[str, List[float]]] = None
    red_team_confusion_matrix: Optional[Dict[str, int]] = None

    def __repr__(self):
        return (
            f"OptimizationConfig("
            f"Number of Tasks: {self.num_tasks}, "
            f"Blue Team Budget: {self.blue_team_budget_fraction}, "
            f")"
        )


@dataclass
class OptimizationResult:
    """Result class containing the best optimization results."""

    best_parameters: Dict[str, float]
    best_score: float
    history: List[Dict[str, float]]
    config: OptimizationConfig

    def __repr__(self):
        tpr, fpr = self.config.red_team_roc_fn(
            self.best_parameters["red_team_roc_threshold"]
        )

        border = f"{'*' * 50}"

        # Extract AUC-ROC value from docstring
        docstring = self.config.red_team_roc_fn.__doc__ or ""
        auc_match = re.search(r"<auc_roc>(.*?)</auc_roc>", docstring)
        auc_roc = float(auc_match.group(1)) if auc_match else "N/A"

        assessment = get_assessment(self.best_score)

        return (
            f"{border}\n"
            f"\n"
            f"Optimization Result\n"
            f"\n"
            f"Blue Team Win Rate: {self.best_score:.3g}\n"
            f"Assessment: {assessment}\n"
            f"\n"
            f"Parameters:\n"
            f"\tBlue Team Honeypot Fraction: {self.best_parameters['blue_team_honeypot_fraction']:.3g}\n"
            f"\tRed Team TPR: {tpr:.3g}\n"
            f"\tRed Team FPR: {fpr:.3g}\n"
            f"\tRed Team Attack Frequency: {self.best_parameters['red_team_p_attack']:.3g}\n"
            f"\n"
            f"Fixed Parameters:\n"
            f"\tNumber of Tasks: {self.config.num_tasks}\n"
            f"\tBlue Team Budget: {self.config.blue_team_budget_fraction:.3g}\n"
            f"\tRed Team AUC-ROC: {auc_roc:.3g}\n"
            f"\n"
            f"{border}"
        )


def build_optimization_config(
    num_tasks: float,
    blue_team_budget_fraction: float,
    red_team_auc_roc: Optional[float] = None,
    red_team_positive_rate_values: Optional[Dict[str, List[float]]] = None,
    red_team_scores: Optional[Dict[str, List[float]]] = None,
    red_team_confusion_matrix: Optional[Dict[str, int]] = None,
) -> OptimizationConfig:
    """Build an OptimizationConfig from various input formats.

    Args:
        num_tasks: Number of tasks in the simulation
        blue_team_budget_fraction: Blue team honeypot + audit budget fraction
        red_team_auc_roc: AUC ROC score (option 1)
        red_team_tpr_values: TPR values (option 2a)
        red_team_fpr_values: FPR values (option 2b)
        positive_scores: Positive class scores (option 3a)
        negative_scores: Negative class scores (option 3b)

    Returns:
        OptimizationConfig with all necessary functions and data
    """

    # Determine which input method was used and get TPR/FPR values
    if red_team_positive_rate_values is not None:
        assert red_team_positive_rate_values.keys() == {"tpr", "fpr"}, (
            "If providing positive rate values, must provide both TPR and FPR"
        )
        # Method 1: Direct TPR/FPR values
        tpr_values = red_team_positive_rate_values["tpr"]
        fpr_values = red_team_positive_rate_values["fpr"]
    elif red_team_scores is not None:
        assert red_team_scores.keys() == {"positive", "negative"}, (
            "If providing red team scores, must provide both positive and negative scores"
        )
        # Method 2: Calculate TPR/FPR from positive/negative scores
        tpr_values, fpr_values = calculate_tpr_fpr_from_scores(
            red_team_scores["positive"], red_team_scores["negative"]
        )
    elif red_team_auc_roc is not None:
        # Method 3: Generate TPR/FPR from AUC ROC
        tpr_values, fpr_values = generate_tpr_and_fpr_values(red_team_auc_roc)
    elif red_team_confusion_matrix is not None:
        # Method 4: Generate from tp, fp, tn, fn counts
        auc_roc, red_team_positive_rate_values = auc_roc_from_confusion_matrix(
            red_team_confusion_matrix
        )
        tpr_values, fpr_values = generate_tpr_and_fpr_values(auc_roc)
        # In this case, we have one score
    else:
        raise ValueError(
            "Must provide either: "
            "(1) red_team_auc_roc, "
            "(2) red_team_tpr_values and red_team_fpr_values, or "
            "(3) positive_scores and negative_scores"
        )

    # Build ROC function
    red_team_roc_fn = build_roc_inverse_interpolator(
        tpr=np.array(tpr_values), fpr=np.array(fpr_values)
    )

    # Build blue team win calculator
    blue_team_win_calculator = build_blue_team_win_calculator(
        num_tasks=num_tasks,
        blue_team_budget_fraction=blue_team_budget_fraction,
        red_team_roc_function=red_team_roc_fn,
    )

    return OptimizationConfig(
        num_tasks=num_tasks,
        blue_team_budget_fraction=blue_team_budget_fraction,
        red_team_roc_fn=red_team_roc_fn,
        blue_team_win_calculator=blue_team_win_calculator,
        red_team_positive_rate_values=red_team_positive_rate_values,
        red_team_confusion_matrix=red_team_confusion_matrix,
    )


def optimize_parameters(
    config: OptimizationConfig,
    max_iterations: int = 100,
    grid_squares_per_iteration: int = 10,
) -> OptimizationResult:
    """Run parameter optimization using the provided configuration.

    Args:
        config: OptimizationConfig containing all necessary data and functions
        max_iterations: Maximum optimization iterations
        grid_squares_per_iteration: Grid points per iteration

    Returns:
        Optimization results dictionary
    """

    parameter_spaces = {
        "blue_team_honeypot_fraction": {
            "min": 0.0,
            "max": 1.0,
            "scale": "linear",
            "ascend_or_descend": "ascend",
        },
        "red_team_roc_threshold": {
            "min": 0.0,
            "max": 1.0,
            "scale": "linear",
            "ascend_or_descend": "descend",
        },
        "red_team_p_attack": {
            "min": 1e-6,
            "max": 1e-3,
            "scale": "log",
            "ascend_or_descend": "descend",
        },
    }

    results = grid_search(
        parameter_spaces,
        config.blue_team_win_calculator,
        max_iterations=max_iterations,
        grid_squares_per_iteration=grid_squares_per_iteration,
        tolerance=1e-9,
    )

    return OptimizationResult(**results, config=config)  # type: ignore


def build_blue_team_win_calculator(
    num_tasks: float,
    blue_team_budget_fraction: float,
    red_team_roc_function: Callable[
        [Union[float, np.ndarray]],
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]],
    ],
) -> Callable[[Dict[str, np.ndarray]], np.ndarray]:
    """Build a win probability calculator for the blue team

    Args:
        num_tasks: Number of tasks in the simulation
        blue_team_budget_fraction: Blue team honeypot + audit budget
        red_team_roc_function: Red team ROC function
    """

    def calculate_blue_team_win(
        params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        blue_team_honeypot_fraction = params["blue_team_honeypot_fraction"]
        red_team_roc_threshold = params["red_team_roc_threshold"]
        red_team_p_attack = params["red_team_p_attack"]

        true_positive_rate, false_positive_rate = red_team_roc_function(
            red_team_roc_threshold
        )

        blue_team_budget = int(blue_team_budget_fraction * num_tasks)
        num_honeypots = int(blue_team_budget * blue_team_honeypot_fraction)
        num_audits = blue_team_budget - num_honeypots

        # Simulate honeypots
        """
        # Calculate the probability of red team passing honeypots
        # Binomial 0-4 is sum
        # n_honeypots, n_honeypots-1, n_honeypots-2 ...
        fpr_powers = num_honeypots - np.arange(0, 5)
        tpr_powers = np.arange(0, 5)
        # 1, n_honeypots, n_honeypots*(n_honeypots-1), ...
        numerators = np.cumprod(num_honeypots - np.arange(-1, 4)) / (
            num_honeypots + 1
        )
        # Factorials for the denominators
        denominators = np.array([1, 1, 2, 6, 24])

        binomial_probs = (
            true_positive_rate**tpr_powers
            * false_positive_rate**fpr_powers
            * numerators
            / denominators
        )
        p_red_team_fails_honeypots = 1 - np.sum(binomial_probs)
        """
        p_red_team_fails_honeypots = 1 - np.sum(
            np.stack(
                [
                    stats.binom.pmf(
                        k=k, n=int(np.round(num_honeypots)), p=false_positive_rate
                    )
                    for k in range(5)
                ]
            ),
            axis=0,
        )

        # Probability of returning to honeypots
        p_go_back_to_honeypots = (1 - red_team_p_attack) ** num_tasks

        # Probability of attack just being caught

        # Calculate probability red team is caught
        # Audit the first few
        p_attack_caught = 1 - (1 - red_team_p_attack) ** num_audits
        # Otherwise hope for flag
        false_negative_rate = 1 - true_positive_rate
        p_attack_flagged = 1 - (1 - red_team_p_attack * false_negative_rate) ** (
            num_tasks - num_audits
        )

        p_attack_caught_or_flagged = p_attack_caught + p_attack_flagged

        # Use Claude's formula from the markov chain
        p_red_team_victory = (
            (1 - p_attack_caught_or_flagged - p_go_back_to_honeypots)
            * (1 - p_red_team_fails_honeypots)
            / (1 - (1 - p_red_team_fails_honeypots) * p_go_back_to_honeypots)
        )

        return 1 - p_red_team_victory

    return calculate_blue_team_win


def build_roc_inverse_interpolator(
    tpr: np.ndarray, fpr: np.ndarray
) -> Callable[
    [Union[float, np.ndarray]],
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray]],
]:
    """
    Build an inverse interpolator for ROC curves parameterized by (tpr+fpr)/2.

    Args:
        tpr: True positive rates (sorted)
        fpr: False positive rates (sorted)

    Returns:
        A jitted function that maps (tpr+fpr)/2 values to (tpr, fpr) pairs
    """
    # Ensure arrays are 1D
    tpr = np.atleast_1d(tpr)
    fpr = np.atleast_1d(fpr)

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
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
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

    return inverse_interpolate


def generate_tpr_and_fpr_values(
    auc_roc: float,
    num_values: int = 100,
) -> Tuple[List[float], List[float]]:
    """Using normal distributions, generate tpr and fpr values"""

    mu_pos = (np.sqrt(2) * stats.norm.ppf(auc_roc)).item()

    x = np.linspace(min(mu_pos - 4, 0), max(mu_pos, 4), num_values)
    x = np.sort(x)[::-1]

    tpr = 1 - stats.norm.cdf(x, loc=mu_pos, scale=1)
    fpr = 1 - stats.norm.cdf(x, loc=0, scale=1)

    return tpr.tolist(), fpr.tolist()


def grid_search(
    param_data: Dict[str, Dict[str, Union[str, float]]],
    function: Callable[[Dict[str, np.ndarray]], np.ndarray],
    max_iterations: int,
    grid_squares_per_iteration: int,
    tolerance: float,
) -> Dict[str, Union[float, Dict[str, float], List[Dict[str, float]]]]:
    """Adversarial coordinate grid search.

    For each parameter with a range [min, max], a scale (linear|log), and a direction
    (ascend|descend), iteratively performs a coordinate-wise search. At each iteration
    and for each parameter, we evaluate a small grid of candidate values while keeping
    the other parameters fixed to their current values, then pick the candidate that
    ascends or descends the objective according to the parameter's direction. The
    parameter's local search interval is then "zoomed in" around the chosen candidate.

    Args:
        param_data: Dict mapping parameter name -> {min, max, scale, ascend_or_descend}
        function: Objective function returning a scalar np.ndarray (higher is better)
        max_iterations: Maximum number of outer iterations (full passes over parameters)
        grid_squares_per_iteration: Number of grid points per parameter per update (>= 2)
        tolerance: Early-stop threshold on objective change

    Returns:
        Dict with keys:
            - best_value: float (objective at final params)
            - best_params: Dict[str, float]
            - history: List of dicts with per-iteration objective and parameters
    """
    # Initialize current parameters at the midpoint of each range
    current_params = {}
    param_ranges = {}

    for param_name, config in param_data.items():
        min_val = float(config["min"])
        max_val = float(config["max"])
        scale = config["scale"]

        # Store the current range for this parameter
        param_ranges[param_name] = {"min": min_val, "max": max_val}

        # Initialize at midpoint based on scale
        if scale == "log":
            current_params[param_name] = np.sqrt(min_val * max_val)
        else:  # linear
            current_params[param_name] = (min_val + max_val) / 2.0

    # Evaluate initial objective
    current_value = function(current_params).item()

    # Initialize history
    history = []
    history.append(
        {
            "iteration": 0,
            "objective": float(current_value),
            "params": {k: float(v) for k, v in current_params.items()},
        }
    )

    # Main optimization loop
    for iteration in range(max_iterations):
        previous_value = current_value

        # Coordinate-wise search over each parameter
        for param_name, config in param_data.items():
            scale = config["scale"]
            direction = config["ascend_or_descend"]

            # Get current range for this parameter
            min_val = param_ranges[param_name]["min"]
            max_val = param_ranges[param_name]["max"]

            # Generate grid points around current value
            if scale == "log":
                # For log scale, work in log space
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                log_candidates = np.linspace(
                    log_min, log_max, grid_squares_per_iteration
                )
                candidates = np.exp(log_candidates)
            else:  # linear
                candidates = np.linspace(min_val, max_val, grid_squares_per_iteration)

            # Evaluate function at each candidate while keeping other params fixed
            objectives = []
            for candidate in candidates:
                test_params = current_params.copy()
                test_params[param_name] = candidate
                obj_val = function(test_params).item()
                objectives.append(obj_val)

            objectives = np.array(objectives)

            # Choose best candidate based on direction
            if direction == "ascend":
                best_idx = np.argmax(objectives)
            else:  # descend
                best_idx = np.argmin(objectives)

            best_candidate = candidates[best_idx]
            best_objective = objectives[best_idx]

            # Update current parameters and value if better
            if (direction == "ascend" and best_objective > current_value) or (
                direction == "descend" and best_objective < current_value
            ):
                current_params[param_name] = best_candidate
                current_value = best_objective

            # Zoom in the search range around the best candidate
            range_width = max_val - min_val
            zoom_factor = 0.5  # Reduce range by half each iteration
            new_width = range_width * zoom_factor

            new_min = np.maximum(min_val, best_candidate - new_width / 2)
            new_max = np.minimum(max_val, best_candidate + new_width / 2)

            # Ensure we don't collapse the range too much
            if new_max - new_min < (max_val - min_val) * 1e-8:
                new_min = min_val
                new_max = max_val

            param_ranges[param_name] = {"min": float(new_min), "max": float(new_max)}

        # Record history
        history.append(
            {
                "iteration": iteration + 1,
                "objective": float(current_value),
                "params": {k: float(v) for k, v in current_params.items()},
            }
        )

        # Check for convergence
        if abs(current_value - previous_value) < tolerance:
            break

    return dict(
        best_score=float(current_value),
        best_parameters={k: float(v) for k, v in current_params.items()},
        history=history,
    )


def calculate_tpr_fpr_from_scores(
    positive_scores: List[float],
    negative_scores: List[float],
    num_thresholds: int = 100,
) -> Tuple[List[float], List[float]]:
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
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    thresholds = np.linspace(min_score, max_score, num_thresholds)

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

    return tpr_values, fpr_values


def auc_roc_from_confusion_matrix(
    confusion_matrix: Dict[str, int],
) -> Tuple[float, Dict[str, List[float]]]:
    """Calculate AUC-ROC from label counts.

    Args:
        confusion_matrix: Dictionary with label counts (TP, FP, TN, FN)

    Returns:
        AUC-ROC score
    """
    assert confusion_matrix.keys() == {"tp", "fp", "tn", "fn"}, (
        "Label count dict must have keys for tp, fp, tn, fn"
    )
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


def get_assessment(score: float) -> str:
    """Get a score assessment."""
    if score < 0.25:
        return "What kinda score is that bro we are gonna die."

    if score < 0.5:
        return "This is exactly what Yudkowsky warned us about."

    if score < 0.75:
        return "Marc Andreessen would probably deploy that."

    if score < 0.9:
        return "Maybe we can show this protocol to Buck."

    if score < 0.99:
        return "NeurIPS here we come!"

    return (
        "Did you solve AI Control or did you just use a dumb model, what happened here?"
    )
