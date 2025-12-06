"""
Copula-based independence testing using statsmodels and scipy.

Uses likelihood ratio tests with copulas for rigorous independence testing.
Implements proper LR test for Student-t copula (mixture distribution).

Installation:
    pip install statsmodels scipy pandas matplotlib

Usage:
    python copula_analysis.py --config path/to/config.yaml
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import rankdata
from statsmodels.distributions.copula.api import GaussianCopula, StudentTCopula

from lasr_labs_2025_control_project.utils.config_loader import (
    RunLabel,
    load_and_normalize_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CopulaResult:
    """Results from copula fitting and independence test."""

    copula_type: str
    pair_name: str
    correlation: float
    degrees_freedom: float = None
    log_likelihood: float = None
    log_likelihood_indep: float = None
    lr_statistic: float = None
    p_value: float = None
    is_independent: bool = None
    copula: object = None
    n_tied_x: int = 0
    n_tied_y: int = 0


def pseudo_obs(data: np.ndarray, ties_method: str = "average") -> np.ndarray:
    """
    Compute pseudo-observations for copula modeling.

    Transforms data to pseudo-observations (approximately uniform marginals)
    using the empirical distribution function scaled by n/(n+1).

    This is equivalent to the R copula/VineCopula pobs() function.

    Parameters
    ----------
    data : np.ndarray
        n x d matrix of observations
    ties_method : str
        Method for handling ties, passed to scipy.stats.rankdata.
        Options: 'average', 'min', 'max', 'dense', 'ordinal'

    Returns
    -------
    np.ndarray
        n x d matrix of pseudo-observations in (0, 1)
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n = data.shape[0]
    result = np.zeros_like(data, dtype=float)

    for j in range(data.shape[1]):
        # Rank the data (1 to n)
        ranks = rankdata(data[:, j], method=ties_method)
        # Scale to (0, 1) using n+1 to avoid boundary issues
        result[:, j] = ranks / (n + 1)

    return result


def load_scores(eval_log_path: Path) -> pd.DataFrame:
    """Load all three scores from eval log."""
    log = read_eval_log(eval_log_path)
    samples = []
    for sample in log.samples:  # pyright: ignore
        meta = {**sample.metadata, **sample.store}
        samples.append(
            {
                "uc": meta["untrusted_collusion_score"],
                "ts": meta["trusted_suspiciousness_score"],
                "us": meta["untrusted_suspiciousness_score"],
            }
        )
    return pd.DataFrame(samples)


def count_tied_datapoints(x: np.ndarray) -> int:
    """
    Count the number of datapoints involved in ties.

    Example: [1, 2, 2, 3, 3, 3, 4] has 5 tied datapoints (two 2s, three 3s)

    Returns:
        Number of datapoints that have at least one duplicate value
    """
    _, counts = np.unique(x, return_counts=True)
    return int(np.sum(counts[counts > 1]))


def check_ties(x: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
    """
    Check for ties in data.

    Returns:
        (n_tied_x, n_tied_y): Number of tied datapoints in each variable
    """
    return count_tied_datapoints(x), count_tied_datapoints(y)


def fit_gaussian_copula_and_test(
    data_uniform: np.ndarray, n_tied_x: int, n_tied_y: int
) -> CopulaResult:
    """
    Fit Gaussian copula and test independence using likelihood ratio test.

    H0: ρ = 0 (independence)
    H1: ρ ≠ 0 (dependence)

    LR statistic follows χ²(1) under H0.
    """
    # Fit correlation parameter from data
    copula_temp = GaussianCopula()
    rho_fitted = copula_temp.fit_corr_param(data_uniform)

    # Fitted copula
    copula_fitted = GaussianCopula(corr=rho_fitted, k_dim=2)
    log_lik_fitted = np.sum(copula_fitted.logpdf(data_uniform))

    # Independence copula (ρ = 0)
    copula_indep = GaussianCopula(corr=0.0, k_dim=2)
    log_lik_indep = np.sum(copula_indep.logpdf(data_uniform))

    # Likelihood ratio test
    lr_stat = 2 * (log_lik_fitted - log_lik_indep)
    # For Gaussian copula, LR ~ χ²(1) under H0
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return CopulaResult(
        copula_type="gaussian",
        pair_name="",
        correlation=float(rho_fitted),
        log_likelihood=float(log_lik_fitted),
        log_likelihood_indep=float(log_lik_indep),
        lr_statistic=float(lr_stat),
        p_value=float(p_value),
        is_independent=p_value > 0.05,  # pyright: ignore
        copula=copula_fitted,
        n_tied_x=n_tied_x,
        n_tied_y=n_tied_y,
    )


def fit_studentt_params(data_uniform: np.ndarray) -> Tuple[float, float]:
    """
    Fit Student-t copula parameters (rho, df) using maximum likelihood.

    Uses a robust two-stage approach:
    1. Grid search to find good starting point
    2. Refined optimization from best grid point with bounds
    """

    # Stage 1: Coarse grid search over (rho, df)
    rho_grid = np.linspace(-0.9, 0.9, 15)
    df_grid = np.concatenate(
        [np.linspace(2.5, 10, 10), np.linspace(12, 30, 5), [50, 100]]
    )

    best_loglik = -np.inf
    best_rho = 0.0
    best_df = 5.0

    logger.info("    Grid search for Student-t parameters...")
    for rho in rho_grid:
        for df_val in df_grid:
            try:
                copula = StudentTCopula(corr=rho, df=df_val, k_dim=2)
                loglik = np.sum(copula.logpdf(data_uniform))

                if np.isfinite(loglik) and loglik > best_loglik:
                    best_loglik = loglik
                    best_rho = rho
                    best_df = df_val
            except:
                continue

    logger.info(
        f"    Grid search result: ρ={best_rho:.3f}, ν={best_df:.1f}, loglik={best_loglik:.2f}"
    )

    # Stage 2: Refined optimization with bounds
    def neg_log_lik(params):  # pyright: ignore
        rho, df_val = params

        try:
            copula = StudentTCopula(corr=rho, df=df_val, k_dim=2)
            loglik = np.sum(copula.logpdf(data_uniform))

            if not np.isfinite(loglik):
                return 1e10

            return -loglik
        except:
            return 1e10

    logger.info("    Refining with L-BFGS-B optimization...")

    # Define bounds: rho in (-0.999, 0.999), df in (2.2, 200)
    bounds = [(-0.999, 0.999), (2.2, 200.0)]

    # Use L-BFGS-B which handles bounds efficiently
    result = minimize(
        neg_log_lik,
        x0=[best_rho, best_df],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    rho_opt, df_opt = result.x
    final_loglik = -result.fun

    logger.info(
        f"    Final result: ρ={rho_opt:.3f}, ν={df_opt:.1f}, loglik={final_loglik:.2f}"
    )

    return float(rho_opt), float(df_opt)


def fit_studentt_copula_and_test(
    data_uniform: np.ndarray, n_tied_x: int, n_tied_y: int
) -> CopulaResult:
    """
    Fit Student-t copula and test independence using likelihood ratio test.

    H0: ρ = 0 (independence)
    H1: ρ ≠ 0 (dependence)

    IMPORTANT: Under H0, the degrees of freedom parameter ν becomes
    unidentifiable. Following Genest & Nešlehová (2007), the proper
    LR test distribution is:

        LR ~ 0.5·χ²(1) + 0.5·χ²(2)

    This is a mixture distribution because H0 lies on the boundary
    of the parameter space.
    """
    # Fit both rho and df jointly under H1
    rho_fitted, df_fitted = fit_studentt_params(data_uniform)

    # Fitted copula (H1: ρ ≠ 0)
    copula_fitted = StudentTCopula(corr=rho_fitted, df=df_fitted, k_dim=2)
    log_lik_fitted = np.sum(copula_fitted.logpdf(data_uniform))

    # Independence copula (H0: ρ = 0)
    # Under H0, ν is unidentifiable, so we use the fitted ν
    # (This is a practical compromise, though theoretically not ideal)
    copula_indep = StudentTCopula(corr=0.0, df=df_fitted, k_dim=2)
    log_lik_indep = np.sum(copula_indep.logpdf(data_uniform))

    # Likelihood ratio test
    lr_stat = 2 * (log_lik_fitted - log_lik_indep)

    # Compute p-value using mixture distribution: 0.5·χ²(1) + 0.5·χ²(2)
    # P(LR > x) = 0.5·P(χ²(1) > x) + 0.5·P(χ²(2) > x)
    p_chi2_1 = 1 - stats.chi2.cdf(lr_stat, df=1)
    p_chi2_2 = 1 - stats.chi2.cdf(lr_stat, df=2)
    p_value = 0.5 * p_chi2_1 + 0.5 * p_chi2_2

    return CopulaResult(
        copula_type="student-t",
        pair_name="",
        correlation=float(rho_fitted),
        degrees_freedom=float(df_fitted),
        log_likelihood=float(log_lik_fitted),
        log_likelihood_indep=float(log_lik_indep),
        lr_statistic=float(lr_stat),
        p_value=float(p_value),
        is_independent=p_value > 0.05,  # pyright: ignore
        copula=copula_fitted,
        n_tied_x=n_tied_x,
        n_tied_y=n_tied_y,
    )


def test_independence(
    df: pd.DataFrame, col1: str, col2: str, pair_name: str, copula_type: str
) -> CopulaResult:
    """Fit copula and test independence using likelihood ratio test."""
    # Clean data
    clean_df = df[[col1, col2]].dropna()
    x = clean_df[col1].values
    y = clean_df[col2].values
    n = len(x)

    # Check for ties
    n_tied_x, n_tied_y = check_ties(x, y)  # pyright: ignore
    if n_tied_x > 0 or n_tied_y > 0:
        pct_tied_x = 100 * n_tied_x / n
        pct_tied_y = 100 * n_tied_y / n
        logger.warning("  ⚠ Data contains ties:")
        logger.warning(f"    {col1}: {n_tied_x}/{n} tied ({pct_tied_x:.1f}%)")
        logger.warning(f"    {col2}: {n_tied_y}/{n} tied ({pct_tied_y:.1f}%)")
        logger.warning("  → Test sensitivity may be reduced with tied data")

    # Transform to pseudo-observations
    # This handles ties properly and is invariant under monotone transforms
    data_uniform = pseudo_obs(clean_df.values)

    # Fit copula and test
    if copula_type == "gaussian":
        result = fit_gaussian_copula_and_test(data_uniform, n_tied_x, n_tied_y)
    else:  # student-t
        result = fit_studentt_copula_and_test(data_uniform, n_tied_x, n_tied_y)

    # Set pair name
    result.pair_name = pair_name

    return result


def analyze_attacks(df: pd.DataFrame) -> Dict[str, CopulaResult]:
    """Analyze all three pairwise combinations for both copulas."""
    pairs = [
        ("uc", "ts", "UC vs TS"),
        ("uc", "us", "UC vs US"),
        ("ts", "us", "TS vs US"),
    ]

    results = {}
    for col1, col2, pair_name in pairs:
        logger.info(f"Testing {pair_name}...")
        for copula_type in ["gaussian", "student-t"]:
            key = f"{pair_name}_{copula_type}"
            results[key] = test_independence(df, col1, col2, pair_name, copula_type)
            result = results[key]

            if copula_type == "gaussian":
                logger.info(
                    f"  {copula_type}: ρ={result.correlation:.3f}, "
                    f"LR={result.lr_statistic:.2f}, p={result.p_value:.4f}, "
                    f"{'✓ Independent' if result.is_independent else '⚠ Dependent'}"
                )
            else:
                logger.info(
                    f"  {copula_type}: ρ={result.correlation:.3f}, "
                    f"ν={result.degrees_freedom:.1f}, "
                    f"LR={result.lr_statistic:.2f}, p={result.p_value:.4f} (mixture dist), "
                    f"{'✓ Independent' if result.is_independent else '⚠ Dependent'}"
                )

    return results


def plot_results(results: Dict[str, CopulaResult], df: pd.DataFrame, save_path: Path):
    """Create 2x3 grid: 2 copulas × 3 pairs."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Copula Independence Analysis - Attacks Dataset\n(Likelihood Ratio Tests)",
        fontsize=16,
        fontweight="bold",
    )

    pairs = [
        ("uc", "ts", "UC vs TS", "Untrusted Collusion", "Trusted Suspiciousness"),
        ("uc", "us", "UC vs US", "Untrusted Collusion", "Untrusted Suspiciousness"),
        ("ts", "us", "TS vs US", "Trusted Suspiciousness", "Untrusted Suspiciousness"),
    ]

    for col_idx, (col1, col2, pair_name, xlabel, ylabel) in enumerate(pairs):
        # Clean data
        clean_df = df[[col1, col2]].dropna()
        x, y = clean_df[col1].values, clean_df[col2].values
        n = len(x)

        # Transform to pseudo-observations
        data_uniform = pseudo_obs(clean_df.values)
        u, v = data_uniform[:, 0], data_uniform[:, 1]

        # Create density grid in uniform space (stay away from boundaries)
        u_grid = np.linspace(0.05, 0.95, 100)
        v_grid = np.linspace(0.05, 0.95, 100)
        grid_u, grid_v = np.meshgrid(u_grid, v_grid)
        grid_points = np.column_stack([grid_u.flatten(), grid_v.flatten()])

        # Plot Gaussian copula (top row)
        ax = axes[0, col_idx]
        ax.scatter(u, v, alpha=0.3, s=10, c="steelblue", label="Data", zorder=2)

        gauss_key = f"{pair_name}_gaussian"
        gauss = results[gauss_key]
        try:
            # Calculate fitted copula PDF
            density = gauss.copula.pdf(grid_points).reshape(100, 100)  # pyright: ignore
            # Remove infinities and NaNs
            density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip to reasonable percentile for visualization
            density = np.clip(density, 0, np.percentile(density[density > 0], 99))
            contours = ax.contour(
                grid_u,
                grid_v,
                density,
                levels=10,
                colors="red",
                alpha=0.6,
                linewidths=1.5,
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
        except Exception as e:
            logger.warning(f"Could not plot Gaussian contours for {pair_name}: {e}")

        ax.set_xlabel(f"{xlabel} (pseudo-obs)", fontsize=10)
        ax.set_ylabel(f"{ylabel} (pseudo-obs)", fontsize=10)

        # Add tie warning to title if needed
        tie_warning = ""
        if gauss.n_tied_x > 0 or gauss.n_tied_y > 0:
            pct_tied = max(100 * gauss.n_tied_x / n, 100 * gauss.n_tied_y / n)
            tie_warning = f"\n⚠ Ties: {pct_tied:.0f}%"

        ax.set_title(
            f"Gaussian Copula - {pair_name}\n"
            f"ρ={gauss.correlation:.3f}, LR={gauss.lr_statistic:.2f}, "
            f"p={gauss.p_value:.4f}\n"
            f"{'✓ Independent' if gauss.is_independent else '⚠ Dependent'}{tie_warning}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plot Student-t copula (bottom row)
        ax = axes[1, col_idx]
        ax.scatter(u, v, alpha=0.3, s=10, c="steelblue", label="Data", zorder=2)

        student_key = f"{pair_name}_student-t"
        student = results[student_key]
        try:
            # Calculate fitted copula PDF
            density = student.copula.pdf(grid_points).reshape(100, 100)  # pyright: ignore
            # Remove infinities and NaNs
            density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip to reasonable percentile for visualization
            density = np.clip(density, 0, np.percentile(density[density > 0], 99))
            contours = ax.contour(
                grid_u,
                grid_v,
                density,
                levels=10,
                colors="darkgreen",
                alpha=0.6,
                linewidths=1.5,
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
        except Exception as e:
            logger.warning(f"Could not plot Student-t contours for {pair_name}: {e}")

        ax.set_xlabel(f"{xlabel} (pseudo-obs)", fontsize=10)
        ax.set_ylabel(f"{ylabel} (pseudo-obs)", fontsize=10)

        # Add tie warning to title if needed
        tie_warning = ""
        if student.n_tied_x > 0 or student.n_tied_y > 0:
            pct_tied = max(100 * student.n_tied_x / n, 100 * student.n_tied_y / n)
            tie_warning = f"\n⚠ Ties: {pct_tied:.0f}%"

        ax.set_title(
            f"Student-t Copula - {pair_name}\n"
            f"ρ={student.correlation:.3f}, ν={student.degrees_freedom:.1f}, "
            f"LR={student.lr_statistic:.2f}, p={student.p_value:.4f}*{tie_warning}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Add footnotes
    fig.text(
        0.5,
        0.02,
        "* Student-t p-values use mixture distribution (0.5·χ²(1) + 0.5·χ²(2)) per Genest & Nešlehová (2007)",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.text(
        0.5,
        0.005,
        "Note: Test is invariant under monotone transforms but may have reduced sensitivity with many ties",
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )

    plt.tight_layout(rect=[0, 0.035, 1, 0.98])  # pyright: ignore
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {save_path}")
    plt.close()


def print_summary(experiment_name: str, results: Dict[str, CopulaResult]):
    """Print concise summary."""
    print(f"\n{'=' * 80}")
    print(f"COPULA-BASED INDEPENDENCE TESTING - {experiment_name}")
    print("ATTACKS DATASET ONLY")
    print("Method: Likelihood Ratio Tests with Pseudo-Observations")
    print(f"{'=' * 80}\n")

    pairs = ["UC vs TS", "UC vs US", "TS vs US"]

    for pair in pairs:
        gauss = results[f"{pair}_gaussian"]
        student = results[f"{pair}_student-t"]

        print(f"{pair}:")

        # Tie warning if present
        if gauss.n_tied_x > 0 or gauss.n_tied_y > 0:
            print(
                f"  ⚠ Warning: Data contains {gauss.n_tied_x + gauss.n_tied_y} tied datapoints"
            )
            print("    (may reduce test sensitivity)")

        print("  Gaussian Copula:")
        print(f"    ρ = {gauss.correlation:.4f}")
        print(f"    Log-likelihood (fitted) = {gauss.log_likelihood:.2f}")
        print(f"    Log-likelihood (indep)  = {gauss.log_likelihood_indep:.2f}")
        print(f"    LR statistic = {gauss.lr_statistic:.2f}")
        print(f"    p-value = {gauss.p_value:.4f} [χ²(1)]")
        print(f"    → {'✓ INDEPENDENT' if gauss.is_independent else '⚠ DEPENDENT'}\n")

        print("  Student-t Copula:")
        print(f"    ρ = {student.correlation:.4f}")
        print(f"    ν = {student.degrees_freedom:.2f}")
        print(f"    Log-likelihood (fitted) = {student.log_likelihood:.2f}")
        print(f"    Log-likelihood (indep)  = {student.log_likelihood_indep:.2f}")
        print(f"    LR statistic = {student.lr_statistic:.2f}")
        print(f"    p-value = {student.p_value:.4f} [0.5·χ²(1) + 0.5·χ²(2)]")
        print(f"    → {'✓ INDEPENDENT' if student.is_independent else '⚠ DEPENDENT'}\n")

    print("Notes:")
    print(
        "  • Student-t p-values use mixture distribution per Genest & Nešlehová (2007)"
    )
    print("  • Tests use pseudo-observations (invariant under monotone transforms)")
    print("  • Ties in data may reduce test sensitivity")
    print(f"{'=' * 80}\n")


def process_config(config_path: Path, output_dir: Path = None):  # pyright: ignore
    """Process a single config file."""
    logger.info(f"Processing: {config_path}")

    # Load config
    config = load_and_normalize_config(config_path, config_type="monitor")
    runs = {run.label: run for run in config.runs}

    # Load attacks data only
    attacks_df = load_scores(runs[RunLabel.ATTACKS].save_path)
    logger.info(f"Loaded {len(attacks_df)} attack samples")

    # Analyze all pairwise combinations
    logger.info(
        "\nAnalyzing pairwise independence with copula likelihood ratio tests..."
    )
    results = analyze_attacks(attacks_df)

    # Output
    if output_dir is None:
        output_dir = Path(config.results_folder or config.experiment_folder / "results")
        output_dir = output_dir / "copula_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f"{config.experiment_name}_copula_attacks.png"
    plot_results(results, attacks_df, plot_path)
    print_summary(config.experiment_name, results)


def main():
    parser = argparse.ArgumentParser(
        description="Copula-based independence testing using likelihood ratio tests"
    )
    parser.add_argument(
        "--config", "-cfg", nargs="+", required=True, help="Config file path(s)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )
    args = parser.parse_args()

    for cfg in args.config:
        try:
            process_config(Path(cfg), args.output_dir)
        except Exception as e:
            logger.error(f"Failed: {cfg}: {e}")
            raise


if __name__ == "__main__":
    main()
