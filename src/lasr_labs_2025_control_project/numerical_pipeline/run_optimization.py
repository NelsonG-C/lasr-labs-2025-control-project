#!/usr/bin/env python3
"""
Numerical Pipeline Runner

This script demonstrates the alternating optimization between red and blue teams.
- Blue team tries to maximize safety and usefulness
- Red team tries to minimize safety and usefulness
- Uses simple grid search optimization for robust convergence
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from lasr_labs_2025_control_project.numerical_pipeline.utils import (
    BlueTeamChoices,
    NumericalPipeline,
    NumericalPipelineParameters,
    RedTeamChoices,
)


def create_default_parameters() -> Dict[str, Any]:
    """Create default parameters for the numerical pipeline"""
    return {
        "self_recognition_auc_roc": 0.8,
        "replace_fraction": 0.2,
        "num_tasks": 1000,
        "blue_team_budget_fraction": 0.01,
        "max_iterations": 15,
        "verbose": True,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    defaults = create_default_parameters()

    parser = argparse.ArgumentParser(
        description="Run numerical pipeline optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # NumericalPipelineParameters arguments
    parser.add_argument(
        "--self-recognition-auc-roc",
        type=float,
        default=defaults["self_recognition_auc_roc"],
        help="Self-recognition AUC-ROC value (0.0 to 1.0)",
    )

    parser.add_argument(
        "--replace-fraction",
        type=float,
        default=defaults["replace_fraction"],
        help="Fraction of tasks to replace (0.0 to 1.0)",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=defaults["num_tasks"],
        help="Number of tasks in the pipeline",
    )

    parser.add_argument(
        "--blue-team-budget-fraction",
        type=float,
        default=defaults["blue_team_budget_fraction"],
        help="Blue team budget fraction (0.0 to 1.0)",
    )

    # Optimization parameters
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=defaults["max_iterations"],
        help="Maximum number of optimization iterations",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=defaults["verbose"],
        help="Enable verbose output",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output (overrides --verbose)",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["single", "multiple", "curves"],
        default="multiple",
        help="Run mode: single scenario, multiple scenarios, or generate curves",
    )

    # Single scenario mode parameters
    parser.add_argument(
        "--scenario-name",
        type=str,
        default="Custom Scenario",
        help="Name for single scenario mode",
    )

    return parser.parse_args()


def main() -> Dict[str, Any] | None:
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()

    # Handle quiet flag
    verbose = args.verbose and not args.quiet

    if verbose:
        print("Starting Numerical Pipeline Optimization")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Parameters:")
        print(f"  Self-recognition AUC-ROC: {args.self_recognition_auc_roc}")
        print(f"  Replace fraction: {args.replace_fraction}")
        print(f"  Number of tasks: {args.num_tasks}")
        print(f"  Blue team budget fraction: {args.blue_team_budget_fraction}")
        print(f"  Max iterations: {args.max_iterations}")
        print(f"  Mode: {args.mode}")

    try:
        if args.mode == "single":
            # Run single scenario with custom parameters
            all_results = run_single_scenario_with_params(args, verbose)
        elif args.mode == "curves":
            # Generate curves only
            all_results = generate_curves_only(args, verbose)
        else:
            # Run multiple scenarios (default)
            all_results = run_multiple_scenarios_with_params(args, verbose)

        if verbose and all_results:
            # Print summary
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)

            if args.mode == "multiple" and "results" in all_results:
                for scenario_name, data in all_results["results"].items():
                    print(
                        f"{scenario_name:25} | AUC: {data['auc']:.2f} | "
                        f"Objective: {data['final_objective']:.3f} | "
                        f"Improvement: {data['improvement']:.3f}"
                    )
            elif args.mode == "single":
                print(f"Scenario: {args.scenario_name}")
                print(f"Final objective: {all_results.get('final_objective', 'N/A')}")
                print(f"Improvement: {all_results.get('improvement', 'N/A')}")

            # Save results
            save_results(all_results)

            # Create safety/usefulness curves and AUC-SUC plots
            if args.mode in ["multiple", "curves"]:
                create_summary_plot(all_results)

            print(
                "\nOptimization complete! Check the results directory for detailed output."
            )

        return all_results

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_single_scenario_with_params(
    args: argparse.Namespace, verbose: bool = True
) -> Dict[str, Any]:
    """Run optimization for a single scenario using command line parameters"""

    # Create pipeline parameters from args
    params = NumericalPipelineParameters(
        self_recognition_auc_roc=args.self_recognition_auc_roc,
        replace_fraction=args.replace_fraction,
        num_tasks=args.num_tasks,
        blue_team_budget_fraction=args.blue_team_budget_fraction,
    )

    final_blue, final_red, history = run_single_scenario(
        args.scenario_name,
        args.self_recognition_auc_roc,
        args.max_iterations,
        verbose,
        params,
    )

    return {
        "scenario": args.scenario_name,
        "parameters": {
            "self_recognition_auc_roc": args.self_recognition_auc_roc,
            "replace_fraction": args.replace_fraction,
            "num_tasks": args.num_tasks,
            "blue_team_budget_fraction": args.blue_team_budget_fraction,
        },
        "final_blue": {"honeypot_fraction": final_blue.honeypot_fraction},
        "final_red": {
            "sus_threshold": final_red.sus_threshold,
            "attack_frequency": final_red.attack_frequency,
        },
        "final_safety": history[-1]["safety"],
        "final_usefulness": history[-1]["usefulness"],
        "final_objective": history[-1]["objective"],
        "improvement": history[-1]["objective"] - history[0]["objective"],
        "history": history,
    }


def run_multiple_scenarios_with_params(
    args: argparse.Namespace, verbose: bool = True
) -> Dict[str, Any]:
    """Run optimization across multiple scenarios with custom parameters"""

    scenarios = [
        {"name": "Random Recognition", "auc": 0.5},
        {"name": "Poor Recognition", "auc": 0.6},
        {"name": "Moderate Recognition", "auc": 0.7},
        {"name": "Good Recognition", "auc": 0.8},
        {"name": "Excellent Recognition", "auc": 0.9},
        {"name": "Near Perfect Recognition", "auc": 0.95},
    ]

    results = {}
    all_history = []

    for scenario in scenarios:
        try:
            # Create pipeline parameters for this scenario
            params = NumericalPipelineParameters(
                self_recognition_auc_roc=scenario["auc"],
                replace_fraction=args.replace_fraction,
                num_tasks=args.num_tasks,
                blue_team_budget_fraction=args.blue_team_budget_fraction,
            )

            final_blue, final_red, history = run_single_scenario(
                scenario["name"], scenario["auc"], args.max_iterations, verbose, params
            )

            results[scenario["name"]] = {
                "auc": scenario["auc"],
                "final_blue": {"honeypot_fraction": final_blue.honeypot_fraction},
                "final_red": {
                    "sus_threshold": final_red.sus_threshold,
                    "attack_frequency": final_red.attack_frequency,
                },
                "final_safety": history[-1]["safety"],
                "final_usefulness": history[-1]["usefulness"],
                "final_objective": history[-1]["objective"],
                "improvement": history[-1]["objective"] - history[0]["objective"],
            }

            # Add scenario info to history
            for entry in history:
                entry["scenario"] = scenario["name"]
                entry["auc"] = scenario["auc"]
            all_history.extend(history)

        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
            continue

    return {"results": results, "history": all_history}


def generate_curves_only(
    args: argparse.Namespace, verbose: bool = True
) -> Dict[str, Any]:
    """Generate only the safety/usefulness curves with custom parameters"""

    if verbose:
        print("Generating safety/usefulness curves only...")

    # Use the curve generation function with custom parameters
    import numpy as np

    self_rec_aucs = [0.6, 0.7, 0.8, 0.9, 0.95]
    replace_fractions = np.linspace(0.0, 0.9, 20)

    curves_data = generate_safety_usefulness_curves_with_params(
        self_rec_aucs, replace_fractions.tolist(), args
    )

    # Calculate AUC-SUC for each curve
    auc_suc_values = []
    for self_rec_auc in self_rec_aucs:
        data = curves_data[self_rec_auc]
        auc_suc = calculate_auc_suc(data["safety"], data["usefulness"])
        auc_suc_values.append(auc_suc)
        if verbose:
            print(f"AUC-SUC for self-rec AUC {self_rec_auc:.2f}: {auc_suc:.4f}")

    return {
        "self_rec_aucs": self_rec_aucs,
        "auc_suc_values": auc_suc_values,
        "curves_data": curves_data,
        "replace_fractions": replace_fractions.tolist(),
    }


def run_single_scenario(
    scenario_name: str,
    self_recognition_auc: float,
    max_iterations: int = 10,
    verbose: bool = True,
    params: NumericalPipelineParameters | None = None,
) -> Tuple[BlueTeamChoices, RedTeamChoices, List[Dict[str, Any]]]:
    """Run optimization for a single scenario"""

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Running scenario: {scenario_name}")
        print(f"Self-recognition AUC: {self_recognition_auc}")
        print(f"{'=' * 60}")

    # Create pipeline parameters (use passed params or create default)
    if params is None:
        params = NumericalPipelineParameters(
            self_recognition_auc_roc=self_recognition_auc,
            replace_fraction=0.2,
            num_tasks=1000,
            blue_team_budget_fraction=0.01,
        )

    # Initialize pipeline
    pipeline = NumericalPipeline(params)

    # Track optimization history
    history = []

    # Get initial choices
    blue_choices = BlueTeamChoices()
    red_choices = RedTeamChoices()

    if verbose:
        print("Initial choices:")
        print(f"  Blue team: honeypot_fraction={blue_choices.honeypot_fraction}")
        print(
            f"  Red team: sus_threshold={red_choices.sus_threshold}, "
            f"attack_frequency={red_choices.attack_frequency}"
        )

    # Calculate initial values
    safety, usefulness = pipeline.calculate_safety_usefulness(blue_choices, red_choices)
    objective = safety + usefulness

    history.append(
        {
            "iteration": 0,
            "blue_honeypot_fraction": blue_choices.honeypot_fraction,
            "red_sus_threshold": red_choices.sus_threshold,
            "red_attack_frequency": red_choices.attack_frequency,
            "safety": safety,
            "usefulness": usefulness,
            "objective": objective,
        }
    )

    if verbose:
        print(
            f"  Initial Safety: {safety:.4f}, Usefulness: {usefulness:.4f}, "
            f"Objective: {objective:.4f}"
        )

    # Run optimization
    if verbose:
        print(f"\nRunning optimization for up to {max_iterations} iterations...")

    final_blue, final_red = pipeline.run_optimization(max_iterations=max_iterations)

    # Calculate final values
    final_safety, final_usefulness = pipeline.calculate_safety_usefulness(
        final_blue, final_red
    )
    final_objective = final_safety + final_usefulness

    # Add final state to history
    history.append(
        {
            "iteration": "final",
            "blue_honeypot_fraction": final_blue.honeypot_fraction,
            "red_sus_threshold": final_red.sus_threshold,
            "red_attack_frequency": final_red.attack_frequency,
            "safety": final_safety,
            "usefulness": final_usefulness,
            "objective": final_objective,
        }
    )

    if verbose:
        print("\nFinal results:")
        print(f"  Blue team: honeypot_fraction={final_blue.honeypot_fraction:.4f}")
        print(
            f"  Red team: sus_threshold={final_red.sus_threshold:.4f}, "
            f"attack_frequency={final_red.attack_frequency:.4f}"
        )
        print(
            f"  Final Safety: {final_safety:.4f}, Usefulness: {final_usefulness:.4f}, "
            f"Objective: {final_objective:.4f}"
        )
        print(f"  Improvement: {final_objective - objective:.4f}")

    return final_blue, final_red, history


def run_multiple_scenarios() -> Dict[str, Any]:
    """Run optimization across multiple scenarios"""

    print("Numerical Pipeline Optimization Runner")
    print("=" * 60)
    print("Testing alternating red/blue team optimization")
    print("Blue team maximizes safety + usefulness")
    print("Red team minimizes safety + usefulness")
    print("Using robust grid search optimization")

    scenarios = [
        {"name": "Random Recognition", "auc": 0.5},
        {"name": "Poor Recognition", "auc": 0.6},
        {"name": "Moderate Recognition", "auc": 0.7},
        {"name": "Good Recognition", "auc": 0.8},
        {"name": "Excellent Recognition", "auc": 0.9},
        {"name": "Near Perfect Recognition", "auc": 0.95},
    ]

    results = {}
    all_history = []

    for scenario in scenarios:
        try:
            final_blue, final_red, history = run_single_scenario(
                scenario["name"], scenario["auc"], max_iterations=15, verbose=True
            )

            results[scenario["name"]] = {
                "auc": scenario["auc"],
                "final_blue": {"honeypot_fraction": final_blue.honeypot_fraction},
                "final_red": {
                    "sus_threshold": final_red.sus_threshold,
                    "attack_frequency": final_red.attack_frequency,
                },
                "final_safety": history[-1]["safety"],
                "final_usefulness": history[-1]["usefulness"],
                "final_objective": history[-1]["objective"],
                "improvement": history[-1]["objective"] - history[0]["objective"],
            }

            # Add scenario info to history
            for entry in history:
                entry["scenario"] = scenario["name"]
                entry["auc"] = scenario["auc"]
            all_history.extend(history)

        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
            continue

    return {"results": results, "history": all_history}


def save_results(results: Dict[str, Any], filename: str | None = None) -> str:
    """Save results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}.json"

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def generate_safety_usefulness_curves_with_params(
    self_rec_aucs: List[float], replace_fractions: List[float], args: argparse.Namespace
) -> Dict[float, Dict[str, List[float]]]:
    """Generate safety/usefulness curves with custom parameters"""

    curves_data = {}

    for self_rec_auc in self_rec_aucs:
        print(f"Generating curves for self-recognition AUC: {self_rec_auc:.2f}")

        safety_values = []
        usefulness_values = []

        for replace_frac in replace_fractions:
            # Create pipeline parameters using args
            params = NumericalPipelineParameters(
                self_recognition_auc_roc=self_rec_auc,
                replace_fraction=replace_frac,
                num_tasks=args.num_tasks,
                blue_team_budget_fraction=args.blue_team_budget_fraction,
            )

            # Initialize pipeline
            pipeline = NumericalPipeline(params)

            # Get optimized choices
            blue_choices, red_choices = pipeline.run_optimization(
                max_iterations=args.max_iterations
            )

            # Calculate safety and usefulness
            safety, usefulness = pipeline.calculate_safety_usefulness(
                blue_choices, red_choices
            )

            safety_values.append(safety)
            usefulness_values.append(usefulness)

        curves_data[self_rec_auc] = {
            "safety": safety_values,
            "usefulness": usefulness_values,
            "replace_fractions": replace_fractions,
        }

    return curves_data


def generate_safety_usefulness_curves(
    self_rec_aucs: List[float], replace_fractions: List[float]
) -> Dict[float, Dict[str, List[float]]]:
    """Generate safety/usefulness curves for different self-recognition AUCs"""

    curves_data = {}

    for self_rec_auc in self_rec_aucs:
        print(f"Generating curves for self-recognition AUC: {self_rec_auc:.2f}")

        safety_values = []
        usefulness_values = []

        for replace_frac in replace_fractions:
            # Create pipeline parameters
            params = NumericalPipelineParameters(
                self_recognition_auc_roc=self_rec_auc,
                replace_fraction=replace_frac,
                num_tasks=1000,
                blue_team_budget_fraction=0.01,
            )

            # Initialize pipeline
            pipeline = NumericalPipeline(params)

            # Get optimized choices
            blue_choices, red_choices = pipeline.run_optimization(max_iterations=10)

            # Calculate safety and usefulness
            safety, usefulness = pipeline.calculate_safety_usefulness(
                blue_choices, red_choices
            )

            safety_values.append(safety)
            usefulness_values.append(usefulness)

        curves_data[self_rec_auc] = {
            "safety": safety_values,
            "usefulness": usefulness_values,
            "replace_fractions": replace_fractions,
        }

    return curves_data


def calculate_auc_suc(
    safety_values: List[float], usefulness_values: List[float]
) -> float:
    """Calculate AUC-SUC (Area Under Curve - Safety vs Usefulness Curve)

    Includes anchor points at (0,1) and (1,0) as specified.
    Uses trapezoidal rule for integration.
    """
    import numpy as np

    # Add anchor points
    safety_extended = [0.0] + safety_values + [1.0]
    usefulness_extended = [1.0] + usefulness_values + [0.0]

    # Sort by safety values for proper integration
    sorted_pairs = sorted(zip(safety_extended, usefulness_extended))
    safety_sorted = [pair[0] for pair in sorted_pairs]
    usefulness_sorted = [pair[1] for pair in sorted_pairs]

    # Calculate AUC using trapezoidal rule
    auc_suc = np.trapezoid(usefulness_sorted, safety_sorted)

    return float(auc_suc)


def create_summary_plot(results: Dict[str, Any]) -> None:
    """Create safety/usefulness curves and AUC-SUC analysis"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        # Set seaborn style
        sns.set_style("whitegrid")
        plt.rcParams.update({"font.size": 12})

        # Define parameters for curve generation
        self_rec_aucs = [0.6, 0.7, 0.8, 0.9, 0.95]
        replace_fractions = np.linspace(0.0, 0.9, 20)

        print("Generating safety/usefulness curves...")
        curves_data = generate_safety_usefulness_curves(
            self_rec_aucs, replace_fractions.tolist()
        )

        # Calculate AUC-SUC for each curve
        auc_suc_values = []
        for self_rec_auc in self_rec_aucs:
            data = curves_data[self_rec_auc]
            auc_suc = calculate_auc_suc(data["safety"], data["usefulness"])
            auc_suc_values.append(auc_suc)
            print(f"AUC-SUC for self-rec AUC {self_rec_auc:.2f}: {auc_suc:.4f}")

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Safety vs Usefulness curves
        colors = sns.color_palette("viridis", len(self_rec_aucs))

        for i, self_rec_auc in enumerate(self_rec_aucs):
            data = curves_data[self_rec_auc]
            ax1.plot(
                data["safety"],
                data["usefulness"],
                color=colors[i],
                linewidth=2.5,
                label=f"Self-Rec AUC: {self_rec_auc:.2f}",
                marker="o",
                markersize=4,
                alpha=0.8,
            )

        # Add anchor points
        ax1.scatter([0, 1], [1, 0], color="red", s=100, zorder=10, marker="*")
        ax1.annotate("(0,1)", (0, 1), xytext=(-20, 10), textcoords="offset points")
        ax1.annotate("(1,0)", (1, 0), xytext=(10, 10), textcoords="offset points")

        ax1.set_xlabel("Safety", fontsize=14)
        ax1.set_ylabel("Usefulness", fontsize=14)
        ax1.set_title("Safety vs Usefulness Curves", fontsize=16, fontweight="bold")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)

        # Plot 2: AUC-SUC vs Self-Recognition AUC
        ax2.plot(
            self_rec_aucs,
            auc_suc_values,
            color="darkblue",
            linewidth=3,
            marker="o",
            markersize=8,
        )
        ax2.fill_between(self_rec_aucs, auc_suc_values, alpha=0.3, color="lightblue")

        ax2.set_xlabel("Self-Recognition AUC-ROC", fontsize=14)
        ax2.set_ylabel("AUC-SUC", fontsize=14)
        ax2.set_title(
            "Area Under Safety-Usefulness Curve\nvs Self-Recognition Capability",
            fontsize=16,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)

        # Add value annotations
        for i, (x, y) in enumerate(zip(self_rec_aucs, auc_suc_values)):
            ax2.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()

        # Save plot
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"results/safety_usefulness_curves_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Safety/usefulness curve plot saved to: {plot_filename}")

        # Also save the curves data
        curves_results = {
            "self_rec_aucs": self_rec_aucs,
            "auc_suc_values": auc_suc_values,
            "curves_data": curves_data,
            "replace_fractions": replace_fractions.tolist(),
        }

        curves_filename = f"results/curves_data_{timestamp}.json"
        with open(curves_filename, "w") as f:
            json.dump(curves_results, f, indent=2)
        print(f"Curves data saved to: {curves_filename}")

    except ImportError as e:
        print(f"Required packages not available - skipping plot generation: {e}")
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    results = main()
