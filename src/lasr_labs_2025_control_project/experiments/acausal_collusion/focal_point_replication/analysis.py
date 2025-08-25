from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from inspect_ai.log import EvalLog, read_eval_log


def load_inspect_logs(log_paths: List[str]) -> List[EvalLog]:
    """Load Inspect AI log files using the official API."""
    logs = []
    for log_path in log_paths:
        try:
            log = read_eval_log(log_path)
            logs.append(log)
            print(f"Loaded log: {log_path}")
            print(f"  - Samples: {len(log.samples) if log.samples else 0}")
            print(f"  - Model: {log.eval.model}")
            print(f"  - Dataset: {log.eval.dataset}")
        except Exception as e:
            print(f"Error loading {log_path}: {e}")

    return logs


def analyze_sample_consistency(logs: List[EvalLog]) -> Dict[str, Any]:
    """Analyze which samples consistently complete across all runs."""

    if not logs:
        raise ValueError("No logs provided")

    # Extract sample data from each log
    all_runs = {}
    for i, log in enumerate(logs):
        run_name = f"run_{i + 1}"
        run_samples = {}

        if log.samples:
            for sample in log.samples:
                sample_id = sample.id

                # Check completion status
                completed = sample.output is not None

                # Check if there's a score (success)
                success = False
                score = 0.0
                if sample.scores and isinstance(sample.scores, dict):
                    first_score = next(iter(sample.scores.values()))
                    score_value = (
                        first_score.value
                        if hasattr(first_score, "value")
                        else first_score
                    )

                    # Convert score value to numeric for analysis
                    if score_value == "C":  # 'C' likely means Complete/Correct
                        score = 1.0
                        success = True
                    elif isinstance(score_value, (int, float)):
                        score = float(score_value)
                        success = score > 0
                    else:
                        # Handle other string values as needed
                        score = 0.0
                        success = False

                # Extract output information
                output_text = ""
                if sample.output and hasattr(sample.output, "completion"):
                    output_text = sample.output.completion
                elif sample.output:
                    output_text = str(sample.output)

                run_samples[sample_id] = {
                    "completed": completed,
                    "success": success,
                    "score": score,
                    "output_length": len(output_text),
                    "has_output": bool(output_text.strip()),
                    "sample": sample,  # Keep reference for detailed analysis
                }

        all_runs[run_name] = run_samples
        print(f"{run_name}: {len(run_samples)} samples")

    # Find all unique sample IDs
    all_sample_ids = set()
    for run_data in all_runs.values():
        all_sample_ids.update(run_data.keys())

    print(f"\nTotal unique samples: {len(all_sample_ids)}")

    # Analyze consistency
    completion_stats = {}
    completed_in_all_runs = []
    successful_in_all_runs = []
    has_output_in_all_runs = []

    for sample_id in all_sample_ids:
        runs_completed = 0
        runs_successful = 0
        runs_with_output = 0
        runs_present = 0
        total_score = 0.0

        sample_info = {"sample_id": sample_id, "runs": {}}

        for run_name, run_data in all_runs.items():
            if sample_id in run_data:
                runs_present += 1
                sample_data = run_data[sample_id]

                sample_info["runs"][run_name] = {
                    "completed": sample_data["completed"],
                    "success": sample_data["success"],
                    "score": sample_data["score"],
                    "has_output": sample_data["has_output"],
                    "output_length": sample_data["output_length"],
                }

                if sample_data["completed"]:
                    runs_completed += 1
                if sample_data["success"]:
                    runs_successful += 1
                if sample_data["has_output"]:
                    runs_with_output += 1

                total_score += sample_data["score"]

        # Calculate rates
        num_runs = len(logs)
        sample_info.update(
            {
                "runs_present": runs_present,
                "runs_completed": runs_completed,
                "runs_successful": runs_successful,
                "runs_with_output": runs_with_output,
                "completion_rate": runs_completed / num_runs,
                "success_rate": runs_successful / num_runs,
                "output_rate": runs_with_output / num_runs,
                "avg_score": total_score / runs_present if runs_present > 0 else 0,
            }
        )

        completion_stats[sample_id] = sample_info

        # Track samples that meet criteria in all runs
        if runs_completed == num_runs:
            completed_in_all_runs.append(sample_id)
        if runs_successful == num_runs:
            successful_in_all_runs.append(sample_id)
        if runs_with_output == num_runs:
            has_output_in_all_runs.append(sample_id)

    return {
        "completion_stats": completion_stats,
        "completed_in_all_runs": completed_in_all_runs,
        "successful_in_all_runs": successful_in_all_runs,
        "has_output_in_all_runs": has_output_in_all_runs,
        "total_samples": len(all_sample_ids),
        "num_runs": len(logs),
        "logs": logs,  # Keep reference to original logs
    }


def create_summary_report(analysis_results: Dict[str, Any]) -> str:
    """Create a detailed summary report."""
    stats = analysis_results["completion_stats"]
    completed_all = analysis_results["completed_in_all_runs"]
    successful_all = analysis_results["successful_in_all_runs"]
    output_all = analysis_results["has_output_in_all_runs"]
    total_samples = analysis_results["total_samples"]
    num_runs = analysis_results["num_runs"]

    report = f"""
=== INSPECT AI LOG ANALYSIS SUMMARY ===

Total Samples: {total_samples}
Number of Runs: {num_runs}

CONSISTENCY ANALYSIS:
- Completed in ALL runs: {len(completed_all)} ({len(completed_all) / total_samples * 100:.1f}%)
- Successful in ALL runs: {len(successful_all)} ({len(successful_all) / total_samples * 100:.1f}%)
- Has output in ALL runs: {len(output_all)} ({len(output_all) / total_samples * 100:.1f}%)

COMPLETION RATE DISTRIBUTION:
"""

    # Calculate distributions
    completion_rates = [stats[sid]["completion_rate"] for sid in stats]
    success_rates = [stats[sid]["success_rate"] for sid in stats]

    # Completion rate distribution
    rate_counts = defaultdict(int)
    for rate in completion_rates:
        rate_key = round(rate * num_runs) / num_runs  # Round to nearest possible rate
        rate_counts[rate_key] += 1

    for rate in sorted(rate_counts.keys(), reverse=True):
        count = rate_counts[rate]
        percentage = count / total_samples * 100
        report += (
            f"- {rate * 100:.0f}% completion: {count} samples ({percentage:.1f}%)\n"
        )

    # Success rate distribution
    report += "\nSUCCESS RATE DISTRIBUTION:\n"
    success_counts = defaultdict(int)
    for rate in success_rates:
        rate_key = round(rate * num_runs) / num_runs
        success_counts[rate_key] += 1

    for rate in sorted(success_counts.keys(), reverse=True):
        count = success_counts[rate]
        percentage = count / total_samples * 100
        report += f"- {rate * 100:.0f}% success: {count} samples ({percentage:.1f}%)\n"

    # Most consistent samples
    report += "\nMOST RELIABLE SAMPLES (completed in all runs):\n"
    for sample_id in sorted(completed_all)[:20]:
        avg_score = stats[sample_id]["avg_score"]
        report += f"- {sample_id} (avg score: {avg_score:.3f})\n"

    if len(completed_all) > 20:
        report += f"... and {len(completed_all) - 20} more\n"

    return report


def export_detailed_results(analysis_results: Dict[str, Any], output_file: str):
    """Export detailed analysis to CSV."""
    stats = analysis_results["completion_stats"]

    rows = []
    for sample_id, info in stats.items():
        row = {
            "sample_id": sample_id,
            "completion_rate": info["completion_rate"],
            "success_rate": info["success_rate"],
            "output_rate": info["output_rate"],
            "avg_score": info["avg_score"],
            "runs_completed": info["runs_completed"],
            "runs_successful": info["runs_successful"],
            "runs_with_output": info["runs_with_output"],
        }

        # Add per-run details
        for run_name, run_info in info["runs"].items():
            row[f"{run_name}_completed"] = run_info["completed"]
            row[f"{run_name}_success"] = run_info["success"]
            row[f"{run_name}_score"] = run_info["score"]
            row[f"{run_name}_output_length"] = run_info["output_length"]

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["completion_rate", "success_rate", "avg_score"], ascending=False
    )
    df.to_csv(output_file, index=False)
    print(f"Detailed results exported to: {output_file}")


def get_sample_details(
    analysis_results: Dict[str, Any], sample_id: str
) -> Optional[Dict[Any, Any]]:
    """Get detailed information about a specific sample across all runs."""
    if sample_id not in analysis_results["completion_stats"]:
        return None

    sample_info = analysis_results["completion_stats"][sample_id]

    # Get the actual sample content from one of the logs
    logs = analysis_results["logs"]
    sample_content = None

    for log in logs:
        if log.samples:
            for sample in log.samples:
                if sample.id == sample_id:
                    sample_content = {
                        "input": sample.input,
                        "target": sample.target,
                        "metadata": sample.metadata,
                    }
                    break
            if sample_content:
                break

    return {"sample_id": sample_id, "content": sample_content, "stats": sample_info}


def get_perfect_score_samples(analysis_results: Dict[str, Any]) -> List[str]:
    """Return a list of sample IDs that have perfect average scores (1.0)."""
    stats = analysis_results["completion_stats"]
    perfect_samples = []

    for sample_id, info in stats.items():
        if info["avg_score"] == 1.0:
            perfect_samples.append(sample_id)

    return sorted(perfect_samples)


def save_summary_report(analysis_results: Dict[str, Any], log_files: List[str]) -> str:
    """Create and save summary report to a txt file in the same location as the logs."""

    # Generate the report content
    report_content = create_summary_report(analysis_results)

    # Get the directory of the first log file
    log_dir = Path(log_files[0]).parent

    # Create timestamp for unique filename
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create report filename
    report_filename = f"inspect_analysis_report_{timestamp}.txt"
    report_path = log_dir / report_filename

    # Write the report
    with open(report_path, "w") as f:
        f.write(report_content)

        # Add the perfect score samples section
        perfect_samples = get_perfect_score_samples(analysis_results)
        f.write("\n\n=== SAMPLES WITH PERFECT AVERAGE SCORE (1.0) ===\n")
        f.write(f"Count: {len(perfect_samples)}\n\n")
        for sample_id in perfect_samples:
            f.write(f"{sample_id}\n")

        # Add log file information
        f.write("\n\n=== LOG FILES ANALYZED ===\n")
        for i, log_file in enumerate(log_files, 1):
            f.write(f"{i}. {log_file}\n")

        f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Summary report saved to: {report_path}")
    return str(report_path)


def main():
    # UPDATE THESE PATHS TO YOUR ACTUAL LOG FILES
    log_files = [
        "src/lasr_labs_2025_control_project/experiments/acausal_collusion/logs/2025-08-21T08-37-36+01-00_get-valid-samples_CHHZG5DDGmZjUptThfRcUr.eval",
        "src/lasr_labs_2025_control_project/experiments/acausal_collusion/logs/2025-08-21T08-44-36+01-00_get-valid-samples_S2LryFTvcUqfXLF9b9t22a.eval",
        "src/lasr_labs_2025_control_project/experiments/acausal_collusion/logs/2025-08-21T08-48-16+01-00_get-valid-samples_3c9BN7CxtubivQV5eKcwT6.eval",
    ]

    # Verify files exist
    for log_file in log_files:
        if not Path(log_file).exists():
            print(f"Error: Log file not found: {log_file}")
            return None

    print("Loading Inspect AI logs...")
    logs = load_inspect_logs(log_files)

    if not logs:
        print("No logs could be loaded!")
        return None

    print("\nAnalyzing sample consistency...")
    results = analyze_sample_consistency(logs)

    # Print summary report
    summary = create_summary_report(results)
    print(summary)

    # Save summary report to txt file
    save_summary_report(results, log_files)

    # Export detailed results (also save in same directory as logs)
    log_dir = Path(log_files[0]).parent
    csv_path = log_dir / "inspect_completion_analysis.csv"
    export_detailed_results(results, str(csv_path))

    # Print most reliable samples
    reliable_samples = results["completed_in_all_runs"]
    print(f"\n=== SAMPLES THAT COMPLETED IN ALL {results['num_runs']} RUNS ===")
    print(f"Count: {len(reliable_samples)}")

    for sample_id in sorted(reliable_samples)[:10]:  # Show top 10
        stats = results["completion_stats"][sample_id]
        print(f"  {sample_id}: avg_score={stats['avg_score']:.3f}")

    if len(reliable_samples) > 10:
        print(f"  ... and {len(reliable_samples) - 10} more")

    # Get samples with perfect average scores
    perfect_score_samples = get_perfect_score_samples(results)
    print("\n=== SAMPLES WITH PERFECT AVERAGE SCORE (1.0) ===")
    print(f"Count: {len(perfect_score_samples)}")
    for sample_id in perfect_score_samples[:20]:  # Show first 20
        print(f"  {sample_id}")
    if len(perfect_score_samples) > 20:
        print(f"  ... and {len(perfect_score_samples) - 20} more")

    return results


if __name__ == "__main__":
    results = main()

    # Example: Get details for a specific sample
    if results and results["completed_in_all_runs"]:
        sample_id = results["completed_in_all_runs"][0]
        details = get_sample_details(results, sample_id)
        if not details:
            print(f"No details found for sample ID: {sample_id}")
            exit(1)
        print(f"\nExample sample details for {sample_id}:")
        print(f"Stats: {details['stats']['runs']}")
