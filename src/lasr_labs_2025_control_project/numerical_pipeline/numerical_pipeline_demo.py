from lasr_labs_2025_control_project.numerical_pipeline import utils, visualization


def main():
    red_team_confusion_matrix = {"tn": 94, "fp": 2, "fn": 5, "tp": 19}

    config = utils.build_optimization_config(
        num_tasks=5000,
        blue_team_budget_fraction=0.02,
        red_team_confusion_matrix=red_team_confusion_matrix,
    )
    result = utils.optimize_parameters(
        config=config,
        max_iterations=100,
        grid_squares_per_iteration=10,
    )

    data = {"GPT-4.1 (Finetuned)": result}

    visualization.plot_safety_usefulness_curves(data)


if __name__ == "__main__":
    main()
