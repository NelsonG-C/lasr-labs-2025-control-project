from lasr_labs_2025_control_project.numerical_pipeline import utils, visualization


def main():
    red_team_confusion_matrices = {
        "GPT-4.1 (normal)": {"tn": 69, "fp": 27, "fn": 11, "tp": 13},
        "GPT-4.1 (finetuned": {"tn": 94, "fp": 2, "fn": 5, "tp": 19},
    }

    data = {}

    for name, confusion_matrix in red_team_confusion_matrices.items():
        config = utils.build_optimization_config(
            num_tasks=5_000,
            blue_team_budget_fraction=0.02,
            red_team_confusion_matrix=confusion_matrix,
        )
        result = utils.optimize_parameters(
            config=config,
            max_iterations=100,
            grid_squares_per_iteration=10,
        )
        print(result)
        data[name] = result

    visualization.plot_data(data)


if __name__ == "__main__":
    main()
