from lasr_labs_2025_control_project.numerical_pipeline import utils


def main():
    config = utils.build_optimization_config(
        num_tasks=5000,
        blue_team_budget_fraction=0.02,
        red_team_auc_roc=0.94,
    )
    result = utils.optimize_parameters(
        config=config,
        max_iterations=100,
        grid_squares_per_iteration=10,
    )

    print(result)


if __name__ == "__main__":
    main()
