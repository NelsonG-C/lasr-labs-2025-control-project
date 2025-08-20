import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval

from lasr_labs_2025_control_project.experiments.acausal_collusion.main import (
    focal_point,
)


def simple_runner():
    rounds = 3
    runs = 1
    counter = 0
    failed = False
    base_path = "src/lasr_labs_2025_control_project/experiments/acausal_collusion/logs"
    model = "anthropic/claude-3-5-haiku-latest"

    # audit tasks
    audit_tasks = {}
    # create key value pairs in audit task depending on how many runs
    starting_tasks = [f"task_{i}" for i in range(0, runs)]
    for task in starting_tasks:
        audit_tasks[task] = []
    print(f"Audit tasks initialized: {audit_tasks}")

    # Load environment variables from .env file
    load_dotenv()

    # Optional: Verify the key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    # Generate run ID and create directory
    run_id = str(uuid.uuid4())
    log_dir = Path(base_path) / run_id

    # Create directory
    log_dir.mkdir(parents=True, exist_ok=True)

    while counter < rounds and not failed:
        tasks = [
            focal_point(
                meta_models_temperature=1.0,
                temperature=1.0,
                num_epochs=1,
                epoch_reducer="mean",
                n_example_tasks=2,
                n_meta_models=5,
                audit_array=audit_tasks[f"task_{i}"],
                model=model,
            )
            for i in range(runs)
        ]
        try:
            # Run the focal point
            eval(
                tasks=tasks,
                limit=1,
                max_samples=1,
                model=model,
                log_dir=f"src/lasr_labs_2025_control_project/experiments/acausal_collusion/logs/{run_id}",
            )
            counter += 1
            print(f"Round {counter} completed successfully.")
            print("Audit tasks:", audit_tasks)

            successful_tasks = []

            for task_id in starting_tasks:
                if len(audit_tasks[task_id]) > 0:
                    successful_tasks.append(task_id)

            if len(successful_tasks) > 0:
                print("success, going to next round")
            else:
                print("no successful tasks, stopping")
                failed = True

        except Exception as e:
            print(f"Error encountered: {e}")
            failed = True


# def runner():
#     # set up dict of audit task lists for each task, give unique ids
#     # set up loop
#     # if first loop, don't read logs just use five tasks
#     # if second loop onward, read logs and see which tasks were successful, create only those
#     # pass them the correct audit task list

#     # read named logs
#     eval(
#         tasks=[
#             focal_point(),
#             focal_point(),
#             focal_point(),
#             focal_point(),
#             focal_point(),
#         ],
#         limit=1,
#         max_samples=1,
#         max_tasks=5,
#     )

# do analysis of all the log files


if __name__ == "__main__":
    simple_runner()
