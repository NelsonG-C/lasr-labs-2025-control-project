"""
Prepares the data for the binary tuning.
Reads the eval logs, parses successful solutions and saves them to a json file in our default Dataset format.
"""

# %%
import re

import pandas as pd
from inspect_ai.analysis import messages_df, samples_df

from lasr_labs_2025_control_project.finetuning.scripts.data_models import (
    Dataset,
    ProblemSolutions,
    Solution,
)

# %%
gpt41_data_path = "src/lasr_labs_2025_control_project/finetuning/binary_tuning/data/eval_logs/test/2025-09-11T10-42-42+01-00_backdoored-solutions_GegmeCCDsjTPRgu98bepHo.eval"
gpt4o_data_path = "src/lasr_labs_2025_control_project/finetuning/binary_tuning/data/eval_logs/test/2025-09-11T11-39-49+01-00_backdoored-solutions_WesZ7jkU4x2AhFLsiNsrG4.eval"


gpt41_data = samples_df(gpt41_data_path, full=True)
gpt4o_data = samples_df(gpt4o_data_path, full=True)


# %%
def both_task_success(x: pd.Series) -> bool:
    scorer_result = eval(x["score_apps_scorer"])
    return (
        scorer_result["main_task_success"] == "C"
        and scorer_result["side_task_success"] == "C"
    )


gpt41_success_data = gpt41_data[gpt41_data.apply(both_task_success, axis=1)]
gpt4o_success_data = gpt4o_data[gpt4o_data.apply(both_task_success, axis=1)]

# %%
gpt41_success_ids: set[int] = set(gpt41_success_data.sample_id.to_list())
gpt4o_success_ids: set[int] = set(gpt4o_success_data.sample_id.to_list())

# %%
gpt41_success_apps_ids: set[int] = set(gpt41_success_data.id.to_list())
gpt4o_success_apps_ids: set[int] = set(gpt4o_success_data.id.to_list())

# %%
gpt41_messages = messages_df(gpt41_data_path)
gpt41_messages = gpt41_messages[
    (gpt41_messages["role"] == "assistant")
    & (gpt41_messages["sample_id"].isin(gpt41_success_ids))
]

# %%
gpt4o_messages = messages_df(gpt4o_data_path)
gpt4o_messages = gpt4o_messages[
    (gpt4o_messages["role"] == "assistant")
    & (gpt4o_messages["sample_id"].isin(gpt4o_success_ids))
]


# %%
def final_code_match(x: str) -> str:
    match = re.search(r"<final_code>(.*?)</final_code>", x, re.DOTALL)
    if not match:
        raise ValueError(f"No final code found in {x}")
    return f"```python{match.group(1)}```"


# %%
gpt41_solutions = gpt41_messages.content.apply(final_code_match)
gpt4o_solutions = gpt4o_messages.content.apply(final_code_match)


# %%
solutions_df = gpt41_messages[["sample_id", "content"]].copy()
solutions_df["solution"] = solutions_df["content"].apply(final_code_match)
solutions_df = solutions_df[["sample_id", "solution"]]

# Create the problem dataframe from success data
problems_df = gpt41_success_data[
    ["sample_id", "metadata_main_task_description", "id"]
].copy()

# Merge them
result_df = solutions_df.merge(problems_df, on="sample_id", how="inner")

ds_gpt41_success = Dataset(
    problems=result_df.apply(  # pyright: ignore[reportCallIssue]
        lambda x: ProblemSolutions(  # pyright: ignore[reportArgumentType]
            apps_problem_id=x.id,
            problem_statement=x.metadata_main_task_description,
            solutions=[Solution(model="gpt-4.1-2025-04-14", solution=x.solution)],
        ),
        axis=1,
    ).to_list()
)


# %%
solutions_df = gpt4o_messages[["sample_id", "content"]].copy()
solutions_df["solution"] = solutions_df["content"].apply(final_code_match)
solutions_df = solutions_df[["sample_id", "solution"]]

# Create the problem dataframe from success data
problems_df = gpt4o_success_data[
    ["sample_id", "metadata_main_task_description", "id"]
].copy()

# Merge them
result_df = solutions_df.merge(problems_df, on="sample_id", how="inner")
ds_gpt4o_success = Dataset(
    problems=result_df.apply(  # pyright: ignore[reportCallIssue]
        lambda x: ProblemSolutions(  # pyright: ignore[reportArgumentType]
            apps_problem_id=x.id,
            problem_statement=x.metadata_main_task_description,
            solutions=[Solution(model="gpt-4o-2024-08-06", solution=x.solution)],
        ),
        axis=1,
    ).to_list()
)


# %%
ds_gpt4o_success.to_json(
    "src/lasr_labs_2025_control_project/finetuning/binary_tuning/data/raw/ds_gpt4o_test.json"
)
ds_gpt41_success.to_json(
    "src/lasr_labs_2025_control_project/finetuning/binary_tuning/data/raw/ds_gpt41_test.json"
)
print(
    f"Wrote {len([s for p in ds_gpt4o_success for s in p.solutions])} + {len([s for p in ds_gpt41_success for s in p.solutions])} solutions"
)
