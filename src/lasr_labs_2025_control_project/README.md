## Running the Pipeline

### Steps:

Set their config in the files.
If you need to change the prompts they use, currently this is done in apps.py

If you need to run human solutions as the baseline, you can use the human_honeypot runner.

This runner uses a different setting setup (see src/utils/human_solution_setting.py) which pulls one of the human solutions from the apps result and uses it as an input. It also uses a different prompt which asks the model to insert the backdoor into the model instead.

#### Run the modelling pipeline

Update the log file paths in the runner() function in src/utils/logs_to_model.py to match the attack (tpr) and honeypot conditions (fpr)

Run the file to get the results:
```bash
uv run src/utils/logs_to_model.py
```

#### Adding paraphrasing
Still adding this, currently some of it works to generate the paraphrasing solutions, which is in the notebook (notebook.ipynb) at the root of the directory

#### Rerunning monitors on other solutions

TBA
