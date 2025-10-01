There is a bunch of scripts in this folder that are used to evaluate models and/or process evaluation results from OpenAI eval.

The pipeline is the following:
1. Generate the data using different models. (utils/scripts/data_generation.py)
```bash
uv run src/lasr_labs_2025_control_project/utils/scripts/data_generation.py \
  -P grok \
  -N 120 \
  -M grok-4-0709 \
  -O src/lasr_labs_2025_control_project/prompt-optimization/train_data_big_models \
  -S train
```

2. Make a train and test datasets for the target model (finetuning/utils/generate_data.py)
```bash
uv run generate_data \
  --dataset-folder src/lasr_labs_2025_control_project/finetuning/train_data_big_models \
  --output-folder src/lasr_labs_2025_control_project/finetuning/experiments/grok/grok-code-fast-1-0825/data \
  --target-model grok-code-fast-1-0825 \
  --max-problems 50
```
3. Evaluate the model on the self-recognition task. (finetuning/utils/model_evaluation.py)
```bash
uv run model_evaluation \
  -t src/lasr_labs_2025_control_project/finetuning/experiments/grok/grok-code-fast-1-0825/data/test_messages.jsonl \
  -m grok-code-fast-1-0825 \
  -p grok \
  -o src/lasr_labs_2025_control_project/finetuning/experiments/grok/grok-code-fast-1-0825/results/eval.json
```
4. Calculate the metrics for the self-recognition task. (finetuning/utils/calc_eval_metrics.py)
```bash
uv run calc_eval_metrics \
  -o src/lasr_labs_2025_control_project/finetuning/experiments/grok/grok-code-fast-1-0825/results/report.json \
  -r src/lasr_labs_2025_control_project/finetuning/experiments/grok/grok-code-fast-1-0825/results/report.txt \
  -v src/lasr_labs_2025_control_project/finetuning/experiments/grok/grok-code-fast-1-0825/results/eval_fixed.jsonl
```


---

## generate_table.py

This script takes the evaluation results from OpenAI eval and generates a table with the columns [system_message, user_message, model, ground_truth, prediction].
```bash
uv run generate_table \
  -i src/lasr_labs_2025_control_project/finetuning/experiments/grok/grok-code-fast-1-0825/results/eval.json \
  -m grok-code-fast-1-0825 \
  -o src/lasr_labs_2025_control_project/finetuning/experiments/grok/grok-code-fast-1-0825/results/table.jsonl
```
