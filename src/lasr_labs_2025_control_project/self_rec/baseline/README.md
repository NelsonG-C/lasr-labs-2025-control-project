# Self-Recognition Evaluation Project

This project implements self-recognition experiments for AI models using programming problem datasets. It includes tools for generating solutions from different models and evaluating their ability to recognize their own work.

## Overview

The project consists of three main components:

1. **Solution Generation**: Generate solutions to programming problems using different AI models
2. **Data Merging**: Combine solutions from multiple models for comparison
3. **Self-Recognition Evaluation**: Test models' ability to identify their own solutions

## Setup

### Prerequisites

- Python environment with required dependencies
- API keys for OpenAI and/or Anthropic
- Access to the `inspect` evaluation framework

### Environment Variables

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

install dependencies
```bash
uv sync
```

## Usage

### 1. Generate Solutions

The solution generation script creates programming solutions from AI models using the APPS dataset.

#### Configuration

Edit the script parameters at the top of the generation script:

```python
num_samples = 100  # Number of problems to solve
model_provider = "anthropic"  # or "openai"
model_name = "claude-3-5-haiku-20241022"  # Model identifier
```

#### Run Solution Generation

Experiment Runs:
o4-mini normal (correct solution is B):
```bash
inspect eval src/lasr_labs_2025_control_project/experiments/self_rec/self_rec_eval.py@binary_self_rec --model openai/o4-mini --limit 10 -T file_path="data/merged/claude-3-5-haiku-20241022_o4-mini_code_formatted.json" -T reversed=False -T match_string="openai/o4-mini"
```

o4-mini reversed (correct solution is A):
```bash
inspect eval src/lasr_labs_2025_control_project/experiments/self_rec/self_rec_eval.py@binary_self_rec --model openai/o4-mini --limit 10 -T file_path="data/merged/claude-3-5-haiku-20241022_o4-mini_code_formatted.json" -T reversed=True -T match_string="openai/o4-mini"
```

claude-3-5-haiku normal (correct solution is A):
```bash
inspect eval src/lasr_labs_2025_control_project/experiments/self_rec/self_rec_eval.py@binary_self_rec --model anthropic/claude-3-5-haiku-20241022 --limit 100 -T file_path="data/merged/claude-3-5-haiku-20241022_o4-mini_code_formatted.json" -T reversed=True -T match_string="claude-3-5-haiku-20241022"
```

claude-3-5-haiku normal (correct solution is A):
```bash
inspect eval src/lasr_labs_2025_control_project/experiments/self_rec/self_rec_eval.py@binary_self_rec --model anthropic/claude-3-5-haiku-20241022 --limit 100 -T file_path="data/merged/claude-3-5-haiku-20241022_o4-mini_code_formatted.json" -T reversed=False -T match_string="claude-3-5-haiku-20241022"
```


```bash
uv run src/lasr_labs_2025_control_project/experiments/self_rec/utils/data_generation.py
```

This will:
- Load the first 100 problems from the APPS dataset
- Generate solutions using the specified model
- Save results to `data/{model_name}/all_solutions.json`

### 2. Merge Solutions from Multiple Models

After generating solutions from different models, use the merge script to combine them:

```bash
python merge_solutions.py
```

This creates merged datasets with solutions from both models:
- `data/{model_a}_{model_b}.json`
- `data/{model_b}_{model_a}.json`

### 3. Run Self-Recognition Evaluations

#### Individual Self-Recognition

Test a model's ability to recognize its own solutions:

```bash
inspect eval src/lasr_labs_2025_control_project/experiments/self_rec/self_rec_eval.py@individual_self_rec --model openai/o4-mini -T file_path="data/o4-mini/all_solutions.json"
```

#### Binary Self-Recognition

Compare two models' solutions and test recognition ability:

**Using Claude 3.5 Haiku as the evaluator:**
```bash
inspect eval src/lasr_labs_2025_control_project/experiments/self_rec/self_rec_eval.py@binary_self_rec --model anthropic/claude-3-5-haiku-20241022 --limit 10 -T file_path="data/merged_solutions.json" -T reversed=false
```

**Using o4-mini as the evaluator:**
```bash
inspect eval src/lasr_labs_2025_control_project/experiments/self_rec/self_rec_eval.py@binary_self_rec --model openai/o4-mini --limit 10 -T file_path="data/merged_solutions.json" -T reversed=false
```

## Parameters

### Evaluation Parameters

- `--model`: Specify the model to use for evaluation
- `--limit`: Limit the number of samples to evaluate (useful for testing)
- `-T file_path`: Path to the data file containing solutions
- `-T reversed`: Boolean flag to reverse the order of comparison

### File Structure

```
data/
├── o4-mini/
│   └── all_solutions.json
├── claude-3-5-haiku-20241022/
│   └── all_solutions.json
├── merged_solutions.json
└── {model_a}_{model_b}.json
```

## Data Format

### Individual Solutions File
```json
[
  {
    "problem_id": "unique_id",
    "problem": "programming problem description",
    "solution": "generated code solution",
    "model": "model_name"
  }
]
```

### Merged Solutions File
```json
[
  {
    "problem_id": "unique_id",
    "problem": "programming problem description",
    "solution_A": "solution from model A",
    "model_A": "model_a_name",
    "solution_B": "solution from model B",
    "model_B": "model_b_name"
  }
]
```

## Supported Models

- **OpenAI**: `openai/o4-mini`
- **Anthropic**: `anthropic/claude-3-5-haiku-20241022`

## Notes

- The generation script uses the APPS dataset with "simple" difficulty level
- Solutions are cached to avoid regenerating identical problems
- The merge script ensures matching problem IDs between datasets
- All results include metadata for tracking and analysis

## Troubleshooting

- Ensure API keys are properly set in your `.env` file
- Verify that the data files exist before running evaluations
- Check that the `inspect` framework is properly installed and configured
- For merge operations, both input files must contain matching problem IDs
