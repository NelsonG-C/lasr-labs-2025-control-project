### Running the experiment

From the root, run the following command:

```bash
uv run inspect eval src/lasr_labs_2025_control_project/experiments/acausal_collusion/main.py --limit 1 --max-samples 1
```

Note: max_samples is important here for replication of the focal points paper as it needs to build up the auditing samples and use this for prompt context.

Any other experiments should consider changing this synchronous reliance for easier experiment design and running considerations (if possible)
