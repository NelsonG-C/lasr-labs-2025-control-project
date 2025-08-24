# Simple Acausal Collusion

### Parameter Details

variant = [word, letter, collusion]

### Commands to run

Temp 0:
```bash
inspect eval src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/main.py -T epochs=1 -T model="openai/gpt-4o-mini" -T temperature=0.0 -T variant="collusion"
```

Top10:
```bash
lasr-labs-2025-control-project % inspect eval src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/main.py@acausal_collusion_top10 -T epochs=100 -T model="openai/gpt-4o-mini" -T temperature=1.0 -T variant="collusion"
```

If you want to run all of the items at once for generating logs, use the following:

```bash
uv run src/lasr_labs_2025_control_project/experiments/acausal_collusion/basic/runner.py --runs 1 --model "openai/gpt-4o-mini" --epochs 1 --seed_prompt="ari"
```
