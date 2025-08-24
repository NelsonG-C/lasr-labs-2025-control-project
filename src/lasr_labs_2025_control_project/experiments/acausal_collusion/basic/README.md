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
