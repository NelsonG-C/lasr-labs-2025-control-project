### Running the evaluation

For base models:

Untrusted Generator
```bash
uv run src/lasr_labs_2025_control_project/apps/runners/self_rec/attack.py
```

Honeypots (Trusted Generator)
```bash
uv run src/lasr_labs_2025_control_project/apps/runners/self_rec/honeypot.py
```

Human honeypot uses a base human solution from the APPS dataset and then has an LLM put a backdoor into it.
```bash
uv run src/lasr_labs_2025_control_project/apps/runners/self_rec/human_honeypot.py
```
