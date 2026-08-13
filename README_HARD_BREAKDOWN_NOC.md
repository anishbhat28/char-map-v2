# Hard-breakdown NoC runner

Add:

```text
experiments/hard_breakdown_noc.py
```

Keep:
- learned/ood_hard_breakdown.py
- checkpoints/burgers2d_surrogate.pt
- current learned dynamics and NoC files

Recommended three runs:

```bash
python -m experiments.hard_breakdown_noc --scenario extreme3 --horizon 16
python -m experiments.hard_breakdown_noc --scenario hard2 --horizon 24
python -m experiments.hard_breakdown_noc --scenario hard4 --horizon 32
```

These correspond approximately to:
- low graph mismatch ~2.7%
- moderate graph mismatch ~10.7%
- high graph mismatch ~21.0%
