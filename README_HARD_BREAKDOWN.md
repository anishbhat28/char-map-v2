# Hard breakdown diagnostic

Add:
- learned/ood_hard_breakdown.py
- experiments/hard_breakdown_diagnostic.py

Do NOT retrain the learned model.
Do NOT run expensive NoC yet.

Run:

python -m experiments.hard_breakdown_diagnostic

This pushes beyond the previous sweep:

Scenarios:
- extreme3
- hard1
- hard2
- hard3
- hard4

Horizons:
- 16
- 24
- 32

Goal:
find representative points in three graph-error regimes:

low:
    1-5%

moderate:
    5-15%

high:
    15-35%

Then use only one point from each regime for expensive NoC runs.
