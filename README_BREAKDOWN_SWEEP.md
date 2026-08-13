# Breakdown-point experiment

Goal:

state prediction error
    ->
future graph error
    ->
hardware performance degradation

Do NOT retrain the learned model.

Add:
- learned/ood_breakdown.py
- experiments/breakdown_diagnostic.py
- experiments/breakdown_noc.py

Keep:
- checkpoints/burgers2d_surrogate.pt
- learned/burgers2d_surrogate.py
- learned/burgers2d_data.py
- mapping/learned_burgers2d_policy.py
- current multi-start NoC simulator

## Step 1: cheap sweep

Run:

python -m experiments.breakdown_diagnostic

This sweeps:

scenarios:
    id
    severe
    extreme1
    extreme2
    extreme3

horizons:
    4
    8
    12
    16

It reports:
- rollout MSE
- mean graph mismatch
- max graph mismatch
- exact graph fraction

The best NoC candidates are those with roughly 5-30% mean graph mismatch.

## Step 2: expensive NoC points

After inspecting the cheap table, choose only 2-3 points.

Example:

python -m experiments.breakdown_noc --scenario extreme1 --horizon 8

python -m experiments.breakdown_noc --scenario extreme2 --horizon 12

Do not blindly run every combination.

What we want:

low graph error:
    learned ~= perfect prediction

moderate graph error:
    learned still > reactive/history

large graph error:
    learned advantage eventually collapses

That gives the paper's decision-robustness curve.
