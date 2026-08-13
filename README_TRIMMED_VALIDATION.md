Add experiments/trimmed_multiseed_validation.py.

Run:
python -m experiments.trimmed_multiseed_validation

It reads results/multiseed_validation_raw.csv if present, preserves existing rows,
and runs only:

moderate: hard2 H=24, seeds 50000-50002
high:     hard4 H=32, seeds 50000-50002

It saves after every completed case.
