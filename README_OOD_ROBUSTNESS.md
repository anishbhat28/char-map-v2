Morning workflow:

1) Put burgers2d_surrogate.pt in:
   checkpoints/burgers2d_surrogate.pt

2) Cheap diagnostic:
   python -m experiments.ood_learned_diagnostics

3) Then run OOD NoC one scenario at a time:
   python -m experiments.ood_learned_noc --scenario mild
   python -m experiments.ood_learned_noc --scenario moderate
   python -m experiments.ood_learned_noc --scenario severe

Optional:
   python -m experiments.ood_learned_noc --scenario all

Do not retrain the model. We want to measure robustness of the already-trained
learned dynamics model under distribution shift.
