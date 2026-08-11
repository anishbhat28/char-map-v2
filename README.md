# char-map-v2

Minimal Phase II falsification repo for characteristic-aware spatial mapping.

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/test_smoke.py
python experiments/regime_switch.py
```

Outputs: `results/regime_switch_summary.csv`, `results/regime_switch_timestep.csv`, `plots/regime_switch_cost.png`.

This first version intentionally uses only weighted Manhattan communication cost. If characteristic mapping moves materially toward oracle, next add XY routing, finite link bandwidth, congestion, SRAM, and remapping costs.
