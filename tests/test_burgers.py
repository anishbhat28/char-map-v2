from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from workloads.burgers import BurgersWorkload
from mapping.burgers_policies import BurgersCharacteristicPolicy,BurgersOraclePolicy

w=BurgersWorkload(num_cells=24,timesteps=20,dt=.003,viscosity=.002)
assert len(w.state_history)==21
assert all(np.all(np.isfinite(s)) for s in w.state_history)
assert sum(w.communication_graph(10).values())==24
p=BurgersCharacteristicPolicy(name="exact")
o=BurgersOraclePolicy()
gp=p.predicted_graph(workload=w,timestep=5,horizon=3,observed_graphs=[])
go=o.predicted_graph(workload=w,timestep=5,horizon=3,observed_graphs=[])
assert gp==go, "Exact Burgers predictor did not reproduce Oracle graph"
print("All Burgers smoke tests passed.")
