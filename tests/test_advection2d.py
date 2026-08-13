
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from workloads.advection2d import RotatingAdvection2D
from mapping.advection2d_policies import Characteristic2DPolicy,Oracle2DPolicy
w=RotatingAdvection2D(nx=4,ny=4,timesteps=20,dt=.01)
assert w.num_cells==16
assert sum(w.communication_graph(10).values())==16
p=Characteristic2DPolicy(name="exact"); o=Oracle2DPolicy()
gp=p.predicted_graph(workload=w,timestep=5,horizon=3,observed_graphs=[])
go=o.predicted_graph(workload=w,timestep=5,horizon=3,observed_graphs=[])
assert gp==go
print("All 2-D advection smoke tests passed.")
