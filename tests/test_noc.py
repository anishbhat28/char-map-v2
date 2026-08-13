
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from architecture.mesh import Mesh
from architecture.noc import NoCConfig,xy_route,route_graph
m=Mesh(4,4); assert len(xy_route(m,0,15))==6
metrics,_=route_graph(m,{(0,15):1.0},{i:i for i in range(16)},
    NoCConfig(bytes_per_unit=256,link_bandwidth_bytes_per_cycle=64,router_latency_cycles=1))
assert metrics["byte_hops"]==1536
assert metrics["estimated_latency_cycles"]==10
print("All NoC smoke tests passed.")
