from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from architecture.mesh import Mesh
from workloads.advection import RegimeSwitchingAdvection
from mapping.optimizer import greedy_swap_optimize
from mapping.policies import CharacteristicPolicy
from simulation.simulator import MappingSimulator,MappingSimConfig
def main():
    m=Mesh(8,8); assert m.distance(0,63)==14
    w=RegimeSwitchingAdvection(num_cells=16,timesteps=8); g=w.communication_graph(5); assert g
    p={i:i for i in range(16)}; m2=Mesh(4,4); before=m2.weighted_manhattan_cost(g,p);_, after, _, _ = greedy_swap_optimize(m2, g, p, 1);assert after<=before
    df=MappingSimulator(w,CharacteristicPolicy(),MappingSimConfig(rows=4,cols=4,horizon=1,optimizer_passes=1)).run();assert len(df)==8
    print('All smoke tests passed.')
if __name__=='__main__':main()
