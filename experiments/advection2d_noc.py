
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
from workloads.advection2d import RotatingAdvection2D
from mapping.policies import StaticPolicy,ReactivePolicy,HistoryPolicy
from mapping.advection2d_policies import Characteristic2DPolicy,Oracle2DPolicy
from simulation.noc_simulator import NoCMappingSimulator,NoCMappingSimConfig

H=4
def workload():
    return RotatingAdvection2D(nx=6,ny=6,timesteps=60,dt=.01,omega=1.5,shear=.35,temporal_frequency=3.0)

def run(p):
    cfg=NoCMappingSimConfig(rows=6,cols=6,horizon=H,optimizer_passes=2,remap_every=1,
        bytes_per_unit=256,link_bandwidth_bytes_per_cycle=64,router_latency_cycles=1,
        alpha_byte_hops=1,beta_max_link_load=4,migration_lambda=.5,task_state_size=256)
    df=NoCMappingSimulator(workload(),p,cfg).run()
    return df[df.timestep>=H].copy()

def main():
    policies=[StaticPolicy(),ReactivePolicy(),HistoryPolicy(),
        Characteristic2DPolicy(name="physics2d_exact"),
        Characteristic2DPolicy(name="physics2d_imperfect",omega_error=.03,shear_error=.08,frequency_error=.05),
        Characteristic2DPolicy(name="physics2d_no_shear",remove_shear=True),
        Oracle2DPolicy()]
    frames=[]
    for p in policies:
        print(f"Running {p.name} ...",flush=True); frames.append(run(p))
    out=pd.concat(frames,ignore_index=True)
    (ROOT/"results").mkdir(exist_ok=True)
    s=out.groupby("policy",as_index=False).agg(
        total_byte_hops=("byte_hops","sum"),
        mean_max_link_load=("max_link_load_bytes","mean"),
        p95_max_link_load=("max_link_load_bytes",lambda x:x.quantile(.95)),
        mean_estimated_latency=("estimated_latency_cycles","mean"),
        p95_estimated_latency=("estimated_latency_cycles",lambda x:x.quantile(.95)),
        total_migration_byte_hops=("migration_byte_hops","sum"),
        total_objective=("realized_objective","sum"),
        total_moved_tasks=("moved_tasks","sum"))
    r=s[s.policy=="reactive"].iloc[0]; o=s[s.policy=="oracle2d"].iloc[0]
    s["objective_improvement_vs_reactive"]=r.total_objective/s.total_objective
    s["byte_hop_improvement_vs_reactive"]=r.total_byte_hops/s.total_byte_hops
    s["latency_improvement_vs_reactive"]=r.mean_estimated_latency/s.mean_estimated_latency
    denom=r.total_objective-o.total_objective
    s["oracle_capture_vs_reactive"]=(r.total_objective-s.total_objective)/denom
    exact=s[s.policy=="physics2d_exact"].iloc[0]
    for c in ["total_byte_hops","mean_max_link_load","p95_max_link_load","mean_estimated_latency",
              "p95_estimated_latency","total_migration_byte_hops","total_objective","total_moved_tasks"]:
        assert abs(float(exact[c])-float(o[c]))<1e-9, f"Exact 2D physics != oracle for {c}"
    s.to_csv(ROOT/"results/advection2d_noc_summary.csv",index=False)
    print("\\n=== 2-D ADVECTION NoC ===")
    print(s.sort_values("total_objective").to_string(index=False))
    print("\\nExact 2-D physics matches Oracle.")
if __name__=="__main__": main()
