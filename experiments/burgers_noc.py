from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd, matplotlib.pyplot as plt
from workloads.burgers import BurgersWorkload
from mapping.policies import StaticPolicy,ReactivePolicy,HistoryPolicy
from mapping.burgers_policies import BurgersCharacteristicPolicy,BurgersOraclePolicy
from simulation.noc_simulator import NoCMappingSimulator,NoCMappingSimConfig

H=4

def workload():
    return BurgersWorkload(num_cells=36,timesteps=60,dt=.003,viscosity=.002)

def run(p):
    cfg=NoCMappingSimConfig(rows=6,cols=6,horizon=H,optimizer_passes=2,remap_every=1,
        bytes_per_unit=256,link_bandwidth_bytes_per_cycle=64,router_latency_cycles=1,
        alpha_byte_hops=1,beta_max_link_load=4,migration_lambda=.5,task_state_size=256)
    df=NoCMappingSimulator(workload(),p,cfg).run()
    return df[df.timestep>=H].copy()

def main():
    policies=[
        StaticPolicy(),ReactivePolicy(),HistoryPolicy(),
        BurgersCharacteristicPolicy(name="burgers_physics_exact"),
        BurgersCharacteristicPolicy(name="burgers_physics_imperfect",
            advection_error=.03,viscosity_error=.10,state_scale_error=.02),
        BurgersOraclePolicy()
    ]
    frames=[]
    for p in policies:
        print(f"Running {p.name} ...",flush=True)
        frames.append(run(p))
    out=pd.concat(frames,ignore_index=True)
    (ROOT/"results").mkdir(exist_ok=True); (ROOT/"plots").mkdir(exist_ok=True)
    out.to_csv(ROOT/"results/burgers_noc_timestep.csv",index=False)
    s=out.groupby("policy",as_index=False).agg(
        total_byte_hops=("byte_hops","sum"),
        mean_max_link_load=("max_link_load_bytes","mean"),
        p95_max_link_load=("max_link_load_bytes",lambda x:x.quantile(.95)),
        mean_estimated_latency=("estimated_latency_cycles","mean"),
        p95_estimated_latency=("estimated_latency_cycles",lambda x:x.quantile(.95)),
        total_migration_byte_hops=("migration_byte_hops","sum"),
        total_objective=("realized_objective","sum"),
        total_moved_tasks=("moved_tasks","sum"))
    r=s[s.policy=="reactive"].iloc[0]; o=s[s.policy=="burgers_oracle"].iloc[0]
    s["objective_improvement_vs_reactive"]=r.total_objective/s.total_objective
    s["byte_hop_improvement_vs_reactive"]=r.total_byte_hops/s.total_byte_hops
    s["latency_improvement_vs_reactive"]=r.mean_estimated_latency/s.mean_estimated_latency
    denom=r.total_objective-o.total_objective
    s["oracle_capture_vs_reactive"]=(r.total_objective-s.total_objective)/denom
    s.to_csv(ROOT/"results/burgers_noc_summary.csv",index=False)
    exact=s[s.policy=="burgers_physics_exact"].iloc[0]
    cols=["total_byte_hops","mean_max_link_load","p95_max_link_load",
          "mean_estimated_latency","p95_estimated_latency",
          "total_migration_byte_hops","total_objective","total_moved_tasks"]
    for c in cols:
        if abs(float(exact[c])-float(o[c]))>1e-9:
            raise AssertionError(f"Exact Burgers physics != Oracle for {c}")
    print("\n=== BURGERS NoC ===")
    print(s.sort_values("total_objective").to_string(index=False))
    print("\nExact Burgers physics matches Oracle.")
    w=workload()
    for t in [0,15,30,45,60]:
        plt.plot(w.x,w.state_history[t],label=f"t={t}")
    plt.xlabel("x"); plt.ylabel("u(x,t)"); plt.legend(); plt.tight_layout()
    plt.savefig(ROOT/"plots/burgers_state_evolution.png",dpi=180); plt.close()

if __name__=="__main__": main()
