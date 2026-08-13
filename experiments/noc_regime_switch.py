
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd, matplotlib.pyplot as plt
from workloads.advection import RegimeSwitchingAdvection
from mapping.policies import StaticPolicy,ReactivePolicy,HistoryPolicy,CharacteristicPolicy,OraclePolicy
from simulation.noc_simulator import NoCMappingSimulator,NoCMappingSimConfig

def workload():
    return RegimeSwitchingAdvection(num_cells=64,timesteps=100,dt=1/64,c0=1.0,omega=4.0,
        amplitude_1=.25,amplitude_2=.75,amplitude_3=.40,rk4_substeps_per_dt=8)

def run(p):
    cfg=NoCMappingSimConfig(horizon=4,optimizer_passes=4,bytes_per_unit=256,
        link_bandwidth_bytes_per_cycle=64,router_latency_cycles=1,
        alpha_byte_hops=1,beta_max_link_load=4,migration_lambda=.5,task_state_size=256)
    df=NoCMappingSimulator(workload(),p,cfg).run()
    return df[df.timestep>=cfg.horizon].copy()

def main():
    policies=[StaticPolicy(),ReactivePolicy(),HistoryPolicy(),
              CharacteristicPolicy(amplitude_error=0.05),OraclePolicy()]
    out=pd.concat([run(p) for p in policies],ignore_index=True)
    (ROOT/"results").mkdir(exist_ok=True); (ROOT/"plots").mkdir(exist_ok=True)
    out.to_csv(ROOT/"results/noc_regime_timestep.csv",index=False)
    s=out.groupby("policy",as_index=False).agg(
        total_byte_hops=("byte_hops","sum"),
        mean_max_link_load=("max_link_load_bytes","mean"),
        p95_max_link_load=("max_link_load_bytes",lambda x:x.quantile(.95)),
        mean_estimated_latency=("estimated_latency_cycles","mean"),
        p95_estimated_latency=("estimated_latency_cycles",lambda x:x.quantile(.95)),
        total_migration_byte_hops=("migration_byte_hops","sum"),
        total_objective=("realized_objective","sum"),
        total_moved_tasks=("moved_tasks","sum"))
    r=s[s.policy=="reactive"].iloc[0]
    s["objective_improvement_vs_reactive"]=r.total_objective/s.total_objective
    s["byte_hop_improvement_vs_reactive"]=r.total_byte_hops/s.total_byte_hops
    s["latency_improvement_vs_reactive"]=r.mean_estimated_latency/s.mean_estimated_latency
    s.to_csv(ROOT/"results/noc_regime_summary.csv",index=False)
    for p in out.policy.unique():
        d=out[out.policy==p]; plt.plot(d.timestep,d.max_link_load_bytes,label=p)
    plt.legend(); plt.xlabel("Timestep"); plt.ylabel("Max directed-link load (bytes)")
    plt.tight_layout(); plt.savefig(ROOT/"plots/noc_max_link_load.png",dpi=180); plt.close()
    print("\\n=== XY NoC + CONTENTION ===")
    print(s.sort_values("total_objective").to_string(index=False))
if __name__=="__main__":main()
