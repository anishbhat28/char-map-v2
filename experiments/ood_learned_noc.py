from pathlib import Path
import sys, argparse, time, pandas as pd, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

from learned.ood_burgers2d import SCENARIOS, make_ood_workload
from mapping.policies import ReactivePolicy, HistoryPolicy
from mapping.burgers2d_policies import Burgers2DCharacteristicPolicy, Burgers2DOraclePolicy
from mapping.learned_burgers2d_policy import LearnedBurgers2DPolicy
from simulation.noc_simulator import NoCMappingSimulator, NoCMappingSimConfig

CKPT=ROOT/"checkpoints/burgers2d_surrogate.pt"; H=4; SEED=50000

def workload(name): return make_ood_workload(SEED,SCENARIOS[name])

def run(p,name):
    cfg=NoCMappingSimConfig(rows=8,cols=8,horizon=H,optimizer_passes=2,remap_every=1,
        bytes_per_unit=256,link_bandwidth_bytes_per_cycle=64,router_latency_cycles=1,
        alpha_byte_hops=1,beta_max_link_load=4,migration_lambda=.5,task_state_size=256,
        num_random_starts=2,multistart_seed=1337)
    df=NoCMappingSimulator(workload(name),p,cfg).run()
    return df[df.timestep>=H].copy()

def summarize(out):
    s=out.groupby("policy",as_index=False).agg(
        total_byte_hops=("byte_hops","sum"),
        mean_max_link_load=("max_link_load_bytes","mean"),
        p95_max_link_load=("max_link_load_bytes",lambda x:x.quantile(.95)),
        mean_estimated_latency=("estimated_latency_cycles","mean"),
        p95_estimated_latency=("estimated_latency_cycles",lambda x:x.quantile(.95)),
        total_migration_byte_hops=("migration_byte_hops","sum"),
        total_objective=("realized_objective","sum"),
        total_moved_tasks=("moved_tasks","sum"))
    r=s[s.policy=="reactive"].iloc[0]; p=s[s.policy=="perfect_future_graph"].iloc[0]
    s["objective_improvement_vs_reactive"]=r.total_objective/s.total_objective
    s["byte_hop_improvement_vs_reactive"]=r.total_byte_hops/s.total_byte_hops
    s["latency_improvement_vs_reactive"]=r.mean_estimated_latency/s.mean_estimated_latency
    den=r.total_objective-p.total_objective
    s["perfect_prediction_capture"]=(r.total_objective-s.total_objective)/den if abs(den)>1e-12 else float("nan")
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenario",choices=list(SCENARIOS)+["all"],default="mild")
    args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    names=list(SCENARIOS) if args.scenario=="all" else [args.scenario]
    allsum=[]
    for name in names:
        print(f"\n===== OOD scenario: {name} =====",flush=True)
        policies=[ReactivePolicy(),HistoryPolicy(),
            Burgers2DCharacteristicPolicy(name="exact_physics"),
            LearnedBurgers2DPolicy(CKPT,device=device,name="learned_dynamics"),
            Burgers2DOraclePolicy()]
        policies[-1].name="perfect_future_graph"
        frames=[]
        for i,p in enumerate(policies,1):
            t=time.perf_counter(); print(f"[{i}/{len(policies)}] {p.name} ...",flush=True)
            frames.append(run(p,name))
            print(f"    {time.perf_counter()-t:.1f} s",flush=True)
        s=summarize(pd.concat(frames,ignore_index=True)); s.insert(0,"scenario",name)
        allsum.append(s)
        print(s.sort_values("total_objective").to_string(index=False))
    out=pd.concat(allsum,ignore_index=True)
    (ROOT/"results").mkdir(exist_ok=True)
    out.to_csv(ROOT/"results/ood_noc_selected.csv",index=False)
    print("\n=== LEARNED DYNAMICS ACROSS OOD SCENARIOS ===")
    print(out[out.policy=="learned_dynamics"][["scenario","total_objective",
        "objective_improvement_vs_reactive","byte_hop_improvement_vs_reactive",
        "latency_improvement_vs_reactive","perfect_prediction_capture"]].to_string(index=False))
if __name__=="__main__": main()
