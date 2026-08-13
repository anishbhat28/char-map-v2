from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from learned.ood_hard_breakdown import SCENARIOS,make_hard_workload
from mapping.policies import ReactivePolicy
from mapping.burgers2d_policies import Burgers2DOraclePolicy
from mapping.learned_burgers2d_policy import LearnedBurgers2DPolicy
from simulation.noc_simulator import NoCMappingSimulator,NoCMappingSimConfig

CKPT=ROOT/"checkpoints/burgers2d_surrogate.pt"
SEEDS=[50000,50001,50002,50003,50004]
POINTS=[("low","extreme3",16,0.026693),
        ("moderate","hard2",24,0.107422),
        ("high","hard4",32,0.209635)]

def workload(scenario,seed):
    return make_hard_workload(seed=seed,scenario=SCENARIOS[scenario],nx=8,ny=8,timesteps=100,dt=.004,base_viscosity=.002)

def run(policy,scenario,h,seed):
    cfg=NoCMappingSimConfig(rows=8,cols=8,horizon=h,optimizer_passes=2,remap_every=1,
        bytes_per_unit=256,link_bandwidth_bytes_per_cycle=64,router_latency_cycles=1,
        alpha_byte_hops=1,beta_max_link_load=4,migration_lambda=.5,task_state_size=256,
        num_random_starts=2,multistart_seed=1337)
    df=NoCMappingSimulator(workload(scenario,seed),policy,cfg).run()
    return df[df.timestep>=h].copy()

def sm(df):
    return dict(obj=float(df.realized_objective.sum()),
                bh=float(df.byte_hops.sum()),
                lat=float(df.estimated_latency_cycles.mean()),
                moved=int(df.moved_tasks.sum()))

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    rows=[]; total=len(SEEDS)*len(POINTS); n=0
    outdir=ROOT/"results"; outdir.mkdir(exist_ok=True)
    for regime,scenario,h,gm in POINTS:
        for seed in SEEDS:
            n+=1; print(f"\n=== CASE {n}/{total}: {regime} {scenario} H={h} seed={seed} ===",flush=True)
            ps=[ReactivePolicy(),
                LearnedBurgers2DPolicy(CKPT,device=device,name="learned_dynamics"),
                Burgers2DOraclePolicy()]
            ps[-1].name="perfect_future_graph"
            m={}
            for i,p in enumerate(ps,1):
                t=time.perf_counter(); print(f"[{i}/3] {p.name} ...",flush=True)
                m[p.name]=sm(run(p,scenario,h,seed))
                print(f"    {time.perf_counter()-t:.1f} s",flush=True)
            r,l,p=m["reactive"],m["learned_dynamics"],m["perfect_future_graph"]
            den=r["obj"]-p["obj"]
            rows.append(dict(regime=regime,scenario=scenario,horizon=h,seed=seed,
                graph_mismatch=gm,
                objective_improvement_vs_reactive=r["obj"]/l["obj"],
                latency_improvement_vs_reactive=r["lat"]/l["lat"],
                byte_hop_improvement_vs_reactive=r["bh"]/l["bh"],
                perfect_prediction_capture=(r["obj"]-l["obj"])/den if abs(den)>1e-12 else np.nan))
            pd.DataFrame(rows).to_csv(outdir/"multiseed_validation_raw.csv",index=False)
    raw=pd.DataFrame(rows)
    s=raw.groupby(["regime","scenario","horizon"],as_index=False).agg(
        graph_mismatch=("graph_mismatch","mean"),
        objective_improvement_mean=("objective_improvement_vs_reactive","mean"),
        objective_improvement_std=("objective_improvement_vs_reactive","std"),
        latency_improvement_mean=("latency_improvement_vs_reactive","mean"),
        latency_improvement_std=("latency_improvement_vs_reactive","std"),
        perfect_capture_mean=("perfect_prediction_capture","mean"),
        perfect_capture_std=("perfect_prediction_capture","std"))
    s.to_csv(outdir/"multiseed_validation_summary.csv",index=False)
    print("\n=== 5-SEED VALIDATION SUMMARY ===")
    print(s.to_string(index=False))
    print("\n=== PAPER-FRIENDLY VIEW ===")
    for _,r in s.iterrows():
        print(f"{r.regime}: mismatch~{100*r.graph_mismatch:.1f}% | objective {r.objective_improvement_mean:.3f} +/- {r.objective_improvement_std:.3f}x | latency {r.latency_improvement_mean:.3f} +/- {r.latency_improvement_std:.3f}x | capture {100*r.perfect_capture_mean:.1f} +/- {100*r.perfect_capture_std:.1f}%")
if __name__=="__main__": main()
