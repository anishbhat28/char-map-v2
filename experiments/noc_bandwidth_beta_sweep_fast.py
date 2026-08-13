from pathlib import Path
import sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from workloads.advection import RegimeSwitchingAdvection
from mapping.policies import StaticPolicy, ReactivePolicy, HistoryPolicy, CharacteristicPolicy, OraclePolicy
from simulation.noc_simulator import NoCMappingSimulator, NoCMappingSimConfig

BETAS = [0.0, 1.0, 2.0, 4.0, 8.0]
BANDWIDTHS = [16.0, 32.0, 64.0, 128.0, 256.0]
HORIZON = 4
PHYSICS_ERROR = 0.05
BYTES_PER_UNIT = 256.0
ROUTER_LATENCY = 1.0
ALPHA = 1.0
MIGRATION_LAMBDA = 0.5
TASK_STATE_SIZE = 256.0
REFERENCE_BW = 64.0

def make_workload():
    return RegimeSwitchingAdvection(
        num_cells=64, timesteps=100, dt=1/64, c0=1.0, omega=4.0,
        amplitude_1=.25, amplitude_2=.75, amplitude_3=.40,
        rk4_substeps_per_dt=8)

def run_mapping_once(policy, beta):
    cfg = NoCMappingSimConfig(
        rows=8, cols=8, horizon=HORIZON, optimizer_passes=4, remap_every=1,
        bytes_per_unit=BYTES_PER_UNIT,
        link_bandwidth_bytes_per_cycle=REFERENCE_BW,
        router_latency_cycles=ROUTER_LATENCY,
        alpha_byte_hops=ALPHA,
        beta_max_link_load=beta,
        migration_lambda=MIGRATION_LAMBDA,
        task_state_size=TASK_STATE_SIZE)
    df = NoCMappingSimulator(make_workload(), policy, cfg).run()
    return df[df.timestep >= HORIZON].copy()

def eval_bw(df, bw, beta, policy):
    ser = df.max_link_load_bytes / bw
    lat = ser + ROUTER_LATENCY * df.max_hops
    return {
        "policy":policy, "bandwidth":bw, "beta":beta,
        "total_byte_hops":float(df.byte_hops.sum()),
        "mean_max_link_load":float(df.max_link_load_bytes.mean()),
        "p95_max_link_load":float(df.max_link_load_bytes.quantile(.95)),
        "mean_latency":float(lat.mean()),
        "p95_latency":float(lat.quantile(.95)),
        "total_migration_byte_hops":float(df.migration_byte_hops.sum()),
        "total_objective":float(df.realized_objective.sum()),
        "total_moved_tasks":int(df.moved_tasks.sum())
    }

def main():
    policies = [
        StaticPolicy(), ReactivePolicy(), HistoryPolicy(),
        CharacteristicPolicy(amplitude_error=PHYSICS_ERROR), OraclePolicy()
    ]
    total = len(BETAS)*len(policies)
    rows=[]; idx=0; t0=time.perf_counter()
    print(f"Expensive mapping runs: {total}", flush=True)

    for beta in BETAS:
        for p in policies:
            idx += 1
            s=time.perf_counter()
            print(f"[{idx}/{total}] beta={beta:g}, policy={p.name} ...", flush=True)
            df=run_mapping_once(p,beta)
            print(f"    finished in {time.perf_counter()-s:.1f} s", flush=True)
            for bw in BANDWIDTHS:
                rows.append(eval_bw(df,bw,beta,p.name))

    out=pd.DataFrame(rows)
    enriched=[]
    for (_, _),case in out.groupby(["beta","bandwidth"]):
        case=case.copy()
        r=case[case.policy=="reactive"].iloc[0]
        o=case[case.policy=="oracle"].iloc[0]
        case["objective_improvement_vs_reactive"]=r.total_objective/case.total_objective
        case["byte_hop_improvement_vs_reactive"]=r.total_byte_hops/case.total_byte_hops
        case["latency_improvement_vs_reactive"]=r.mean_latency/case.mean_latency
        denom=r.total_objective-o.total_objective
        case["oracle_capture_vs_reactive"]=(
            (r.total_objective-case.total_objective)/denom
            if abs(denom)>1e-12 else float("nan"))
        enriched.append(case)
    out=pd.concat(enriched,ignore_index=True)
    char=out[out.policy=="characteristic"].copy()

    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"plots").mkdir(exist_ok=True)
    out.to_csv(ROOT/"results/noc_bandwidth_beta_sweep_fast.csv",index=False)
    char.to_csv(ROOT/"results/noc_bandwidth_beta_charmap_fast.csv",index=False)

    p=char.pivot(index="beta",columns="bandwidth",values="latency_improvement_vs_reactive")
    plt.imshow(p.values,origin="lower",aspect="auto")
    plt.xticks(range(len(p.columns)),[int(x) for x in p.columns])
    plt.yticks(range(len(p.index)),p.index)
    plt.xlabel("Link bandwidth (bytes/cycle)")
    plt.ylabel("Congestion weight beta")
    plt.colorbar(label="Reactive latency / Char-Map latency")
    plt.tight_layout()
    plt.savefig(ROOT/"plots/noc_latency_heatmap_fast.png",dpi=180)
    plt.close()

    print("\n=== CHARMAP FAST NoC SWEEP ===")
    print(char[[
        "bandwidth","beta","total_byte_hops","mean_max_link_load","mean_latency","p95_latency",
        "total_migration_byte_hops","total_objective","objective_improvement_vs_reactive",
        "byte_hop_improvement_vs_reactive","latency_improvement_vs_reactive",
        "oracle_capture_vs_reactive"
    ]].sort_values(["bandwidth","beta"]).to_string(index=False))
    print(f"\nTotal wall time: {time.perf_counter()-t0:.1f} s")

if __name__=="__main__":
    main()
