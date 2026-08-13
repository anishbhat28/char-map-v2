from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from workloads.advection import RegimeSwitchingAdvection
from mapping.policies import StaticPolicy, ReactivePolicy, HistoryPolicy, CharacteristicPolicy, OraclePolicy
from simulation.noc_simulator import NoCMappingSimulator, NoCMappingSimConfig

BANDWIDTHS = [16.0, 32.0, 64.0, 128.0, 256.0]
BETAS = [0.0, 1.0, 2.0, 4.0, 8.0]
HORIZON = 4
PHYSICS_ERROR = 0.05
BYTES_PER_UNIT = 256.0
ROUTER_LATENCY = 1.0
ALPHA = 1.0
MIGRATION_LAMBDA = 0.5
TASK_STATE_SIZE = 256.0

def make_workload():
    return RegimeSwitchingAdvection(
        num_cells=64, timesteps=100, dt=1/64, c0=1.0, omega=4.0,
        amplitude_1=.25, amplitude_2=.75, amplitude_3=.40,
        rk4_substeps_per_dt=8,
    )

def run_policy(policy, bandwidth, beta):
    cfg = NoCMappingSimConfig(
        rows=8, cols=8, horizon=HORIZON, optimizer_passes=4, remap_every=1,
        bytes_per_unit=BYTES_PER_UNIT,
        link_bandwidth_bytes_per_cycle=bandwidth,
        router_latency_cycles=ROUTER_LATENCY,
        alpha_byte_hops=ALPHA,
        beta_max_link_load=beta,
        migration_lambda=MIGRATION_LAMBDA,
        task_state_size=TASK_STATE_SIZE,
    )
    df = NoCMappingSimulator(make_workload(), policy, cfg).run()
    df = df[df.timestep >= HORIZON].copy()
    return {
        "policy": policy.name,
        "bandwidth": bandwidth,
        "beta": beta,
        "total_byte_hops": float(df.byte_hops.sum()),
        "mean_max_link_load": float(df.max_link_load_bytes.mean()),
        "p95_max_link_load": float(df.max_link_load_bytes.quantile(.95)),
        "mean_latency": float(df.estimated_latency_cycles.mean()),
        "p95_latency": float(df.estimated_latency_cycles.quantile(.95)),
        "total_migration_byte_hops": float(df.migration_byte_hops.sum()),
        "total_objective": float(df.realized_objective.sum()),
        "total_moved_tasks": int(df.moved_tasks.sum()),
    }

def main():
    blocks = []
    for bw in BANDWIDTHS:
        for beta in BETAS:
            policies = [
                StaticPolicy(), ReactivePolicy(), HistoryPolicy(),
                CharacteristicPolicy(amplitude_error=PHYSICS_ERROR),
                OraclePolicy(),
            ]
            case = pd.DataFrame([run_policy(p, bw, beta) for p in policies])
            r = case[case.policy=="reactive"].iloc[0]
            o = case[case.policy=="oracle"].iloc[0]

            case["objective_improvement_vs_reactive"] = r.total_objective / case.total_objective
            case["byte_hop_improvement_vs_reactive"] = r.total_byte_hops / case.total_byte_hops
            case["latency_improvement_vs_reactive"] = r.mean_latency / case.mean_latency

            denom = r.total_objective - o.total_objective
            case["oracle_capture_vs_reactive"] = (
                (r.total_objective - case.total_objective) / denom
                if abs(denom) > 1e-12 else float("nan")
            )
            blocks.append(case)

    out = pd.concat(blocks, ignore_index=True)
    char = out[out.policy=="characteristic"].copy()

    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"plots").mkdir(exist_ok=True)
    out.to_csv(ROOT/"results/noc_bandwidth_beta_sweep.csv", index=False)
    char.to_csv(ROOT/"results/noc_bandwidth_beta_charmap.csv", index=False)

    p = char.pivot(index="beta", columns="bandwidth", values="latency_improvement_vs_reactive")
    plt.imshow(p.values, origin="lower", aspect="auto")
    plt.xticks(range(len(p.columns)), [int(x) for x in p.columns])
    plt.yticks(range(len(p.index)), p.index)
    plt.xlabel("Link bandwidth (bytes/cycle)")
    plt.ylabel("Congestion weight beta")
    plt.colorbar(label="Reactive latency / Char-Map latency")
    plt.tight_layout()
    plt.savefig(ROOT/"plots/noc_latency_heatmap.png", dpi=180)
    plt.close()

    p = char.pivot(index="beta", columns="bandwidth", values="objective_improvement_vs_reactive")
    plt.imshow(p.values, origin="lower", aspect="auto")
    plt.xticks(range(len(p.columns)), [int(x) for x in p.columns])
    plt.yticks(range(len(p.index)), p.index)
    plt.xlabel("Link bandwidth (bytes/cycle)")
    plt.ylabel("Congestion weight beta")
    plt.colorbar(label="Reactive objective / Char-Map objective")
    plt.tight_layout()
    plt.savefig(ROOT/"plots/noc_objective_heatmap.png", dpi=180)
    plt.close()

    print("\n=== CHARMAP NoC BANDWIDTH x CONGESTION SWEEP ===")
    print(char[[
        "bandwidth","beta","total_byte_hops","mean_max_link_load","mean_latency","p95_latency",
        "total_migration_byte_hops","total_objective","objective_improvement_vs_reactive",
        "byte_hop_improvement_vs_reactive","latency_improvement_vs_reactive",
        "oracle_capture_vs_reactive"
    ]].sort_values(["bandwidth","beta"]).to_string(index=False))

if __name__ == "__main__":
    main()
