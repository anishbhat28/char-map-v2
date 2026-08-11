from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from workloads.advection import RegimeSwitchingAdvection
from mapping.policies import StaticPolicy, ReactivePolicy, HistoryPolicy, CharacteristicPolicy, OraclePolicy
from simulation.simulator import MappingSimulator, MappingSimConfig

LAMBDAS = [0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0]
HORIZON = 4
PHYSICS_ERROR = 0.0

def make_workload():
    return RegimeSwitchingAdvection(
        num_cells=64, timesteps=100, dt=1.0/64.0, c0=1.0, omega=4.0,
        amplitude_1=0.25, amplitude_2=0.75, amplitude_3=0.40,
        rk4_substeps_per_dt=8,
    )

def run_policy(policy, migration_lambda):
    workload = make_workload()
    cfg = MappingSimConfig(
        rows=8, cols=8, horizon=HORIZON, optimizer_passes=4, remap_every=1,
        migration_lambda=migration_lambda, task_state_size=1.0,
    )
    df = MappingSimulator(workload, policy, cfg).run()
    df = df[df["timestep"] >= HORIZON].copy()

    if policy.name == "static":
        mig = float(df["activation_migration_cost"].sum())
        if abs(mig) > 1e-9:
            raise AssertionError(f"Static mapper migrated: {mig}")

    return {
        "policy": policy.name,
        "migration_lambda": migration_lambda,
        "total_comm_cost": float(df["actual_comm_cost"].sum()),
        "total_migration_cost": float(df["activation_migration_cost"].sum()),
        "total_weighted_cost": float(df["actual_total_cost"].sum()),
        "mean_moved_tasks": float(df["activation_moved_tasks"].mean()),
        "total_moved_task_events": int(df["activation_moved_tasks"].sum()),
        "p95_total_step_cost": float(df["actual_total_cost"].quantile(0.95)),
    }

def main():
    all_cases = []
    for lam in LAMBDAS:
        policies = [
            StaticPolicy(),
            ReactivePolicy(),
            HistoryPolicy(),
            CharacteristicPolicy(amplitude_error=PHYSICS_ERROR),
            OraclePolicy(),
        ]
        case = pd.DataFrame([run_policy(p, lam) for p in policies])

        static_cost = float(case.loc[case.policy=="static","total_weighted_cost"].iloc[0])
        reactive_cost = float(case.loc[case.policy=="reactive","total_weighted_cost"].iloc[0])
        history_cost = float(case.loc[case.policy=="history","total_weighted_cost"].iloc[0])
        oracle_cost = float(case.loc[case.policy=="oracle","total_weighted_cost"].iloc[0])

        case["improvement_vs_static"] = static_cost / case["total_weighted_cost"]
        case["improvement_vs_reactive"] = reactive_cost / case["total_weighted_cost"]
        case["improvement_vs_history"] = history_cost / case["total_weighted_cost"]
        denom = reactive_cost - oracle_cost
        case["oracle_capture_vs_reactive"] = (
            (reactive_cost - case["total_weighted_cost"]) / denom
            if abs(denom) > 1e-12 else float("nan")
        )
        all_cases.append(case)

    out = pd.concat(all_cases, ignore_index=True)
    result_dir = ROOT/"results"; plot_dir = ROOT/"plots"
    result_dir.mkdir(exist_ok=True); plot_dir.mkdir(exist_ok=True)

    out.to_csv(result_dir/"remapping_cost_sweep_fixed.csv", index=False)

    for policy in ["static","reactive","history","characteristic","oracle"]:
        d = out[out.policy==policy]
        plt.plot(d.migration_lambda, d.total_weighted_cost, marker="o", label=policy)
    plt.xlabel("Migration penalty lambda")
    plt.ylabel("Communication + lambda * migration cost")
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_dir/"remapping_weighted_cost_fixed.png", dpi=180); plt.close()

    char = out[out.policy=="characteristic"]
    plt.plot(char.migration_lambda, char.improvement_vs_reactive, marker="o", label="vs reactive")
    plt.plot(char.migration_lambda, char.improvement_vs_history, marker="o", label="vs history")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Migration penalty lambda")
    plt.ylabel("Baseline cost / Char-Map cost")
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_dir/"remapping_charmap_advantage_fixed.png", dpi=180); plt.close()

    print("\n=== FIXED REMAPPING COST SWEEP ===")
    print(out[[
        "migration_lambda","policy","total_comm_cost","total_migration_cost",
        "total_weighted_cost","improvement_vs_reactive","improvement_vs_history",
        "oracle_capture_vs_reactive","total_moved_task_events"
    ]].sort_values(["migration_lambda","policy"]).to_string(index=False))

    print("\n=== CHARMAP ONLY ===")
    print(char[[
        "migration_lambda","total_comm_cost","total_migration_cost",
        "total_weighted_cost","improvement_vs_reactive","improvement_vs_history",
        "oracle_capture_vs_reactive","total_moved_task_events"
    ]].sort_values("migration_lambda").to_string(index=False))

if __name__ == "__main__":
    main()
