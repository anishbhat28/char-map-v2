from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from workloads.advection import RegimeSwitchingAdvection
from mapping.policies import (
    StaticPolicy,
    ReactivePolicy,
    HistoryPolicy,
    CharacteristicPolicy,
    OraclePolicy,
)
from simulation.simulator import (
    MappingSimulator,
    MappingSimConfig,
)

ERRORS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]
HORIZONS = [1, 2, 4, 8, 16]


def make_workload():
    return RegimeSwitchingAdvection(
        num_cells=64,
        timesteps=100,
        dt=1.0 / 64.0,
        c0=1.0,
        omega=4.0,
        amplitude_1=0.25,
        amplitude_2=0.75,
        amplitude_3=0.40,
        rk4_substeps_per_dt=8,
    )


def run_policy(workload, policy, horizon):
    cfg = MappingSimConfig(
        rows=8,
        cols=8,
        horizon=horizon,
        optimizer_passes=4,
        remap_every=1,
    )

    df = MappingSimulator(
        workload=workload,
        policy=policy,
        cfg=cfg,
    ).run()

    # Ignore the startup window before an H-step-ahead decision
    # could have become active.
    df = df[df["timestep"] >= horizon].copy()

    return {
        "policy": policy.name,
        "horizon": horizon,
        "mean_cost": float(df["actual_cost"].mean()),
        "p95_cost": float(df["actual_cost"].quantile(0.95)),
        "total_cost": float(df["actual_cost"].sum()),
    }


def main():
    rows = []

    for horizon in HORIZONS:
        workload = make_workload()

        # Shared baselines for this horizon.
        baseline_rows = [
            run_policy(workload, StaticPolicy(), horizon),
            run_policy(workload, ReactivePolicy(), horizon),
            run_policy(workload, HistoryPolicy(), horizon),
            run_policy(workload, OraclePolicy(), horizon),
        ]

        base = pd.DataFrame(baseline_rows)

        static_cost = float(
            base.loc[base["policy"] == "static", "total_cost"].iloc[0]
        )
        reactive_cost = float(
            base.loc[base["policy"] == "reactive", "total_cost"].iloc[0]
        )
        history_cost = float(
            base.loc[base["policy"] == "history", "total_cost"].iloc[0]
        )
        oracle_cost = float(
            base.loc[base["policy"] == "oracle", "total_cost"].iloc[0]
        )

        for r in baseline_rows:
            r["physics_error"] = np.nan
            r["improvement_vs_static"] = static_cost / r["total_cost"]
            r["improvement_vs_reactive"] = reactive_cost / r["total_cost"]
            r["improvement_vs_history"] = history_cost / r["total_cost"]
            denom = reactive_cost - oracle_cost
            r["oracle_capture_vs_reactive"] = (
                (reactive_cost - r["total_cost"]) / denom
                if abs(denom) > 1e-12 else np.nan
            )
            rows.append(r)

        # Physics sweep.
        for eps in ERRORS:
            result = run_policy(
                workload,
                CharacteristicPolicy(amplitude_error=eps),
                horizon,
            )
            result["physics_error"] = eps
            result["improvement_vs_static"] = static_cost / result["total_cost"]
            result["improvement_vs_reactive"] = reactive_cost / result["total_cost"]
            result["improvement_vs_history"] = history_cost / result["total_cost"]

            denom = reactive_cost - oracle_cost
            result["oracle_capture_vs_reactive"] = (
                (reactive_cost - result["total_cost"]) / denom
                if abs(denom) > 1e-12 else np.nan
            )
            rows.append(result)

    out = pd.DataFrame(rows)

    result_dir = ROOT / "results"
    plot_dir = ROOT / "plots"
    result_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    out.to_csv(
        result_dir / "error_horizon_sweep.csv",
        index=False,
    )

    # Compact physics-only table.
    phys = out[out["policy"] == "characteristic"].copy()
    phys.to_csv(
        result_dir / "error_horizon_physics_only.csv",
        index=False,
    )

    # Heatmap 1: characteristic improvement over reactive
    pivot = phys.pivot(
        index="physics_error",
        columns="horizon",
        values="improvement_vs_reactive",
    )

    plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
    )
    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
    )
    plt.yticks(
        range(len(pivot.index)),
        [f"{100*x:.0f}%" for x in pivot.index],
    )
    plt.xlabel("Prediction horizon H")
    plt.ylabel("Physics amplitude error")
    plt.colorbar(label="Reactive cost / Char-Map cost")
    plt.tight_layout()
    plt.savefig(
        plot_dir / "error_horizon_vs_reactive.png",
        dpi=180,
    )
    plt.close()

    # Heatmap 2: oracle capture
    pivot2 = phys.pivot(
        index="physics_error",
        columns="horizon",
        values="oracle_capture_vs_reactive",
    )

    plt.imshow(
        pivot2.values,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    plt.xticks(
        range(len(pivot2.columns)),
        pivot2.columns,
    )
    plt.yticks(
        range(len(pivot2.index)),
        [f"{100*x:.0f}%" for x in pivot2.index],
    )
    plt.xlabel("Prediction horizon H")
    plt.ylabel("Physics amplitude error")
    plt.colorbar(label="Oracle capture vs reactive")
    plt.tight_layout()
    plt.savefig(
        plot_dir / "error_horizon_oracle_capture.png",
        dpi=180,
    )
    plt.close()

    # Line plot: cost vs horizon for each physics error
    for eps in ERRORS:
        d = phys[phys["physics_error"] == eps]
        plt.plot(
            d["horizon"],
            d["total_cost"],
            marker="o",
            label=f"{100*eps:.0f}% error",
        )

    # Add reactive/oracle horizon curves for context
    reactive = out[out["policy"] == "reactive"].drop_duplicates("horizon")
    oracle = out[out["policy"] == "oracle"].drop_duplicates("horizon")

    plt.plot(
        reactive["horizon"],
        reactive["total_cost"],
        marker="o",
        label="reactive",
    )
    plt.plot(
        oracle["horizon"],
        oracle["total_cost"],
        marker="o",
        label="oracle",
    )

    plt.xlabel("Prediction horizon H")
    plt.ylabel("Total weighted Manhattan cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_dir / "error_horizon_cost_curves.png",
        dpi=180,
    )
    plt.close()

    print("\n=== PHYSICS ERROR x HORIZON ===")
    print(
        phys[
            [
                "physics_error",
                "horizon",
                "total_cost",
                "improvement_vs_reactive",
                "improvement_vs_history",
                "oracle_capture_vs_reactive",
            ]
        ]
        .sort_values(["physics_error", "horizon"])
        .to_string(index=False)
    )

    print("\n=== BASELINES BY HORIZON ===")
    print(
        out[
            out["policy"].isin(["static", "reactive", "history", "oracle"])
        ][
            [
                "policy",
                "horizon",
                "total_cost",
                "improvement_vs_static",
            ]
        ]
        .drop_duplicates(["policy", "horizon"])
        .sort_values(["horizon", "policy"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
