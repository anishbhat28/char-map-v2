from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

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


def main():
    workload = RegimeSwitchingAdvection(
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

    cfg = MappingSimConfig(
        rows=8,
        cols=8,
        horizon=2,
        optimizer_passes=4,
        remap_every=1,
    )

    # IMPORTANT:
    # First correctness run uses exact physics.
    policies = [
        StaticPolicy(),
        ReactivePolicy(),
        HistoryPolicy(),
        CharacteristicPolicy(amplitude_error=0.0),
        OraclePolicy(),
    ]

    frames = []

    for policy in policies:
        sim = MappingSimulator(
            workload=workload,
            policy=policy,
            cfg=cfg,
        )
        frames.append(sim.run())

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    result_dir = ROOT / "results"
    plot_dir = ROOT / "plots"
    result_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    out.to_csv(
        result_dir / "regime_switch_timestep.csv",
        index=False,
    )

    # Exclude warm-start timesteps before any H-step-ahead placement can
    # possibly become active. This makes policy comparisons cleaner.
    eval_out = out[
        out["timestep"] >= cfg.horizon
    ].copy()

    summary = (
        eval_out.groupby("policy", as_index=False)
        .agg(
            mean_cost=("actual_cost", "mean"),
            median_cost=("actual_cost", "median"),
            p95_cost=("actual_cost", lambda x: x.quantile(0.95)),
            total_cost=("actual_cost", "sum"),
        )
    )

    static_total = float(
        summary.loc[
            summary.policy == "static",
            "total_cost",
        ].iloc[0]
    )

    summary["improvement_vs_static"] = (
        static_total / summary["total_cost"]
    )

    summary.to_csv(
        result_dir / "regime_switch_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Correctness diagnostics
    # ------------------------------------------------------------
    char_total = float(
        summary.loc[
            summary.policy == "characteristic",
            "total_cost",
        ].iloc[0]
    )

    oracle_total = float(
        summary.loc[
            summary.policy == "oracle",
            "total_cost",
        ].iloc[0]
    )

    # With exact physics, characteristic and oracle should be identical
    # (or effectively identical) because they receive the same future graph.
    char_oracle_gap = abs(char_total - oracle_total)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== CORRECTNESS DIAGNOSTIC ===")
    print(f"Characteristic total cost: {char_total:.6f}")
    print(f"Oracle total cost:         {oracle_total:.6f}")
    print(f"Absolute gap:              {char_oracle_gap:.6f}")

    if char_oracle_gap > 1e-6:
        raise AssertionError(
            "Exact characteristic and oracle diverged. "
            "Do not interpret the mapping experiment yet."
        )

    # Oracle should not be materially worse than other predictive policies
    # when all use the same optimizer and lead time.
    adaptive = summary[
        summary.policy.isin(
            ["reactive", "history", "characteristic", "oracle"]
        )
    ]

    best_nonoracle = float(
        adaptive.loc[
            adaptive.policy != "oracle",
            "total_cost",
        ].min()
    )

    if oracle_total > best_nonoracle + 1e-6:
        print(
            "\nWARNING: Oracle is worse than another policy. "
            "This can happen because the placement optimizer is heuristic/local, "
            "even with a better graph. If the gap is large, increase optimizer "
            "quality before interpreting results."
        )

    # ------------------------------------------------------------
    # Plot actual realized communication cost
    # ------------------------------------------------------------
    for policy in out["policy"].unique():
        d = out[out["policy"] == policy]
        plt.plot(
            d["timestep"],
            d["actual_cost"],
            label=policy,
        )

    s1 = workload.timesteps // 3
    s2 = 2 * workload.timesteps // 3

    plt.axvline(s1, linestyle="--")
    plt.axvline(s2, linestyle="--")

    plt.xlabel("Timestep")
    plt.ylabel("Weighted Manhattan communication cost")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plot_dir / "regime_switch_cost.png",
        dpi=180,
    )
    plt.close()


if __name__ == "__main__":
    main()
