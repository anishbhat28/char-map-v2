from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from workloads.advection import RegimeSwitchingAdvection
from mapping.policies import (
    ReactivePolicy,
    HistoryPolicy,
    WrongPhysicsPolicy,
    OraclePolicy,
)
from simulation.simulator import (
    MappingSimulator,
    MappingSimConfig,
)


# ----------------------------------------------------------------------
# Keep this suite relatively cheap.
# Once a failure mode looks important, sweep it more densely later.
# ----------------------------------------------------------------------

HORIZON = 4

RELATIVE_ERRORS = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.20,
]

STATE_DELAYS = [
    0,
    1,
    2,
    4,
    8,
]

REGIME_SCHEDULES = {
    "true_schedule": (
        (0.25, 0.75, 0.40),
        (1.0/3.0, 2.0/3.0),
    ),

    # Underestimates the strong middle regime and overestimates the final one.
    "wrong_amplitudes": (
        (0.25, 0.50, 0.60),
        (1.0/3.0, 2.0/3.0),
    ),

    # Correct amplitudes, but believes both switches occur later.
    "late_switches": (
        (0.25, 0.75, 0.40),
        (0.43, 0.76),
    ),

    # Correct amplitudes, but believes switches occur earlier.
    "early_switches": (
        (0.25, 0.75, 0.40),
        (0.23, 0.56),
    ),
}


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


def run_policy(policy):
    workload = make_workload()

    cfg = MappingSimConfig(
        rows=8,
        cols=8,
        horizon=HORIZON,
        optimizer_passes=4,
        remap_every=1,
    )

    df = MappingSimulator(
        workload=workload,
        policy=policy,
        cfg=cfg,
    ).run()

    # Remove startup steps before an H-ahead placement can be active.
    df = df[
        df["timestep"] >= HORIZON
    ].copy()

    return {
        "policy": policy.name,
        "total_cost": float(df["actual_cost"].sum()),
        "mean_cost": float(df["actual_cost"].mean()),
        "p95_cost": float(df["actual_cost"].quantile(0.95)),
    }


def main():
    rows = []

    reactive = run_policy(
        ReactivePolicy()
    )
    history = run_policy(
        HistoryPolicy()
    )
    oracle = run_policy(
        OraclePolicy()
    )

    reactive_cost = reactive["total_cost"]
    history_cost = history["total_cost"]
    oracle_cost = oracle["total_cost"]

    # --------------------------------------------------------------
    # Reference rows
    # --------------------------------------------------------------
    for r in [reactive, history, oracle]:
        r = dict(r)
        r.update(
            {
                "family": "baseline",
                "severity": 0.0,
                "label": r["policy"],
            }
        )
        rows.append(r)

    # --------------------------------------------------------------
    # 1. Wrong mean propagation speed c0
    # --------------------------------------------------------------
    for eps in RELATIVE_ERRORS:
        policy = WrongPhysicsPolicy(
            name=f"wrong_c0_{eps:g}",
            c0_scale=1.0 + eps,
        )
        r = run_policy(policy)
        r.update(
            family="wrong_c0",
            severity=eps,
            label=f"{100*eps:.0f}% c0 error",
        )
        rows.append(r)

    # --------------------------------------------------------------
    # 2. Wrong temporal oscillation frequency omega
    # --------------------------------------------------------------
    for eps in RELATIVE_ERRORS:
        policy = WrongPhysicsPolicy(
            name=f"wrong_omega_{eps:g}",
            omega_scale=1.0 + eps,
        )
        r = run_policy(policy)
        r.update(
            family="wrong_omega",
            severity=eps,
            label=f"{100*eps:.0f}% omega error",
        )
        rows.append(r)

    # --------------------------------------------------------------
    # 3. Delayed regime/state signal
    # --------------------------------------------------------------
    for delay in STATE_DELAYS:
        policy = WrongPhysicsPolicy(
            name=f"state_delay_{delay}",
            state_delay_steps=delay,
        )
        r = run_policy(policy)
        r.update(
            family="state_delay",
            severity=float(delay),
            label=f"{delay} step delay",
        )
        rows.append(r)

    # --------------------------------------------------------------
    # 4. Structural mismatch: predictor ignores spatial variation
    # --------------------------------------------------------------
    policy = WrongPhysicsPolicy(
        name="no_spatial_variation",
        include_spatial_variation=False,
    )
    r = run_policy(policy)
    r.update(
        family="structural",
        severity=1.0,
        label="constant-velocity model",
    )
    rows.append(r)

    # --------------------------------------------------------------
    # 5. Wrong regime schedule
    # --------------------------------------------------------------
    for schedule_name, (
        amplitudes,
        switch_fractions,
    ) in REGIME_SCHEDULES.items():

        policy = WrongPhysicsPolicy(
            name=schedule_name,
            model_amplitudes=amplitudes,
            model_switch_fractions=switch_fractions,
        )

        r = run_policy(policy)

        r.update(
            family="wrong_regime",
            severity=0.0,
            label=schedule_name,
        )

        rows.append(r)

    out = pd.DataFrame(rows)

    out["improvement_vs_reactive"] = (
        reactive_cost / out["total_cost"]
    )

    out["improvement_vs_history"] = (
        history_cost / out["total_cost"]
    )

    denom = reactive_cost - oracle_cost

    out["oracle_capture_vs_reactive"] = (
        (reactive_cost - out["total_cost"])
        / denom
        if abs(denom) > 1e-12
        else float("nan")
    )

    result_dir = ROOT / "results"
    plot_dir = ROOT / "plots"

    result_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    out.to_csv(
        result_dir / "wrong_physics_suite.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Plot parametric error families
    # --------------------------------------------------------------
    for family in [
        "wrong_c0",
        "wrong_omega",
    ]:
        d = out[
            out["family"] == family
        ].sort_values("severity")

        plt.plot(
            100.0 * d["severity"],
            d["improvement_vs_reactive"],
            marker="o",
            label=family,
        )

    plt.axhline(
        1.0,
        linestyle="--",
    )

    plt.xlabel("Relative model error (%)")
    plt.ylabel("Reactive cost / Char-Map cost")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plot_dir / "wrong_parameter_physics.png",
        dpi=180,
    )
    plt.close()

    # --------------------------------------------------------------
    # Plot stale-state behavior
    # --------------------------------------------------------------
    d = out[
        out["family"] == "state_delay"
    ].sort_values("severity")

    plt.plot(
        d["severity"],
        d["improvement_vs_reactive"],
        marker="o",
    )

    plt.axhline(
        1.0,
        linestyle="--",
    )

    plt.xlabel("Physical-state delay (timesteps)")
    plt.ylabel("Reactive cost / Char-Map cost")
    plt.tight_layout()

    plt.savefig(
        plot_dir / "wrong_physics_state_delay.png",
        dpi=180,
    )
    plt.close()

    # --------------------------------------------------------------
    # Bar plot for qualitative structural/model mismatch
    # --------------------------------------------------------------
    qualitative = out[
        out["family"].isin(
            [
                "structural",
                "wrong_regime",
            ]
        )
    ].copy()

    plt.bar(
        range(len(qualitative)),
        qualitative["improvement_vs_reactive"],
    )

    plt.xticks(
        range(len(qualitative)),
        qualitative["label"],
        rotation=25,
        ha="right",
    )

    plt.axhline(
        1.0,
        linestyle="--",
    )

    plt.ylabel("Reactive cost / Char-Map cost")
    plt.tight_layout()

    plt.savefig(
        plot_dir / "wrong_physics_structural.png",
        dpi=180,
    )
    plt.close()

    # --------------------------------------------------------------
    # Print
    # --------------------------------------------------------------
    print(
        "\n=== BASELINES ==="
    )
    print(
        out[
            out["family"] == "baseline"
        ][
            [
                "label",
                "total_cost",
                "improvement_vs_reactive",
                "oracle_capture_vs_reactive",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== WRONG c0 ==="
    )
    print(
        out[
            out["family"] == "wrong_c0"
        ][
            [
                "label",
                "total_cost",
                "improvement_vs_reactive",
                "oracle_capture_vs_reactive",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== WRONG omega ==="
    )
    print(
        out[
            out["family"] == "wrong_omega"
        ][
            [
                "label",
                "total_cost",
                "improvement_vs_reactive",
                "oracle_capture_vs_reactive",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== DELAYED PHYSICAL STATE ==="
    )
    print(
        out[
            out["family"] == "state_delay"
        ][
            [
                "label",
                "total_cost",
                "improvement_vs_reactive",
                "oracle_capture_vs_reactive",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== STRUCTURAL / REGIME MISMATCH ==="
    )
    print(
        out[
            out["family"].isin(
                [
                    "structural",
                    "wrong_regime",
                ]
            )
        ][
            [
                "family",
                "label",
                "total_cost",
                "improvement_vs_reactive",
                "improvement_vs_history",
                "oracle_capture_vs_reactive",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
