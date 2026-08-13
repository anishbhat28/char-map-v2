from pathlib import Path
import sys

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

import pandas as pd

from workloads.burgers2d import (
    Burgers2DWorkload,
)

from mapping.policies import (
    StaticPolicy,
    ReactivePolicy,
    HistoryPolicy,
)

from mapping.burgers2d_policies import (
    Burgers2DCharacteristicPolicy,
    Burgers2DOraclePolicy,
)

from simulation.noc_simulator import (
    NoCMappingSimulator,
    NoCMappingSimConfig,
)


HORIZON = 4


def workload():
    return Burgers2DWorkload(
        nx=8,
        ny=8,
        timesteps=100,
        dt=0.004,
        viscosity=0.002,
        advection_scale=1.0,
    )


def run(
    policy,
):
    cfg = NoCMappingSimConfig(
        rows=8,
        cols=8,

        horizon=HORIZON,

        optimizer_passes=2,
        remap_every=1,

        bytes_per_unit=256.0,
        link_bandwidth_bytes_per_cycle=64.0,
        router_latency_cycles=1.0,

        alpha_byte_hops=1.0,
        beta_max_link_load=4.0,

        migration_lambda=0.5,
        task_state_size=256.0,

        # Keep this tractable on laptop.
        num_random_starts=2,
        multistart_seed=1337,
    )

    df = NoCMappingSimulator(
        workload(),
        policy,
        cfg,
    ).run()

    return df[
        df[
            "timestep"
        ] >= HORIZON
    ].copy()


def main():
    # Fail fast before expensive mapping.
    w = workload()

    late_nonlocal = (
        w.nonlocal_edge_fraction(
            w.timesteps
        )
    )

    print(
        f"Late nonlocal-edge fraction: "
        f"{late_nonlocal:.4f}"
    )

    if late_nonlocal < 0.25:
        raise AssertionError(
            "2-D Burgers workload is still too local. "
            "Run diagnostic and rescale before mapping."
        )

    policies = [
        StaticPolicy(),
        ReactivePolicy(),
        HistoryPolicy(),

        Burgers2DCharacteristicPolicy(
            name=(
                "burgers2d_physics_exact"
            ),
        ),

        Burgers2DCharacteristicPolicy(
            name=(
                "burgers2d_physics_imperfect"
            ),
            advection_error=0.03,
            viscosity_error=0.10,
            u_state_error=0.02,
            v_state_error=-0.02,
        ),

        Burgers2DCharacteristicPolicy(
            name=(
                "burgers2d_no_crossflow"
            ),
            remove_crossflow=True,
        ),

        Burgers2DOraclePolicy(),
    ]

    frames = []

    for p in policies:
        print(
            f"Running {p.name} ...",
            flush=True,
        )

        frames.append(
            run(p)
        )

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    result_dir = (
        ROOT / "results"
    )

    result_dir.mkdir(
        exist_ok=True
    )

    out.to_csv(
        result_dir
        / "burgers2d_noc_timestep_fixed.csv",
        index=False,
    )

    summary = (
        out.groupby(
            "policy",
            as_index=False,
        )
        .agg(
            total_byte_hops=(
                "byte_hops",
                "sum",
            ),

            mean_max_link_load=(
                "max_link_load_bytes",
                "mean",
            ),

            p95_max_link_load=(
                "max_link_load_bytes",
                lambda x: x.quantile(
                    0.95
                ),
            ),

            mean_estimated_latency=(
                "estimated_latency_cycles",
                "mean",
            ),

            p95_estimated_latency=(
                "estimated_latency_cycles",
                lambda x: x.quantile(
                    0.95
                ),
            ),

            total_migration_byte_hops=(
                "migration_byte_hops",
                "sum",
            ),

            total_objective=(
                "realized_objective",
                "sum",
            ),

            total_moved_tasks=(
                "moved_tasks",
                "sum",
            ),
        )
    )

    reactive = summary[
        summary[
            "policy"
        ] == "reactive"
    ].iloc[0]

    oracle = summary[
        summary[
            "policy"
        ] == "burgers2d_oracle"
    ].iloc[0]

    summary[
        "objective_improvement_vs_reactive"
    ] = (
        reactive[
            "total_objective"
        ]
        / summary[
            "total_objective"
        ]
    )

    summary[
        "byte_hop_improvement_vs_reactive"
    ] = (
        reactive[
            "total_byte_hops"
        ]
        / summary[
            "total_byte_hops"
        ]
    )

    summary[
        "latency_improvement_vs_reactive"
    ] = (
        reactive[
            "mean_estimated_latency"
        ]
        / summary[
            "mean_estimated_latency"
        ]
    )

    denom = (
        reactive[
            "total_objective"
        ]
        - oracle[
            "total_objective"
        ]
    )

    summary[
        "oracle_capture_vs_reactive"
    ] = (
        (
            reactive[
                "total_objective"
            ]
            - summary[
                "total_objective"
            ]
        )
        / denom
        if abs(
            denom
        ) > 1e-12
        else float(
            "nan"
        )
    )

    exact = summary[
        summary[
            "policy"
        ] == "burgers2d_physics_exact"
    ].iloc[0]

    cols = [
        "total_byte_hops",
        "mean_max_link_load",
        "p95_max_link_load",
        "mean_estimated_latency",
        "p95_estimated_latency",
        "total_migration_byte_hops",
        "total_objective",
        "total_moved_tasks",
    ]

    for c in cols:
        if abs(
            float(
                exact[c]
            )
            - float(
                oracle[c]
            )
        ) > 1e-9:
            raise AssertionError(
                "Exact 2-D Burgers physics "
                f"!= Oracle for {c}"
            )

    summary.to_csv(
        result_dir
        / "burgers2d_noc_summary_fixed.csv",
        index=False,
    )

    print(
        "\n=== FIXED 2-D BURGERS NoC ==="
    )

    print(
        summary.sort_values(
            "total_objective"
        ).to_string(
            index=False
        )
    )

    print(
        "\nExact 2-D Burgers physics matches Oracle."
    )


if __name__ == "__main__":
    main()
