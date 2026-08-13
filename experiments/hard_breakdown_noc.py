from pathlib import Path
import sys
import argparse
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from learned.ood_hard_breakdown import (
    SCENARIOS,
    make_hard_workload,
)
from mapping.policies import (
    ReactivePolicy,
    HistoryPolicy,
)
from mapping.burgers2d_policies import (
    Burgers2DCharacteristicPolicy,
    Burgers2DOraclePolicy,
)
from mapping.learned_burgers2d_policy import (
    LearnedBurgers2DPolicy,
)
from simulation.noc_simulator import (
    NoCMappingSimulator,
    NoCMappingSimConfig,
)

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "burgers2d_surrogate.pt"
)

SEED = 50000


def make_workload(
    scenario_name,
):
    return make_hard_workload(
        seed=SEED,
        scenario=SCENARIOS[
            scenario_name
        ],
        nx=8,
        ny=8,
        timesteps=100,
        dt=0.004,
        base_viscosity=0.002,
    )


def run_policy(
    policy,
    scenario_name,
    horizon,
):
    cfg = NoCMappingSimConfig(
        rows=8,
        cols=8,

        horizon=horizon,

        optimizer_passes=2,
        remap_every=1,

        bytes_per_unit=256.0,
        link_bandwidth_bytes_per_cycle=64.0,
        router_latency_cycles=1.0,

        alpha_byte_hops=1.0,
        beta_max_link_load=4.0,

        migration_lambda=0.5,
        task_state_size=256.0,

        num_random_starts=2,
        multistart_seed=1337,
    )

    df = NoCMappingSimulator(
        make_workload(
            scenario_name
        ),
        policy,
        cfg,
    ).run()

    return df[
        df[
            "timestep"
        ] >= horizon
    ].copy()


def summarize(
    out,
):
    s = (
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

    reactive = s[
        s[
            "policy"
        ] == "reactive"
    ].iloc[0]

    perfect = s[
        s[
            "policy"
        ] == "perfect_future_graph"
    ].iloc[0]

    s[
        "objective_improvement_vs_reactive"
    ] = (
        reactive[
            "total_objective"
        ]
        / s[
            "total_objective"
        ]
    )

    s[
        "byte_hop_improvement_vs_reactive"
    ] = (
        reactive[
            "total_byte_hops"
        ]
        / s[
            "total_byte_hops"
        ]
    )

    s[
        "latency_improvement_vs_reactive"
    ] = (
        reactive[
            "mean_estimated_latency"
        ]
        / s[
            "mean_estimated_latency"
        ]
    )

    denom = (
        reactive[
            "total_objective"
        ]
        - perfect[
            "total_objective"
        ]
    )

    s[
        "perfect_prediction_capture"
    ] = (
        (
            reactive[
                "total_objective"
            ]
            - s[
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

    return s


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        choices=list(
            SCENARIOS.keys()
        ),
        required=True,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        choices=[
            16,
            24,
            32,
        ],
        required=True,
    )

    args = parser.parse_args()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Scenario={args.scenario}, "
        f"H={args.horizon}, "
        f"device={device}"
    )

    policies = [
        ReactivePolicy(),
        HistoryPolicy(),

        Burgers2DCharacteristicPolicy(
            name="exact_physics",
        ),

        LearnedBurgers2DPolicy(
            CHECKPOINT,
            device=device,
            name="learned_dynamics",
        ),

        Burgers2DOraclePolicy(),
    ]

    policies[-1].name = (
        "perfect_future_graph"
    )

    frames = []

    for i, p in enumerate(
        policies,
        start=1,
    ):
        start = time.perf_counter()

        print(
            f"[{i}/{len(policies)}] "
            f"Running {p.name} ...",
            flush=True,
        )

        frames.append(
            run_policy(
                p,
                args.scenario,
                args.horizon,
            )
        )

        print(
            f"    finished in "
            f"{time.perf_counter()-start:.1f} s",
            flush=True,
        )

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    summary = summarize(
        out
    )

    summary.insert(
        0,
        "scenario",
        args.scenario,
    )

    summary.insert(
        1,
        "horizon",
        args.horizon,
    )

    result_dir = (
        ROOT
        / "results"
    )

    result_dir.mkdir(
        exist_ok=True
    )

    out_path = (
        result_dir
        / (
            f"hard_breakdown_noc_"
            f"{args.scenario}_"
            f"h{args.horizon}.csv"
        )
    )

    summary.to_csv(
        out_path,
        index=False,
    )

    print(
        "\n=== HARD BREAKDOWN NoC ==="
    )

    print(
        summary.sort_values(
            "total_objective"
        ).to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {out_path}"
    )


if __name__ == "__main__":
    main()
