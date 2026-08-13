from __future__ import annotations

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
import torch

from learned.burgers2d_data import (
    make_randomized_workload,
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
from mapping.learned_burgers2d_policy import (
    LearnedBurgers2DPolicy,
)
from simulation.noc_simulator import (
    NoCMappingSimulator,
    NoCMappingSimConfig,
)


HORIZON = 4

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "burgers2d_surrogate.pt"
)

# Held-out seed: deliberately outside train/validation ranges.
EVAL_SEED = 50000


def workload():
    return make_randomized_workload(
        seed=EVAL_SEED,
        nx=8,
        ny=8,
        timesteps=100,
        dt=0.004,
        viscosity=0.002,
    )


def run(
    policy,
):
    cfg = NoCMappingSimConfig(
        rows=8,
        cols=8,

        horizon=HORIZON,

        # Keep the same strong mapper used in the final PDE experiments.
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
        workload(),
        policy,
        cfg,
    ).run()

    return df[
        df[
            "timestep"
        ] >= HORIZON
    ].copy()


def summarize(
    out,
):
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

    perfect_prediction = summary[
        summary[
            "policy"
        ] == "perfect_future_graph"
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
        - perfect_prediction[
            "total_objective"
        ]
    )

    summary[
        "perfect_prediction_capture"
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

    return summary


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT}\n"
            "Train first:\n"
            "python learned/train_burgers2d_surrogate.py"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Learned policy inference device: "
        f"{device}"
    )

    w = workload()

    late_nonlocal = (
        w.nonlocal_edge_fraction(
            w.timesteps
        )
    )

    print(
        f"Held-out workload "
        f"late nonlocal-edge fraction: "
        f"{late_nonlocal:.4f}"
    )

    policies = [
        StaticPolicy(),
        ReactivePolicy(),
        HistoryPolicy(),

        # Exact PDE-based prediction, retained only as a reference.
        Burgers2DCharacteristicPolicy(
            name="exact_physics",
        ),

        # Learned future dynamics.
        LearnedBurgers2DPolicy(
            CHECKPOINT,
            device=device,
            name="learned_dynamics",
        ),

        # Rename the old Oracle more precisely.
        Burgers2DOraclePolicy(),
    ]

    # Rename policy for paper terminology:
    policies[-1].name = (
        "perfect_future_graph"
    )

    frames = []

    for p in policies:
        print(
            f"Running {p.name} ...",
            flush=True,
        )

        frames.append(
            run(
                p
            )
        )

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    result_dir = (
        ROOT
        / "results"
    )

    result_dir.mkdir(
        exist_ok=True
    )

    out.to_csv(
        result_dir
        / "learned_burgers2d_noc_timestep.csv",
        index=False,
    )

    summary = summarize(
        out
    )

    summary.to_csv(
        result_dir
        / "learned_burgers2d_noc_summary.csv",
        index=False,
    )

    # Exact PDE physics and perfect future graph should still match.
    exact = summary[
        summary[
            "policy"
        ] == "exact_physics"
    ].iloc[0]

    perfect = summary[
        summary[
            "policy"
        ] == "perfect_future_graph"
    ].iloc[0]

    invariant_cols = [
        "total_byte_hops",
        "mean_max_link_load",
        "p95_max_link_load",
        "mean_estimated_latency",
        "p95_estimated_latency",
        "total_migration_byte_hops",
        "total_objective",
        "total_moved_tasks",
    ]

    for c in invariant_cols:
        if abs(
            float(
                exact[c]
            )
            - float(
                perfect[c]
            )
        ) > 1e-9:
            raise AssertionError(
                "Exact PDE physics != "
                f"perfect future graph for {c}"
            )

    print(
        "\n=== LEARNED-DYNAMICS CHAR-MAP ==="
    )

    print(
        summary.sort_values(
            "total_objective"
        ).to_string(
            index=False
        )
    )

    print(
        "\nExact physics matches "
        "perfect-future-graph reference."
    )


if __name__ == "__main__":
    main()
