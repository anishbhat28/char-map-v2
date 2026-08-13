from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from learned.ood_hard_breakdown import (
    SCENARIOS,
    make_hard_workload,
)
from mapping.policies import (
    ReactivePolicy,
)
from mapping.burgers2d_policies import (
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

RESULT_DIR = (
    ROOT
    / "results"
)

RAW_PATH = (
    RESULT_DIR
    / "multiseed_validation_raw.csv"
)

SUMMARY_PATH = (
    RESULT_DIR
    / "trimmed_multiseed_validation_summary.csv"
)


# Only the useful remaining validation points.
POINTS = [
    ("moderate", "hard2", 24, 0.107422),
    ("high", "hard4", 32, 0.209635),
]

SEEDS = [
    50000,
    50001,
    50002,
]


def make_workload(
    scenario_name,
    seed,
):
    return make_hard_workload(
        seed=seed,
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
    *,
    scenario_name,
    horizon,
    seed,
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
            scenario_name,
            seed,
        ),
        policy,
        cfg,
    ).run()

    return df[
        df[
            "timestep"
        ] >= horizon
    ].copy()


def summarize_policy(
    df,
):
    return {
        "total_objective":
            float(
                df[
                    "realized_objective"
                ].sum()
            ),

        "total_byte_hops":
            float(
                df[
                    "byte_hops"
                ].sum()
            ),

        "mean_estimated_latency":
            float(
                df[
                    "estimated_latency_cycles"
                ].mean()
            ),

        "total_moved_tasks":
            int(
                df[
                    "moved_tasks"
                ].sum()
            ),
    }


def case_key(
    row,
):
    return (
        str(
            row[
                "regime"
            ]
        ),
        str(
            row[
                "scenario"
            ]
        ),
        int(
            row[
                "horizon"
            ]
        ),
        int(
            row[
                "seed"
            ]
        ),
    )


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT}"
        )

    RESULT_DIR.mkdir(
        exist_ok=True
    )

    if RAW_PATH.exists():
        existing = pd.read_csv(
            RAW_PATH
        )

        rows = existing.to_dict(
            orient="records"
        )

        completed = {
            case_key(
                row
            )
            for row in rows
            if all(
                k in row
                for k in [
                    "regime",
                    "scenario",
                    "horizon",
                    "seed",
                ]
            )
        }

        print(
            f"Loaded existing CSV with "
            f"{len(rows)} completed rows."
        )

    else:
        rows = []
        completed = set()

        print(
            "No existing CSV found; "
            "starting from scratch."
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    targets = [
        (
            regime,
            scenario,
            horizon,
            seed,
            mismatch,
        )
        for (
            regime,
            scenario,
            horizon,
            mismatch,
        ) in POINTS
        for seed in SEEDS
    ]

    missing = [
        x
        for x in targets
        if (
            x[0],
            x[1],
            x[2],
            x[3],
        )
        not in completed
    ]

    print(
        f"Target cases: {len(targets)}"
    )

    print(
        f"Already complete: "
        f"{len(targets)-len(missing)}"
    )

    print(
        f"Remaining: {len(missing)}"
    )

    for idx, (
        regime,
        scenario,
        horizon,
        seed,
        mismatch,
    ) in enumerate(
        missing,
        start=1,
    ):
        print(
            f"\n=== REMAINING CASE "
            f"{idx}/{len(missing)} ==="
        )

        print(
            f"{regime} | "
            f"{scenario} | "
            f"H={horizon} | "
            f"seed={seed}",
            flush=True,
        )

        policies = [
            ReactivePolicy(),

            LearnedBurgers2DPolicy(
                CHECKPOINT,
                device=device,
                name="learned_dynamics",
            ),

            Burgers2DOraclePolicy(),
        ]

        policies[
            -1
        ].name = (
            "perfect_future_graph"
        )

        metrics = {}

        for i, policy in enumerate(
            policies,
            start=1,
        ):
            start = (
                time.perf_counter()
            )

            print(
                f"[{i}/3] "
                f"Running {policy.name} ...",
                flush=True,
            )

            df = run_policy(
                policy,
                scenario_name=scenario,
                horizon=horizon,
                seed=seed,
            )

            metrics[
                policy.name
            ] = summarize_policy(
                df
            )

            print(
                f"    finished in "
                f"{time.perf_counter()-start:.1f} s",
                flush=True,
            )

        r = metrics[
            "reactive"
        ]

        l = metrics[
            "learned_dynamics"
        ]

        p = metrics[
            "perfect_future_graph"
        ]

        denom = (
            r[
                "total_objective"
            ]
            - p[
                "total_objective"
            ]
        )

        capture = (
            (
                r[
                    "total_objective"
                ]
                - l[
                    "total_objective"
                ]
            )
            / denom
            if abs(
                denom
            ) > 1e-12
            else np.nan
        )

        row = {
            "regime":
                regime,

            "scenario":
                scenario,

            "horizon":
                horizon,

            "seed":
                seed,

            "graph_mismatch":
                mismatch,

            "objective_improvement_vs_reactive":
                (
                    r[
                        "total_objective"
                    ]
                    / l[
                        "total_objective"
                    ]
                ),

            "latency_improvement_vs_reactive":
                (
                    r[
                        "mean_estimated_latency"
                    ]
                    / l[
                        "mean_estimated_latency"
                    ]
                ),

            "byte_hop_improvement_vs_reactive":
                (
                    r[
                        "total_byte_hops"
                    ]
                    / l[
                        "total_byte_hops"
                    ]
                ),

            "perfect_prediction_capture":
                capture,
        }

        rows.append(
            row
        )

        pd.DataFrame(
            rows
        ).to_csv(
            RAW_PATH,
            index=False,
        )

        print(
            "Saved case immediately."
        )

        print(
            f"objective improvement="
            f"{row['objective_improvement_vs_reactive']:.4f}x"
        )

        print(
            f"latency improvement="
            f"{row['latency_improvement_vs_reactive']:.4f}x"
        )

        print(
            f"perfect capture="
            f"{row['perfect_prediction_capture']:.4f}"
        )

    final = pd.DataFrame(
        rows
    )

    # Only aggregate the moderate/high target cases here.
    subset = final[
        final.apply(
            lambda row:
            (
                str(
                    row.get(
                        "regime",
                        ""
                    )
                ),
                str(
                    row.get(
                        "scenario",
                        ""
                    )
                ),
                int(
                    row.get(
                        "horizon",
                        -1
                    )
                ),
                int(
                    row.get(
                        "seed",
                        -1
                    )
                ),
            )
            in {
                (
                    regime,
                    scenario,
                    horizon,
                    seed,
                )
                for (
                    regime,
                    scenario,
                    horizon,
                    _
                ) in POINTS
                for seed in SEEDS
            },
            axis=1,
        )
    ].copy()

    summary = (
        subset.groupby(
            [
                "regime",
                "scenario",
                "horizon",
            ],
            as_index=False,
        )
        .agg(
            graph_mismatch=(
                "graph_mismatch",
                "mean",
            ),

            objective_improvement_mean=(
                "objective_improvement_vs_reactive",
                "mean",
            ),

            objective_improvement_std=(
                "objective_improvement_vs_reactive",
                "std",
            ),

            latency_improvement_mean=(
                "latency_improvement_vs_reactive",
                "mean",
            ),

            latency_improvement_std=(
                "latency_improvement_vs_reactive",
                "std",
            ),

            perfect_capture_mean=(
                "perfect_prediction_capture",
                "mean",
            ),

            perfect_capture_std=(
                "perfect_prediction_capture",
                "std",
            ),

            n=(
                "seed",
                "count",
            ),
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        "\n=== TRIMMED MULTI-SEED SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n=== PAPER-FRIENDLY VIEW ==="
    )

    for _, row in summary.iterrows():
        print(
            f"{row['regime']:>8s} | "
            f"n={int(row['n'])} | "
            f"graph mismatch ~ "
            f"{100*row['graph_mismatch']:.1f}% | "
            f"objective "
            f"{row['objective_improvement_mean']:.3f} "
            f"+/- "
            f"{row['objective_improvement_std']:.3f}x | "
            f"latency "
            f"{row['latency_improvement_mean']:.3f} "
            f"+/- "
            f"{row['latency_improvement_std']:.3f}x | "
            f"capture "
            f"{100*row['perfect_capture_mean']:.1f} "
            f"+/- "
            f"{100*row['perfect_capture_std']:.1f}%"
        )


if __name__ == "__main__":
    main()
