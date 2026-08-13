from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from learned.burgers2d_surrogate import (
    Burgers2DSurrogate,
    rollout_surrogate,
)
from learned.burgers2d_data import (
    state_to_tensor,
)
from learned.ood_hard_breakdown import (
    SCENARIOS,
    make_hard_workload,
)
from mapping.learned_burgers2d_policy import (
    LearnedBurgers2DPolicy,
)


CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "burgers2d_surrogate.pt"
)

SEEDS = [
    50000,
    50001,
    50002,
]

HORIZONS = [
    16,
    24,
    32,
]

# Keep starts away from trajectory end so all horizons are valid.
STARTS = [
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
]


def graph_mismatch_fraction(
    gp,
    gt,
):
    keys = (
        set(gp)
        | set(gt)
    )

    diff = sum(
        abs(
            gp.get(k, 0.0)
            - gt.get(k, 0.0)
        )
        for k in keys
    )

    total = sum(
        gt.values()
    )

    return float(
        0.5
        * diff
        / max(
            total,
            1e-12,
        )
    )


def load_model(
    device,
):
    ckpt = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model = (
        Burgers2DSurrogate(
            hidden_channels=int(
                ckpt.get(
                    "hidden_channels",
                    48,
                )
            )
        )
        .to(device)
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = load_model(
        device
    )

    learned_policy = (
        LearnedBurgers2DPolicy(
            CHECKPOINT,
            device=device,
            name="learned_dynamics",
        )
    )

    rows = []

    for scenario_name, scenario in SCENARIOS.items():
        print(
            f"\nScenario: {scenario_name}",
            flush=True,
        )

        for seed in SEEDS:
            w = make_hard_workload(
                seed=seed,
                scenario=scenario,
            )

            # Basic truth sanity: avoid interpreting blown-up PDE traces.
            finite = True
            for u, v in w.state_history:
                if not (
                    np.all(
                        np.isfinite(u)
                    )
                    and
                    np.all(
                        np.isfinite(v)
                    )
                ):
                    finite = False
                    break

            if not finite:
                rows.append(
                    {
                        "scenario":
                            scenario_name,
                        "seed":
                            seed,
                        "horizon":
                            -1,
                        "mean_rollout_mse":
                            np.nan,
                        "mean_graph_mismatch":
                            np.nan,
                        "max_graph_mismatch":
                            np.nan,
                        "exact_graph_fraction":
                            np.nan,
                        "status":
                            "nonfinite_truth",
                    }
                )
                continue

            for horizon in HORIZONS:
                mses = []
                mismatches = []
                exacts = []

                for current_step in STARTS:
                    target_step = (
                        current_step
                        + horizon
                    )

                    if target_step > w.timesteps:
                        continue

                    x = (
                        state_to_tensor(
                            w.state_history[
                                current_step
                            ]
                        )
                        .unsqueeze(0)
                        .to(device)
                    )

                    pred = (
                        rollout_surrogate(
                            model,
                            x,
                            horizon,
                        )[-1][0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    true = (
                        state_to_tensor(
                            w.state_history[
                                target_step
                            ]
                        )
                        .numpy()
                    )

                    mse = float(
                        np.mean(
                            (
                                pred
                                - true
                            )
                            ** 2
                        )
                    )

                    policy_timestep = (
                        current_step
                        - 1
                    )

                    gp = (
                        learned_policy.predicted_graph(
                            workload=w,
                            timestep=policy_timestep,
                            horizon=horizon,
                            observed_graphs=[],
                        )
                    )

                    gt = (
                        w.communication_graph(
                            query_step=target_step
                        )
                    )

                    gm = (
                        graph_mismatch_fraction(
                            gp,
                            gt,
                        )
                    )

                    mses.append(
                        mse
                    )

                    mismatches.append(
                        gm
                    )

                    exacts.append(
                        1.0
                        if gm == 0.0
                        else 0.0
                    )

                rows.append(
                    {
                        "scenario":
                            scenario_name,

                        "seed":
                            seed,

                        "horizon":
                            horizon,

                        "mean_rollout_mse":
                            float(
                                np.mean(mses)
                            ),

                        "mean_graph_mismatch":
                            float(
                                np.mean(
                                    mismatches
                                )
                            ),

                        "max_graph_mismatch":
                            float(
                                np.max(
                                    mismatches
                                )
                            ),

                        "exact_graph_fraction":
                            float(
                                np.mean(
                                    exacts
                                )
                            ),

                        "status":
                            "ok",
                    }
                )

    raw = pd.DataFrame(
        rows
    )

    result_dir = (
        ROOT
        / "results"
    )

    result_dir.mkdir(
        exist_ok=True
    )

    raw.to_csv(
        result_dir
        / "hard_breakdown_raw.csv",
        index=False,
    )

    valid = raw[
        raw[
            "status"
        ] == "ok"
    ].copy()

    summary = (
        valid.groupby(
            [
                "scenario",
                "horizon",
            ],
            as_index=False,
        )
        .agg(
            rollout_mse=(
                "mean_rollout_mse",
                "mean",
            ),

            graph_mismatch=(
                "mean_graph_mismatch",
                "mean",
            ),

            max_graph_mismatch=(
                "max_graph_mismatch",
                "max",
            ),

            exact_graph_fraction=(
                "exact_graph_fraction",
                "mean",
            ),
        )
    )

    summary.to_csv(
        result_dir
        / "hard_breakdown_summary.csv",
        index=False,
    )

    print(
        "\n=== HARD BREAKDOWN DIAGNOSTIC ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    def print_band(
        title,
        lo,
        hi,
    ):
        subset = summary[
            (
                summary[
                    "graph_mismatch"
                ] >= lo
            )
            &
            (
                summary[
                    "graph_mismatch"
                ] < hi
            )
        ]

        print(
            f"\n=== {title} ==="
        )

        if len(subset):
            print(
                subset.to_string(
                    index=False
                )
            )
        else:
            print(
                "No candidates."
            )

    print_band(
        "LOW MISMATCH: 1-5%",
        0.01,
        0.05,
    )

    print_band(
        "MODERATE MISMATCH: 5-15%",
        0.05,
        0.15,
    )

    print_band(
        "HIGH MISMATCH: 15-35%",
        0.15,
        0.35,
    )


if __name__ == "__main__":
    main()
