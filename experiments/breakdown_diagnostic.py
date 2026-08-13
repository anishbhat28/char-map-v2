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
from learned.ood_breakdown import (
    SCENARIOS,
    make_breakdown_workload,
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
    4,
    8,
    12,
    16,
]

# Sample enough points to see decision failures without doing every timestep.
STARTS = [
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
]


def graph_mismatch_fraction(
    gp,
    gt,
):
    keys = set(gp) | set(gt)

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
            w = make_breakdown_workload(
                seed=seed,
                scenario=scenario,
            )

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

                    # policy current state convention:
                    # simulator timestep + 1 = current_step
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
                                np.mean(
                                    mses
                                )
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
        / "breakdown_diagnostic_raw.csv",
        index=False,
    )

    summary = (
        raw.groupby(
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
        / "breakdown_diagnostic_summary.csv",
        index=False,
    )

    print(
        "\n=== BREAKDOWN DIAGNOSTIC ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n=== CANDIDATES WITH 5-30% MEAN GRAPH MISMATCH ==="
    )

    candidates = summary[
        (
            summary[
                "graph_mismatch"
            ] >= 0.05
        )
        &
        (
            summary[
                "graph_mismatch"
            ] <= 0.30
        )
    ]

    if len(candidates):
        print(
            candidates.to_string(
                index=False
            )
        )
    else:
        print(
            "No candidate currently lands in the 5-30% mismatch band."
        )


if __name__ == "__main__":
    main()
