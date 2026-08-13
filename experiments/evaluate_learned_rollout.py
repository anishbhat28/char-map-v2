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

import numpy as np
import torch

from learned.burgers2d_data import (
    make_randomized_workload,
    state_to_tensor,
)
from learned.burgers2d_surrogate import (
    Burgers2DSurrogate,
    rollout_surrogate,
)


CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "burgers2d_surrogate.pt"
)

EVAL_SEED = 50000


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
        .to(
            device
        )
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )

    model.eval()

    w = make_randomized_workload(
        seed=EVAL_SEED,
        nx=8,
        ny=8,
        timesteps=100,
        dt=0.004,
        viscosity=0.002,
    )

    horizons = [
        1,
        2,
        4,
        8,
    ]

    starts = [
        10,
        30,
        50,
        70,
    ]

    print(
        "\n=== LEARNED ROLLOUT MSE ==="
    )

    for h in horizons:
        errors = []

        for t in starts:
            x = (
                state_to_tensor(
                    w.state_history[t]
                )
                .unsqueeze(0)
                .to(
                    device
                )
            )

            pred = rollout_surrogate(
                model,
                x,
                h,
            )[-1][0].detach().cpu().numpy()

            true = (
                state_to_tensor(
                    w.state_history[
                        t + h
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

            errors.append(
                mse
            )

        print(
            f"H={h:2d} "
            f"mean_mse="
            f"{np.mean(errors):.8f}"
        )


if __name__ == "__main__":
    main()
