from __future__ import annotations

from pathlib import Path
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learned.burgers2d_data import (
    Burgers2DOneStepDataset,
)
from learned.burgers2d_surrogate import (
    Burgers2DSurrogate,
)


ROOT = Path(
    __file__
).resolve().parents[1]


def set_seed(
    seed: int,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    total = 0.0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(
                device
            )

            y = y.to(
                device
            )

            pred = model(
                x
            )

            loss = F.mse_loss(
                pred,
                y,
                reduction="sum",
            )

            total += float(
                loss.item()
            )

            n += int(
                y.numel()
            )

    return (
        total / max(
            n,
            1,
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3,
    )

    parser.add_argument(
        "--train-trajectories",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--val-trajectories",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            ROOT
            / "checkpoints"
            / "burgers2d_surrogate.pt"
        ),
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Training on: {device}"
    )

    train_seeds = list(
        range(
            args.train_trajectories
        )
    )

    val_seeds = list(
        range(
            10000,
            10000
            + args.val_trajectories,
        )
    )

    print(
        "Generating training trajectories...",
        flush=True,
    )

    train_ds = (
        Burgers2DOneStepDataset(
            seeds=train_seeds,
            timesteps=80,
        )
    )

    print(
        "Generating validation trajectories...",
        flush=True,
    )

    val_ds = (
        Burgers2DOneStepDataset(
            seeds=val_seeds,
            timesteps=80,
        )
    )

    print(
        f"Train samples: {len(train_ds)}"
    )

    print(
        f"Val samples:   {len(val_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = (
        Burgers2DSurrogate(
            hidden_channels=48
        )
        .to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    best_val = float(
        "inf"
    )

    checkpoint = Path(
        args.checkpoint
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.perf_counter()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running = 0.0
        count = 0

        for x, y in train_loader:
            x = x.to(
                device
            )

            y = y.to(
                device
            )

            pred = model(
                x
            )

            loss = F.mse_loss(
                pred,
                y,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            running += float(
                loss.item()
            )

            count += 1

        train_mse = (
            running
            / max(
                count,
                1,
            )
        )

        val_mse = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"epoch={epoch:03d} "
            f"train_mse={train_mse:.8f} "
            f"val_mse={val_mse:.8f}",
            flush=True,
        )

        if val_mse < best_val:
            best_val = val_mse

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "hidden_channels":
                        48,

                    "best_val_mse":
                        best_val,

                    "seed":
                        args.seed,
                },
                checkpoint,
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"\nBest validation MSE: "
        f"{best_val:.10f}"
    )

    print(
        f"Saved checkpoint: "
        f"{checkpoint}"
    )

    print(
        f"Training wall time: "
        f"{elapsed:.1f} s"
    )


if __name__ == "__main__":
    main()
