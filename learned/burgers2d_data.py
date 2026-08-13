from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from workloads.burgers2d import (
    Burgers2DWorkload,
)


def make_randomized_workload(
    *,
    seed: int,
    nx: int = 8,
    ny: int = 8,
    timesteps: int = 100,
    dt: float = 0.004,
    viscosity: float = 0.002,
):
    """
    Create a Burgers2DWorkload with randomized initial conditions.

    We reuse the validated numerical solver from workloads/burgers2d.py,
    but overwrite the initial state and regenerate its trajectory.

    The model therefore trains across a family of dynamics rather than
    memorizing a single trajectory.
    """
    rng = random.Random(
        seed
    )

    w = Burgers2DWorkload(
        nx=nx,
        ny=ny,
        timesteps=timesteps,
        dt=dt,
        viscosity=viscosity,
        advection_scale=1.0,
    )

    xs = (
        np.arange(nx)
        + 0.5
    ) * w.dx

    ys = (
        np.arange(ny)
        + 0.5
    ) * w.dy

    X, Y = np.meshgrid(
        xs,
        ys,
    )

    mean_u = rng.uniform(
        0.68,
        0.92,
    )

    mean_v = rng.uniform(
        0.44,
        0.68,
    )

    amp_u_y = rng.uniform(
        0.12,
        0.24,
    )

    amp_u_x = rng.uniform(
        0.05,
        0.13,
    )

    amp_v_x = rng.uniform(
        0.10,
        0.22,
    )

    amp_v_y = rng.uniform(
        0.04,
        0.12,
    )

    phi1 = rng.uniform(
        0.0,
        2.0 * math.pi,
    )

    phi2 = rng.uniform(
        0.0,
        2.0 * math.pi,
    )

    phi3 = rng.uniform(
        0.0,
        2.0 * math.pi,
    )

    phi4 = rng.uniform(
        0.0,
        2.0 * math.pi,
    )

    u0 = (
        mean_u
        + amp_u_y
        * np.sin(
            2.0
            * np.pi
            * Y
            + phi1
        )
        + amp_u_x
        * np.cos(
            2.0
            * np.pi
            * X
            + phi2
        )
    )

    v0 = (
        mean_v
        + amp_v_x
        * np.sin(
            2.0
            * np.pi
            * X
            + phi3
        )
        - amp_v_y
        * np.cos(
            2.0
            * np.pi
            * Y
            + phi4
        )
    )

    w.initial_u = (
        u0.astype(float)
    )

    w.initial_v = (
        v0.astype(float)
    )

    w.state_history = (
        w._precompute_true_history()
    )

    return w


def state_to_tensor(
    state,
):
    u, v = state

    x = np.stack(
        [
            u,
            v,
        ],
        axis=0,
    ).astype(
        np.float32
    )

    return torch.from_numpy(
        x
    )


class Burgers2DOneStepDataset(
    Dataset
):
    """
    One-step training pairs:
        z_t -> z_{t+1}

    Training over many randomized trajectories lets the network learn
    spatial dynamics rather than a single trace.
    """

    def __init__(
        self,
        *,
        seeds: List[int],
        nx: int = 8,
        ny: int = 8,
        timesteps: int = 80,
        dt: float = 0.004,
        viscosity: float = 0.002,
        stride: int = 1,
    ):
        self.samples: List[
            Tuple[
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []

        for seed in seeds:
            w = make_randomized_workload(
                seed=seed,
                nx=nx,
                ny=ny,
                timesteps=timesteps,
                dt=dt,
                viscosity=viscosity,
            )

            for t in range(
                0,
                timesteps,
                stride,
            ):
                x = state_to_tensor(
                    w.state_history[t]
                )

                y = state_to_tensor(
                    w.state_history[
                        t + 1
                    ]
                )

                self.samples.append(
                    (
                        x,
                        y,
                    )
                )

    def __len__(
        self,
    ):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx,
    ):
        return self.samples[
            idx
        ]
