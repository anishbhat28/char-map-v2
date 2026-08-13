from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from learned.burgers2d_surrogate import (
    Burgers2DSurrogate,
    rollout_surrogate,
)
from learned.burgers2d_data import (
    state_to_tensor,
)


class LearnedBurgers2DPolicy:
    """
    Learned-dynamics Char-Map policy.

    Important:
    - No future ground-truth state is read.
    - No PDE solver is used for future prediction.
    - The policy sees the current true state only.
    - Future state is produced by the trained neural surrogate.
    - The learned future state trajectory is converted into a predicted
      future communication graph using characteristic geometry.

    This is the first bridge from:
        physics-aware mapping
    to:
        world-model-aware mapping.
    """

    name = "learned_dynamics"

    def __init__(
        self,
        checkpoint_path,
        *,
        device=None,
        name="learned_dynamics",
    ):
        self.name = name

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Learned dynamics checkpoint not found: "
                f"{self.checkpoint_path}. "
                "Train it first with "
                "`python learned/train_burgers2d_surrogate.py`."
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        hidden = int(
            checkpoint.get(
                "hidden_channels",
                48,
            )
        )

        self.model = (
            Burgers2DSurrogate(
                hidden_channels=hidden
            )
            .to(
                self.device
            )
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

    def reset(
        self,
    ):
        pass

    def predicted_graph(
        self,
        *,
        workload,
        timestep,
        horizon,
        observed_graphs,
    ):
        # Simulator's realized G_t uses state/query timestep + 1.
        current_step = min(
            timestep + 1,
            workload.timesteps,
        )

        target_step = min(
            timestep
            + horizon
            + 1,
            workload.timesteps,
        )

        effective_horizon = (
            target_step
            - current_step
        )

        current_state = (
            workload.state_history[
                current_step
            ]
        )

        x = (
            state_to_tensor(
                current_state
            )
            .unsqueeze(0)
            .to(
                self.device
            )
        )

        learned_rollout = (
            rollout_surrogate(
                self.model,
                x,
                effective_horizon,
            )
        )

        # Convert tensors to the workload's [(u,v), ...] format.
        future_states = []

        for z in learned_rollout:
            arr = (
                z[
                    0
                ]
                .detach()
                .cpu()
                .numpy()
            )

            u = (
                arr[0]
                .astype(
                    np.float64
                )
            )

            v = (
                arr[1]
                .astype(
                    np.float64
                )
            )

            future_states.append(
                (
                    u,
                    v,
                )
            )

        # Observed true history through current_step, then learned future.
        combined = [
            (
                u.copy(),
                v.copy(),
            )
            for u, v
            in workload.state_history[
                : current_step + 1
            ]
        ]

        combined.extend(
            [
                (
                    u.copy(),
                    v.copy(),
                )
                for u, v
                in future_states[
                    1:
                ]
            ]
        )

        edges = {}

        for dest in range(
            workload.num_cells
        ):
            src = (
                workload._backtrack_over_states(
                    dest,
                    combined,
                    advection_scale=(
                        workload.advection_scale
                    ),
                    remove_crossflow=False,
                )
            )

            key = (
                src,
                dest,
            )

            edges[
                key
            ] = (
                edges.get(
                    key,
                    0.0,
                )
                + 1.0
            )

        return edges
