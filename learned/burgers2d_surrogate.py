from __future__ import annotations

import torch
import torch.nn as nn


class Burgers2DSurrogate(nn.Module):
    """
    Tiny residual convolutional world model for 2-D Burgers.

    Input:
        [B, 2, H, W]
        channel 0 = u
        channel 1 = v

    Output:
        predicted next state with same shape.

    The model predicts a delta and adds it to the current state:
        z_{t+1} = z_t + Delta_theta(z_t)

    Circular padding preserves the periodic spatial topology.
    """

    def __init__(
        self,
        hidden_channels: int = 48,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                2,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                2,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return (
            state
            + self.net(state)
        )


@torch.no_grad()
def rollout_surrogate(
    model: nn.Module,
    initial_state: torch.Tensor,
    horizon: int,
):
    """
    Autoregressive learned rollout.

    Returns a list:
        [z_t, z_{t+1}, ..., z_{t+H}]

    Each entry is a tensor with shape [1, 2, H, W].
    """
    model.eval()

    states = [
        initial_state
    ]

    current = (
        initial_state
    )

    for _ in range(
        int(horizon)
    ):
        current = model(
            current
        )

        states.append(
            current
        )

    return states
