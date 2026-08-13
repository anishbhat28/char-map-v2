from pathlib import Path
import sys

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

import torch

from learned.burgers2d_surrogate import (
    Burgers2DSurrogate,
    rollout_surrogate,
)


model = Burgers2DSurrogate(
    hidden_channels=16
)

x = torch.randn(
    2,
    2,
    8,
    8,
)

y = model(
    x
)

assert (
    y.shape
    == x.shape
)

rollout = rollout_surrogate(
    model,
    x[:1],
    4,
)

assert (
    len(
        rollout
    )
    == 5
)

assert all(
    z.shape
    == (
        1,
        2,
        8,
        8,
    )
    for z in rollout
)

print(
    "All learned Burgers2D tests passed."
)
