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

from workloads.burgers2d import (
    Burgers2DWorkload,
)


w = Burgers2DWorkload(
    nx=8,
    ny=8,
    timesteps=100,
    dt=0.004,
    viscosity=0.002,
)

assert (
    w.num_cells
    == 64
)

assert (
    len(
        w.state_history
    )
    == 101
)

for u, v in w.state_history:
    assert np.all(
        np.isfinite(
            u
        )
    )

    assert np.all(
        np.isfinite(
            v
        )
    )

late = (
    w.nonlocal_edge_fraction(
        100
    )
)

assert (
    late >= 0.25
), (
    "2-D Burgers workload did not develop enough "
    "nonlocal communication."
)

print(
    "All fixed 2-D Burgers workload tests passed."
)
