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

from mapping.burgers2d_policies import (
    Burgers2DCharacteristicPolicy,
    Burgers2DOraclePolicy,
)


w = Burgers2DWorkload(
    nx=4,
    ny=4,
    timesteps=20,
    dt=0.0025,
    viscosity=0.002,
)

assert (
    w.num_cells
    == 16
)

assert (
    len(
        w.state_history
    )
    == 21
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

g = w.communication_graph(
    10
)

assert (
    sum(
        g.values()
    )
    == 16
)

p = Burgers2DCharacteristicPolicy(
    name="exact"
)

o = Burgers2DOraclePolicy()

gp = p.predicted_graph(
    workload=w,
    timestep=5,
    horizon=3,
    observed_graphs=[],
)

go = o.predicted_graph(
    workload=w,
    timestep=5,
    horizon=3,
    observed_graphs=[],
)

assert (
    gp == go
), (
    "Exact 2-D Burgers physics "
    "did not reproduce Oracle graph"
)

print(
    "All 2-D Burgers smoke tests passed."
)
