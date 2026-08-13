from pathlib import Path
import sys

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from architecture.mesh import Mesh
from architecture.noc import NoCConfig
from mapping.noc_optimizer import (
    greedy_noc_swap_optimize,
    multistart_noc_swap_optimize,
)


mesh = Mesh(
    4,
    4,
)

placement = {
    i: i
    for i in range(
        16
    )
}

# Simple asymmetric graph.
edges = {
    (0, 15): 1.0,
    (1, 14): 1.0,
    (2, 13): 1.0,
    (3, 12): 1.0,
}

cfg = NoCConfig(
    bytes_per_unit=256,
    link_bandwidth_bytes_per_cycle=64,
    router_latency_cycles=1,
)

single = greedy_noc_swap_optimize(
    mesh,
    edges,
    placement,
    noc_cfg=cfg,
    max_passes=2,
    alpha=1.0,
    beta=4.0,
    migration_lambda=0.5 * 256.0,
    state_size=1.0,
    reference_placement=placement,
)

multi = multistart_noc_swap_optimize(
    mesh,
    edges,
    placement,
    noc_cfg=cfg,
    max_passes=2,
    alpha=1.0,
    beta=4.0,
    migration_lambda=0.5 * 256.0,
    state_size=1.0,
    reference_placement=placement,
    num_random_starts=4,
    seed=123,
)

assert (
    multi[1]
    <= single[1]
    + 1e-9
), (
    "Multi-start optimizer was worse than "
    "single-start optimizer"
)

print(
    "All multi-start optimizer tests passed."
)
