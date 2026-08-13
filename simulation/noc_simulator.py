from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from architecture.mesh import Mesh
from architecture.noc import (
    NoCConfig,
    route_graph,
)
from mapping.optimizer import (
    migration_cost,
    moved_tasks,
)
from mapping.noc_optimizer import (
    multistart_noc_swap_optimize,
)


@dataclass
class NoCMappingSimConfig:
    rows: int = 8
    cols: int = 8

    horizon: int = 4
    optimizer_passes: int = 4
    remap_every: int = 1

    bytes_per_unit: float = 256.0
    link_bandwidth_bytes_per_cycle: float = 64.0
    router_latency_cycles: float = 1.0

    alpha_byte_hops: float = 1.0
    beta_max_link_load: float = 1.0

    migration_lambda: float = 0.5
    task_state_size: float = 256.0

    # NEW: stronger deterministic optimizer
    num_random_starts: int = 4
    multistart_seed: int = 1337


class NoCMappingSimulator:
    """
    NoC-aware mapping with:
      - delayed activation
      - migration planning
      - deterministic multi-start placement optimization

    Every policy receives the same number of starts and same seed schedule.
    """

    def __init__(
        self,
        workload,
        policy,
        cfg: NoCMappingSimConfig,
    ):
        self.workload = workload
        self.policy = policy
        self.cfg = cfg

        self.mesh = Mesh(
            cfg.rows,
            cfg.cols,
        )

        if workload.num_cells > self.mesh.num_pes:
            raise ValueError(
                "Need at least one PE per task in v0."
            )

        self.noc_cfg = NoCConfig(
            cfg.bytes_per_unit,
            cfg.link_bandwidth_bytes_per_cycle,
            cfg.router_latency_cycles,
        )

        self.initial = {
            i: i
            for i in range(
                workload.num_cells
            )
        }

        self.active = dict(
            self.initial
        )

        self.pending: Dict[
            int,
            Dict[int, int],
        ] = {}

        self.hist_place: Dict[
            int,
            Dict[int, int],
        ] = {}

        self.obs = []

        self.policy.reset()

    def expected(
        self,
        t: int,
    ):
        if t < 0:
            return dict(
                self.initial
            )

        if t in self.hist_place:
            return dict(
                self.hist_place[t]
            )

        for tt in range(
            t,
            -1,
            -1,
        ):
            if tt in self.pending:
                return dict(
                    self.pending[tt]
                )

            if tt in self.hist_place:
                return dict(
                    self.hist_place[tt]
                )

        return dict(
            self.initial
        )

    def run(self):
        rows = []

        static = (
            self.policy.name
            == "static"
        )

        for t in range(
            self.workload.timesteps
        ):
            prev = dict(
                self.active
            )

            migdist = 0.0
            moved = 0

            # --------------------------------------------------------
            # Activate queued mapping.
            # --------------------------------------------------------
            if (
                not static
                and t in self.pending
            ):
                nxt = self.pending.pop(
                    t
                )

                migdist = migration_cost(
                    self.mesh,
                    prev,
                    nxt,
                    state_size=1.0,
                )

                moved = moved_tasks(
                    prev,
                    nxt,
                )

                self.active = dict(
                    nxt
                )

            self.hist_place[
                t
            ] = dict(
                self.active
            )

            # --------------------------------------------------------
            # Realized communication.
            # --------------------------------------------------------
            g = (
                self.workload.communication_graph(
                    query_step=t + 1,
                    amplitude_scale=1.0,
                )
            )

            m, _ = route_graph(
                self.mesh,
                g,
                self.active,
                self.noc_cfg,
            )

            mig_bh = (
                migdist
                * self.cfg.task_state_size
            )

            obj = (
                self.cfg.alpha_byte_hops
                * m["byte_hops"]

                + self.cfg.beta_max_link_load
                * m["max_link_load_bytes"]

                + self.cfg.migration_lambda
                * mig_bh
            )

            # --------------------------------------------------------
            # Plan placement for target t+H.
            # --------------------------------------------------------
            target = (
                t
                + self.cfg.horizon
            )

            if (
                not static
                and target
                < self.workload.timesteps
                and t
                % self.cfg.remap_every
                == 0
            ):
                pg = (
                    self.policy.predicted_graph(
                        workload=self.workload,
                        timestep=t,
                        horizon=self.cfg.horizon,
                        observed_graphs=self.obs,
                    )
                )

                pred = (
                    self.expected(
                        target - 1
                    )
                )

                # Same deterministic seed at a given timestep for all policies.
                seed = (
                    self.cfg.multistart_seed
                    + t
                )

                (
                    np_,
                    _,
                    _,
                    _,
                ) = multistart_noc_swap_optimize(
                    self.mesh,
                    pg,
                    pred,
                    noc_cfg=self.noc_cfg,
                    max_passes=(
                        self.cfg.optimizer_passes
                    ),
                    alpha=(
                        self.cfg.alpha_byte_hops
                    ),
                    beta=(
                        self.cfg.beta_max_link_load
                    ),
                    migration_lambda=(
                        self.cfg.migration_lambda
                        * self.cfg.task_state_size
                    ),
                    state_size=1.0,
                    reference_placement=pred,
                    num_random_starts=(
                        self.cfg.num_random_starts
                    ),
                    seed=seed,
                )

                self.pending[
                    target
                ] = dict(
                    np_
                )

            rows.append(
                {
                    "timestep":
                        t,

                    "policy":
                        self.policy.name,

                    **m,

                    "migration_byte_hops":
                        mig_bh,

                    "moved_tasks":
                        moved,

                    "realized_objective":
                        obj,
                }
            )

            self.obs.append(
                g
            )

        return pd.DataFrame(
            rows
        )
