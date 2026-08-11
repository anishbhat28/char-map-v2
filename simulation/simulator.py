from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import pandas as pd

from architecture.mesh import Mesh
from mapping.optimizer import greedy_swap_optimize


@dataclass
class MappingSimConfig:
    rows: int = 8
    cols: int = 8

    # A placement decision made at timestep t becomes active at t + horizon.
    horizon: int = 2

    optimizer_passes: int = 4
    remap_every: int = 1


class MappingSimulator:
    """
    Correct temporal semantics:

    At timestep t:
      1. Evaluate the placement that is ACTIVE at t against actual G_t.
      2. Using information available at t, predict/choose a graph for t+H.
      3. Optimize a new placement for that predicted graph.
      4. Queue that placement so it becomes active at t+H.

    Therefore every policy is evaluated on the SAME actual graph G_t.
    What differs is the information used H steps earlier to choose the placement
    that became active at t.
    """

    def __init__(self, workload, policy, cfg: MappingSimConfig):
        self.workload = workload
        self.policy = policy
        self.cfg = cfg
        self.mesh = Mesh(cfg.rows, cfg.cols)

        if workload.num_cells > self.mesh.num_pes:
            raise ValueError(
                "Need at least one PE per task in v0."
            )

        # Identity placement at startup.
        self.initial_placement = {
            task: task
            for task in range(workload.num_cells)
        }

        self.active_placement = dict(self.initial_placement)

        # activation_timestep -> placement
        self.pending_placements: Dict[int, Dict[int, int]] = {}

        self.observed_graphs = []
        self.policy.reset()

    def run(self):
        rows = []

        for t in range(self.workload.timesteps):
            # ------------------------------------------------------------
            # 1. Activate any placement whose lead time has elapsed.
            # ------------------------------------------------------------
            if t in self.pending_placements:
                self.active_placement = self.pending_placements.pop(t)

            # ------------------------------------------------------------
            # 2. Ground-truth communication occurring NOW.
            # ------------------------------------------------------------
            actual_graph = self.workload.communication_graph(
                query_step=t + 1,
                amplitude_scale=1.0,
            )

            # Evaluate ALL policies on actual G_t using the placement that
            # was chosen H timesteps earlier.
            actual_cost = self.mesh.weighted_manhattan_cost(
                actual_graph,
                self.active_placement,
            )

            # ------------------------------------------------------------
            # 3. Make a placement decision for t + H.
            # ------------------------------------------------------------
            target_t = t + self.cfg.horizon

            predicted_cost = float("nan")
            scheduled = False

            if (
                target_t < self.workload.timesteps
                and t % self.cfg.remap_every == 0
            ):
                predicted_graph = self.policy.predicted_graph(
                    workload=self.workload,
                    timestep=t,
                    horizon=self.cfg.horizon,
                    observed_graphs=self.observed_graphs,
                )

                # Use the currently active placement as optimizer seed.
                new_placement, predicted_cost = greedy_swap_optimize(
                    self.mesh,
                    predicted_graph,
                    self.active_placement,
                    max_passes=self.cfg.optimizer_passes,
                )

                self.pending_placements[target_t] = dict(new_placement)
                scheduled = True

            rows.append(
                {
                    "timestep": t,
                    "policy": self.policy.name,
                    "actual_cost": actual_cost,
                    "predicted_cost": predicted_cost,
                    "placement_scheduled": scheduled,
                    "activation_target": (
                        target_t if scheduled else -1
                    ),
                }
            )

            # ------------------------------------------------------------
            # 4. Only after evaluation/decision does G_t become history.
            # ------------------------------------------------------------
            self.observed_graphs.append(actual_graph)

        return pd.DataFrame(rows)
