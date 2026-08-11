from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd
from architecture.mesh import Mesh
from mapping.optimizer import greedy_swap_optimize, migration_cost, moved_tasks

@dataclass
class MappingSimConfig:
    rows: int = 8
    cols: int = 8
    horizon: int = 2
    optimizer_passes: int = 4
    remap_every: int = 1
    migration_lambda: float = 0.0
    task_state_size: float = 1.0

class MappingSimulator:
    def __init__(self, workload, policy, cfg: MappingSimConfig):
        self.workload = workload
        self.policy = policy
        self.cfg = cfg
        self.mesh = Mesh(cfg.rows, cfg.cols)
        if workload.num_cells > self.mesh.num_pes:
            raise ValueError("Need at least one PE per task in v0.")
        self.initial_placement = {task: task for task in range(workload.num_cells)}
        self.active_placement = dict(self.initial_placement)
        self.pending_placements: Dict[int, Dict[int, int]] = {}
        self.placement_history: Dict[int, Dict[int, int]] = {}
        self.observed_graphs = []
        self.policy.reset()

    def _expected_placement_at(self, timestep: int) -> Dict[int, int]:
        if timestep < 0:
            return dict(self.initial_placement)
        if timestep in self.placement_history:
            return dict(self.placement_history[timestep])
        for tt in range(timestep, -1, -1):
            if tt in self.pending_placements:
                return dict(self.pending_placements[tt])
            if tt in self.placement_history:
                return dict(self.placement_history[tt])
        return dict(self.initial_placement)

    def run(self):
        rows = []
        cumulative_migration = 0.0
        cumulative_comm = 0.0
        is_static = (self.policy.name == "static")

        for t in range(self.workload.timesteps):
            previous_placement = dict(self.active_placement)
            activation_migration = 0.0
            activation_moved_tasks = 0

            if (not is_static) and t in self.pending_placements:
                next_placement = self.pending_placements.pop(t)
                activation_migration = migration_cost(
                    self.mesh, previous_placement, next_placement,
                    state_size=self.cfg.task_state_size,
                )
                activation_moved_tasks = moved_tasks(previous_placement, next_placement)
                self.active_placement = dict(next_placement)

            self.placement_history[t] = dict(self.active_placement)
            cumulative_migration += activation_migration

            actual_graph = self.workload.communication_graph(
                query_step=t + 1, amplitude_scale=1.0
            )
            actual_comm_cost = self.mesh.weighted_manhattan_cost(
                actual_graph, self.active_placement
            )
            cumulative_comm += actual_comm_cost
            actual_total_cost = (
                actual_comm_cost
                + self.cfg.migration_lambda * activation_migration
            )

            target_t = t + self.cfg.horizon
            predicted_obj = float("nan")
            predicted_comm = float("nan")
            predicted_migration = float("nan")
            scheduled = False

            if (
                (not is_static)
                and target_t < self.workload.timesteps
                and t % self.cfg.remap_every == 0
            ):
                predicted_graph = self.policy.predicted_graph(
                    workload=self.workload,
                    timestep=t,
                    horizon=self.cfg.horizon,
                    observed_graphs=self.observed_graphs,
                )
                predecessor_placement = self._expected_placement_at(target_t - 1)

                (
                    new_placement,
                    predicted_obj,
                    predicted_comm,
                    predicted_migration,
                ) = greedy_swap_optimize(
                    self.mesh,
                    predicted_graph,
                    predecessor_placement,
                    max_passes=self.cfg.optimizer_passes,
                    migration_lambda=self.cfg.migration_lambda,
                    state_size=self.cfg.task_state_size,
                    reference_placement=predecessor_placement,
                )

                self.pending_placements[target_t] = dict(new_placement)
                scheduled = True

            rows.append(
                {
                    "timestep": t,
                    "policy": self.policy.name,
                    "actual_comm_cost": actual_comm_cost,
                    "activation_migration_cost": activation_migration,
                    "activation_moved_tasks": activation_moved_tasks,
                    "actual_total_cost": actual_total_cost,
                    "predicted_objective": predicted_obj,
                    "predicted_comm_cost": predicted_comm,
                    "predicted_migration_cost": predicted_migration,
                    "placement_scheduled": scheduled,
                    "activation_target": target_t if scheduled else -1,
                    "cumulative_comm_cost": cumulative_comm,
                    "cumulative_migration_cost": cumulative_migration,
                }
            )
            self.observed_graphs.append(actual_graph)

        return pd.DataFrame(rows)
