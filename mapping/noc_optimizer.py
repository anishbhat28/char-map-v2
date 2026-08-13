from __future__ import annotations

from typing import Dict
import random

from architecture.noc import (
    NoCConfig,
    route_graph,
)
from mapping.optimizer import (
    migration_cost,
)


def noc_objective(
    mesh,
    edges,
    reference_placement,
    candidate_placement,
    *,
    noc_cfg: NoCConfig,
    alpha: float = 1.0,
    beta: float = 1.0,
    migration_lambda: float = 0.0,
    state_size: float = 1.0,
):
    metrics, _ = route_graph(
        mesh,
        edges,
        candidate_placement,
        noc_cfg,
    )

    mig = migration_cost(
        mesh,
        reference_placement,
        candidate_placement,
        state_size=state_size,
    )

    obj = (
        alpha * metrics["byte_hops"]
        + beta * metrics["max_link_load_bytes"]
        + migration_lambda * mig
    )

    return obj, metrics, mig


def greedy_noc_swap_optimize(
    mesh,
    edges,
    initial_placement: Dict[int, int],
    *,
    noc_cfg: NoCConfig,
    max_passes: int = 4,
    alpha: float = 1.0,
    beta: float = 1.0,
    migration_lambda: float = 0.0,
    state_size: float = 1.0,
    reference_placement: Dict[int, int] | None = None,
):
    """
    Single-start pairwise-swap hill climber.
    """
    placement = dict(initial_placement)

    if reference_placement is None:
        reference_placement = dict(initial_placement)
    else:
        reference_placement = dict(reference_placement)

    (
        best_obj,
        best_metrics,
        best_mig,
    ) = noc_objective(
        mesh,
        edges,
        reference_placement,
        placement,
        noc_cfg=noc_cfg,
        alpha=alpha,
        beta=beta,
        migration_lambda=migration_lambda,
        state_size=state_size,
    )

    tasks = sorted(
        placement.keys()
    )

    for _ in range(max_passes):
        improved = False

        for ai in range(len(tasks)):
            a = tasks[ai]

            for bi in range(
                ai + 1,
                len(tasks),
            ):
                b = tasks[bi]

                pa = placement[a]
                pb = placement[b]

                placement[a], placement[b] = (
                    pb,
                    pa,
                )

                obj, metrics, mig = noc_objective(
                    mesh,
                    edges,
                    reference_placement,
                    placement,
                    noc_cfg=noc_cfg,
                    alpha=alpha,
                    beta=beta,
                    migration_lambda=migration_lambda,
                    state_size=state_size,
                )

                if obj + 1e-12 < best_obj:
                    best_obj = obj
                    best_metrics = metrics
                    best_mig = mig
                    improved = True
                else:
                    placement[a], placement[b] = (
                        pa,
                        pb,
                    )

        if not improved:
            break

    return (
        placement,
        best_obj,
        best_metrics,
        best_mig,
    )


def multistart_noc_swap_optimize(
    mesh,
    edges,
    initial_placement: Dict[int, int],
    *,
    noc_cfg: NoCConfig,
    max_passes: int = 4,
    alpha: float = 1.0,
    beta: float = 1.0,
    migration_lambda: float = 0.0,
    state_size: float = 1.0,
    reference_placement: Dict[int, int] | None = None,
    num_random_starts: int = 4,
    seed: int = 0,
):
    """
    Deterministic multi-start NoC optimizer.

    Starts from:
      1. predecessor/reference placement
      2. current initial placement
      3. identity placement when task and PE IDs align
      4. `num_random_starts` deterministic random PE permutations

    Every candidate is optimized using the SAME full objective:

        alpha * byte_hops
      + beta  * max_link_load
      + lambda * migration

    This is important: we are not choosing candidates using a surrogate.
    """
    if reference_placement is None:
        reference_placement = dict(
            initial_placement
        )
    else:
        reference_placement = dict(
            reference_placement
        )

    task_ids = sorted(
        initial_placement.keys()
    )

    pe_ids = sorted(
        initial_placement.values()
    )

    starts = []

    # Reference/predecessor placement.
    starts.append(
        dict(reference_placement)
    )

    # Existing initial placement, if distinct.
    starts.append(
        dict(initial_placement)
    )

    # Identity if valid.
    if set(task_ids) == set(pe_ids):
        starts.append(
            {
                task: task
                for task in task_ids
            }
        )

    # Deterministic random starts.
    rng = random.Random(seed)

    for _ in range(
        num_random_starts
    ):
        shuffled = pe_ids.copy()
        rng.shuffle(
            shuffled
        )

        starts.append(
            {
                task: pe
                for task, pe
                in zip(
                    task_ids,
                    shuffled,
                )
            }
        )

    # Deduplicate starts.
    unique_starts = []
    seen = set()

    for start in starts:
        key = tuple(
            start[t]
            for t in task_ids
        )

        if key not in seen:
            seen.add(key)
            unique_starts.append(
                start
            )

    best_placement = None
    best_obj = float("inf")
    best_metrics = None
    best_mig = None

    for start in unique_starts:
        (
            placement,
            obj,
            metrics,
            mig,
        ) = greedy_noc_swap_optimize(
            mesh,
            edges,
            start,
            noc_cfg=noc_cfg,
            max_passes=max_passes,
            alpha=alpha,
            beta=beta,
            migration_lambda=migration_lambda,
            state_size=state_size,
            reference_placement=reference_placement,
        )

        if obj + 1e-12 < best_obj:
            best_placement = dict(
                placement
            )
            best_obj = obj
            best_metrics = metrics
            best_mig = mig

    return (
        best_placement,
        best_obj,
        best_metrics,
        best_mig,
    )
