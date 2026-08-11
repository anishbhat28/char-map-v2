from __future__ import annotations

from typing import Dict


def communication_cost(mesh, edges, placement):
    return mesh.weighted_manhattan_cost(
        edges,
        placement,
    )


def migration_cost(
    mesh,
    old_placement: Dict[int, int],
    new_placement: Dict[int, int],
    state_size: float = 1.0,
) -> float:
    """
    Simple migration model:

        C_migration =
            sum_i state_size_i *
            ManhattanDistance(old_PE_i, new_PE_i)

    v0 assumes equal state size for all tasks.
    """
    cost = 0.0

    for task, old_pe in old_placement.items():
        new_pe = new_placement[task]

        cost += (
            state_size
            * mesh.distance(
                old_pe,
                new_pe,
            )
        )

    return float(cost)


def moved_tasks(
    old_placement: Dict[int, int],
    new_placement: Dict[int, int],
) -> int:
    return sum(
        1
        for task in old_placement
        if old_placement[task]
        != new_placement[task]
    )


def migration_aware_objective(
    mesh,
    edges,
    old_placement,
    candidate_placement,
    migration_lambda: float,
    state_size: float,
):
    comm = communication_cost(
        mesh,
        edges,
        candidate_placement,
    )

    mig = migration_cost(
        mesh,
        old_placement,
        candidate_placement,
        state_size=state_size,
    )

    return (
        comm
        + migration_lambda * mig,
        comm,
        mig,
    )


def greedy_swap_optimize(
    mesh,
    edges,
    initial_placement: Dict[int, int],
    max_passes: int = 6,
    migration_lambda: float = 0.0,
    state_size: float = 1.0,
    reference_placement: Dict[int, int] | None = None,
):
    """
    Pairwise-swap hill climbing.

    IMPORTANT:
    - `initial_placement` is the optimizer seed.
    - `reference_placement` is the placement whose state must be migrated
      to realize the candidate placement.

    If reference_placement is None, we use initial_placement.

    Objective:
        predicted communication cost
        + lambda * migration cost
    """
    placement = dict(initial_placement)

    if reference_placement is None:
        reference_placement = dict(initial_placement)
    else:
        reference_placement = dict(reference_placement)

    tasks = sorted(
        placement.keys()
    )

    best_obj, best_comm, best_mig = migration_aware_objective(
        mesh,
        edges,
        reference_placement,
        placement,
        migration_lambda,
        state_size,
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

                obj, comm, mig = migration_aware_objective(
                    mesh,
                    edges,
                    reference_placement,
                    placement,
                    migration_lambda,
                    state_size,
                )

                if obj + 1e-12 < best_obj:
                    best_obj = obj
                    best_comm = comm
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
        best_comm,
        best_mig,
    )
