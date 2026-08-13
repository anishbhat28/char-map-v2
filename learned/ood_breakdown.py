from dataclasses import dataclass

from learned.burgers2d_data import (
    make_randomized_workload,
)


@dataclass(frozen=True)
class BreakdownScenario:
    name: str
    viscosity_scale: float
    advection_scale: float
    u_initial_scale: float
    v_initial_scale: float


SCENARIOS = {
    # Reference
    "id": BreakdownScenario(
        "id",
        1.0,
        1.0,
        1.0,
        1.0,
    ),

    # Existing severe case
    "severe": BreakdownScenario(
        "severe",
        4.0,
        1.30,
        1.25,
        0.80,
    ),

    # Stronger shifts
    "extreme1": BreakdownScenario(
        "extreme1",
        6.0,
        1.45,
        1.35,
        0.70,
    ),

    "extreme2": BreakdownScenario(
        "extreme2",
        8.0,
        1.60,
        1.45,
        0.60,
    ),

    "extreme3": BreakdownScenario(
        "extreme3",
        10.0,
        1.80,
        1.60,
        0.50,
    ),
}


def make_breakdown_workload(
    *,
    seed: int,
    scenario: BreakdownScenario,
    nx: int = 8,
    ny: int = 8,
    timesteps: int = 100,
    dt: float = 0.004,
    base_viscosity: float = 0.002,
):
    w = make_randomized_workload(
        seed=seed,
        nx=nx,
        ny=ny,
        timesteps=timesteps,
        dt=dt,
        viscosity=(
            base_viscosity
            * scenario.viscosity_scale
        ),
    )

    w.advection_scale = (
        scenario.advection_scale
    )

    w.initial_u = (
        w.initial_u
        * scenario.u_initial_scale
    )

    w.initial_v = (
        w.initial_v
        * scenario.v_initial_scale
    )

    # Regenerate the true trajectory after changing dynamics.
    w.state_history = (
        w._precompute_true_history()
    )

    return w
