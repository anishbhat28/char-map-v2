from dataclasses import dataclass

from learned.burgers2d_data import (
    make_randomized_workload,
)


@dataclass(frozen=True)
class HardScenario:
    name: str
    viscosity_scale: float
    advection_scale: float
    u_initial_scale: float
    v_initial_scale: float


SCENARIOS = {
    "extreme3": HardScenario(
        "extreme3",
        viscosity_scale=10.0,
        advection_scale=1.80,
        u_initial_scale=1.60,
        v_initial_scale=0.50,
    ),

    "hard1": HardScenario(
        "hard1",
        viscosity_scale=12.0,
        advection_scale=2.00,
        u_initial_scale=1.75,
        v_initial_scale=0.42,
    ),

    "hard2": HardScenario(
        "hard2",
        viscosity_scale=14.0,
        advection_scale=2.30,
        u_initial_scale=1.95,
        v_initial_scale=0.34,
    ),

    "hard3": HardScenario(
        "hard3",
        viscosity_scale=16.0,
        advection_scale=2.60,
        u_initial_scale=2.15,
        v_initial_scale=0.28,
    ),

    "hard4": HardScenario(
        "hard4",
        viscosity_scale=18.0,
        advection_scale=2.90,
        u_initial_scale=2.35,
        v_initial_scale=0.22,
    ),
}


def make_hard_workload(
    *,
    seed: int,
    scenario: HardScenario,
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

    # Recompute truth under shifted dynamics.
    w.state_history = (
        w._precompute_true_history()
    )

    return w
