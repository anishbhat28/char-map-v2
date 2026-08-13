from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


EdgeMap = Dict[Tuple[int, int], float]


@dataclass
class Burgers2DWorkload:
    """
    Periodic 2-D viscous Burgers system:

        u_t + u u_x + v u_y = nu Laplacian(u)
        v_t + u v_x + v v_y = nu Laplacian(v)

    Characteristics:
        dx/dt = u(x,y,t)
        dy/dt = v(x,y,t)

    This version is deliberately rescaled so characteristics traverse
    multiple grid cells over the simulated time horizon.
    """

    nx: int = 8
    ny: int = 8
    timesteps: int = 100

    domain_x: float = 1.0
    domain_y: float = 1.0

    dt: float = 0.004
    viscosity: float = 0.002

    advection_scale: float = 1.0

    @property
    def num_cells(self) -> int:
        return self.nx * self.ny

    @property
    def dx(self) -> float:
        return self.domain_x / self.nx

    @property
    def dy(self) -> float:
        return self.domain_y / self.ny

    def __post_init__(self):
        xs = (
            np.arange(self.nx)
            + 0.5
        ) * self.dx

        ys = (
            np.arange(self.ny)
            + 0.5
        ) * self.dy

        X, Y = np.meshgrid(
            xs,
            ys,
        )

        # Stronger positive mean flow than the previous failed setup.
        # Goal: physical displacement over T should span multiple cells.
        u0 = (
            0.82
            + 0.20
            * np.sin(
                2.0 * np.pi * Y
            )
            + 0.10
            * np.cos(
                2.0 * np.pi * X
            )
        )

        v0 = (
            0.58
            + 0.18
            * np.sin(
                2.0 * np.pi * X
                + 0.3
            )
            - 0.09
            * np.cos(
                2.0 * np.pi * Y
            )
        )

        self.initial_u = (
            u0.astype(float)
        )

        self.initial_v = (
            v0.astype(float)
        )

        self.state_history = (
            self._precompute_true_history()
        )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def cell_to_xy(
        self,
        cell: int,
    ):
        iy = cell // self.nx
        ix = cell % self.nx

        return (
            (ix + 0.5) * self.dx,
            (iy + 0.5) * self.dy,
        )

    def xy_to_cell(
        self,
        x: float,
        y: float,
    ) -> int:
        x = (
            x % self.domain_x
        )

        y = (
            y % self.domain_y
        )

        ix = int(
            np.floor(
                x / self.dx
            )
        ) % self.nx

        iy = int(
            np.floor(
                y / self.dy
            )
        ) % self.ny

        return (
            iy * self.nx
            + ix
        )

    def _interp_periodic(
        self,
        field: np.ndarray,
        x: float,
        y: float,
    ) -> float:
        x = (
            x % self.domain_x
        )

        y = (
            y % self.domain_y
        )

        sx = (
            x / self.dx
            - 0.5
        )

        sy = (
            y / self.dy
            - 0.5
        )

        ix0f = np.floor(
            sx
        )

        iy0f = np.floor(
            sy
        )

        fx = (
            sx - ix0f
        )

        fy = (
            sy - iy0f
        )

        ix0 = int(
            ix0f
        ) % self.nx

        iy0 = int(
            iy0f
        ) % self.ny

        ix1 = (
            ix0 + 1
        ) % self.nx

        iy1 = (
            iy0 + 1
        ) % self.ny

        f00 = field[
            iy0,
            ix0,
        ]

        f10 = field[
            iy0,
            ix1,
        ]

        f01 = field[
            iy1,
            ix0,
        ]

        f11 = field[
            iy1,
            ix1,
        ]

        return float(
            (1.0 - fx)
            * (1.0 - fy)
            * f00
            + fx
            * (1.0 - fy)
            * f10
            + (1.0 - fx)
            * fy
            * f01
            + fx
            * fy
            * f11
        )

    # ------------------------------------------------------------
    # PDE solver
    # ------------------------------------------------------------

    def _rusanov_flux(
        self,
        qL,
        qR,
        velL,
        velR,
        scale,
    ):
        fL = (
            scale
            * velL
            * qL
        )

        fR = (
            scale
            * velR
            * qR
        )

        a = np.maximum(
            np.abs(
                scale * velL
            ),
            np.abs(
                scale * velR
            ),
        )

        return (
            0.5
            * (fL + fR)
            - 0.5
            * a
            * (qR - qL)
        )

    def _step(
        self,
        u: np.ndarray,
        v: np.ndarray,
        *,
        advection_scale: float,
        viscosity: float,
    ):
        uE = np.roll(
            u,
            -1,
            axis=1,
        )

        uW = np.roll(
            u,
            1,
            axis=1,
        )

        uN = np.roll(
            u,
            -1,
            axis=0,
        )

        uS = np.roll(
            u,
            1,
            axis=0,
        )

        vE = np.roll(
            v,
            -1,
            axis=1,
        )

        vW = np.roll(
            v,
            1,
            axis=1,
        )

        vN = np.roll(
            v,
            -1,
            axis=0,
        )

        vS = np.roll(
            v,
            1,
            axis=0,
        )

        Fux_E = self._rusanov_flux(
            u,
            uE,
            u,
            uE,
            advection_scale,
        )

        Fux_W = self._rusanov_flux(
            uW,
            u,
            uW,
            u,
            advection_scale,
        )

        Fvx_E = self._rusanov_flux(
            v,
            vE,
            u,
            uE,
            advection_scale,
        )

        Fvx_W = self._rusanov_flux(
            vW,
            v,
            uW,
            u,
            advection_scale,
        )

        Fuy_N = self._rusanov_flux(
            u,
            uN,
            v,
            vN,
            advection_scale,
        )

        Fuy_S = self._rusanov_flux(
            uS,
            u,
            vS,
            v,
            advection_scale,
        )

        Fvy_N = self._rusanov_flux(
            v,
            vN,
            v,
            vN,
            advection_scale,
        )

        Fvy_S = self._rusanov_flux(
            vS,
            v,
            vS,
            v,
            advection_scale,
        )

        div_u = (
            (
                Fux_E
                - Fux_W
            ) / self.dx
            + (
                Fuy_N
                - Fuy_S
            ) / self.dy
        )

        div_v = (
            (
                Fvx_E
                - Fvx_W
            ) / self.dx
            + (
                Fvy_N
                - Fvy_S
            ) / self.dy
        )

        lap_u = (
            (
                uE
                - 2.0 * u
                + uW
            )
            / (
                self.dx
                * self.dx
            )
            + (
                uN
                - 2.0 * u
                + uS
            )
            / (
                self.dy
                * self.dy
            )
        )

        lap_v = (
            (
                vE
                - 2.0 * v
                + vW
            )
            / (
                self.dx
                * self.dx
            )
            + (
                vN
                - 2.0 * v
                + vS
            )
            / (
                self.dy
                * self.dy
            )
        )

        un = (
            u
            - self.dt
            * div_u
            + self.dt
            * viscosity
            * lap_u
        )

        vn = (
            v
            - self.dt
            * div_v
            + self.dt
            * viscosity
            * lap_v
        )

        return un, vn

    def _precompute_true_history(
        self,
    ):
        history = [
            (
                self.initial_u.copy(),
                self.initial_v.copy(),
            )
        ]

        u = self.initial_u.copy()
        v = self.initial_v.copy()

        for _ in range(
            self.timesteps
        ):
            u, v = self._step(
                u,
                v,
                advection_scale=(
                    self.advection_scale
                ),
                viscosity=(
                    self.viscosity
                ),
            )

            if not (
                np.all(
                    np.isfinite(
                        u
                    )
                )
                and
                np.all(
                    np.isfinite(
                        v
                    )
                )
            ):
                raise FloatingPointError(
                    "2-D Burgers state became non-finite. "
                    "Reduce dt or initial velocity."
                )

            history.append(
                (
                    u.copy(),
                    v.copy(),
                )
            )

        return history

    def forecast_from(
        self,
        timestep: int,
        horizon: int,
        *,
        advection_scale: float = 1.0,
        viscosity_scale: float = 1.0,
        u_state_scale: float = 1.0,
        v_state_scale: float = 1.0,
    ):
        t = max(
            0,
            min(
                int(timestep),
                self.timesteps,
            ),
        )

        u = (
            self.state_history[
                t
            ][0].copy()
            * u_state_scale
        )

        v = (
            self.state_history[
                t
            ][1].copy()
            * v_state_scale
        )

        states = [
            (
                u.copy(),
                v.copy(),
            )
        ]

        for _ in range(
            horizon
        ):
            u, v = self._step(
                u,
                v,
                advection_scale=(
                    advection_scale
                ),
                viscosity=(
                    self.viscosity
                    * viscosity_scale
                ),
            )

            states.append(
                (
                    u.copy(),
                    v.copy(),
                )
            )

        return states

    # ------------------------------------------------------------
    # Characteristics
    # ------------------------------------------------------------

    def _velocity_from_state(
        self,
        state,
        x,
        y,
        *,
        advection_scale: float,
        remove_crossflow: bool,
    ):
        u, v = state

        ux = self._interp_periodic(
            u,
            x,
            y,
        )

        vy = self._interp_periodic(
            v,
            x,
            y,
        )

        if remove_crossflow:
            vy = 0.0

        return (
            advection_scale * ux,
            advection_scale * vy,
        )

    def _backtrack_over_states(
        self,
        dest_cell: int,
        states,
        *,
        advection_scale: float,
        remove_crossflow: bool = False,
    ):
        x, y = self.cell_to_xy(
            dest_cell
        )

        for k in range(
            len(states) - 1,
            0,
            -1,
        ):
            ux_now, vy_now = (
                self._velocity_from_state(
                    states[k],
                    x,
                    y,
                    advection_scale=(
                        advection_scale
                    ),
                    remove_crossflow=(
                        remove_crossflow
                    ),
                )
            )

            xmid = (
                x
                - 0.5
                * self.dt
                * ux_now
            ) % self.domain_x

            ymid = (
                y
                - 0.5
                * self.dt
                * vy_now
            ) % self.domain_y

            ux_prev, vy_prev = (
                self._velocity_from_state(
                    states[
                        k - 1
                    ],
                    xmid,
                    ymid,
                    advection_scale=(
                        advection_scale
                    ),
                    remove_crossflow=(
                        remove_crossflow
                    ),
                )
            )

            x = (
                x
                - self.dt
                * ux_prev
            ) % self.domain_x

            y = (
                y
                - self.dt
                * vy_prev
            ) % self.domain_y

        return self.xy_to_cell(
            x,
            y,
        )

    # ------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------

    def communication_graph(
        self,
        query_step: int,
        amplitude_scale: float = 1.0,
    ) -> EdgeMap:
        q = max(
            0,
            min(
                int(query_step),
                self.timesteps,
            ),
        )

        states = (
            self.state_history[
                : q + 1
            ]
        )

        edges: EdgeMap = {}

        for dest in range(
            self.num_cells
        ):
            src = (
                self._backtrack_over_states(
                    dest,
                    states,
                    advection_scale=(
                        self.advection_scale
                    ),
                    remove_crossflow=False,
                )
            )

            key = (
                src,
                dest,
            )

            edges[
                key
            ] = (
                edges.get(
                    key,
                    0.0,
                )
                + 1.0
            )

        return edges

    def predicted_communication_graph(
        self,
        *,
        timestep: int,
        horizon: int,
        advection_scale: float = 1.0,
        viscosity_scale: float = 1.0,
        u_state_scale: float = 1.0,
        v_state_scale: float = 1.0,
        remove_crossflow: bool = False,
    ) -> EdgeMap:
        t = max(
            0,
            min(
                int(timestep),
                self.timesteps,
            ),
        )

        target = min(
            t
            + int(horizon),
            self.timesteps,
        )

        forecast = self.forecast_from(
            t,
            target - t,
            advection_scale=(
                advection_scale
            ),
            viscosity_scale=(
                viscosity_scale
            ),
            u_state_scale=(
                u_state_scale
            ),
            v_state_scale=(
                v_state_scale
            ),
        )

        combined = [
            (
                u.copy(),
                v.copy(),
            )
            for u, v
            in self.state_history[
                : t + 1
            ]
        ]

        combined.extend(
            [
                (
                    u.copy(),
                    v.copy(),
                )
                for u, v
                in forecast[
                    1:
                ]
            ]
        )

        edges: EdgeMap = {}

        for dest in range(
            self.num_cells
        ):
            src = (
                self._backtrack_over_states(
                    dest,
                    combined,
                    advection_scale=(
                        advection_scale
                    ),
                    remove_crossflow=(
                        remove_crossflow
                    ),
                )
            )

            key = (
                src,
                dest,
            )

            edges[
                key
            ] = (
                edges.get(
                    key,
                    0.0,
                )
                + 1.0
            )

        return edges

    # ------------------------------------------------------------
    # Cheap workload diagnostics
    # ------------------------------------------------------------

    def nonlocal_edge_fraction(
        self,
        query_step: int,
    ) -> float:
        g = (
            self.communication_graph(
                query_step
            )
        )

        total = (
            sum(
                g.values()
            )
        )

        if total <= 0:
            return 0.0

        nonlocal_weight = sum(
            w
            for (
                src,
                dst
            ), w
            in g.items()
            if src != dst
        )

        return float(
            nonlocal_weight
            / total
        )
