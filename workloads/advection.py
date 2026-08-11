from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class RegimeSwitchingAdvection:
    """
    Ground-truth workload:
        u_t + c(x,t) u_x = 0

    with
        c(x,t) = c0 + A(t) sin(2*pi*x/L) cos(omega*t)

    True regime schedule:
        first third  -> amplitude_1
        second third -> amplitude_2
        final third  -> amplitude_3

    The ground-truth communication graph always uses the true parameters.

    The `predicted_communication_graph(...)` method lets a mapper deliberately
    use a misspecified physical model without changing the actual workload.
    """

    num_cells: int = 64
    timesteps: int = 100
    dt: float = 1.0 / 64.0
    domain_length: float = 1.0

    c0: float = 1.0
    omega: float = 4.0

    amplitude_1: float = 0.25
    amplitude_2: float = 0.75
    amplitude_3: float = 0.40

    rk4_substeps_per_dt: int = 8

    @property
    def dx(self) -> float:
        return self.domain_length / self.num_cells

    @property
    def total_time(self) -> float:
        return max(self.dt, self.timesteps * self.dt)

    # ------------------------------------------------------------------
    # TRUE PHYSICS
    # ------------------------------------------------------------------

    def amplitude_at(self, t: float) -> float:
        frac = t / self.total_time

        if frac < 1.0 / 3.0:
            return self.amplitude_1
        if frac < 2.0 / 3.0:
            return self.amplitude_2
        return self.amplitude_3

    def velocity(
        self,
        x: float,
        t: float,
    ) -> float:
        x = x % self.domain_length
        amp = self.amplitude_at(t)

        return (
            self.c0
            + amp
            * math.sin(2.0 * math.pi * x / self.domain_length)
            * math.cos(self.omega * t)
        )

    def _rk4_step_true(
        self,
        x: float,
        t: float,
        h: float,
    ) -> float:
        def f(xx, tt):
            return self.velocity(xx, tt)

        k1 = f(x, t)
        k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(x + h * k3, t + h)

        return (
            x
            + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        ) % self.domain_length

    def source_index(
        self,
        dest_cell: int,
        query_step: int,
    ) -> int:
        if query_step <= 0:
            return dest_cell % self.num_cells

        x = dest_cell * self.dx
        t = query_step * self.dt

        nsteps = max(
            1,
            query_step * self.rk4_substeps_per_dt,
        )
        h = -t / nsteps

        for _ in range(nsteps):
            x = self._rk4_step_true(
                x,
                t,
                h,
            )
            t += h

        return int(round(x / self.dx)) % self.num_cells

    def communication_graph(
        self,
        query_step: int,
        amplitude_scale: float = 1.0,
    ) -> Dict[Tuple[int, int], float]:
        """
        Compatibility note:
        amplitude_scale is retained because earlier policies call this method
        with amplitude_scale=1.0. Ground truth is always exact here.
        """
        edges: Dict[Tuple[int, int], float] = {}

        for dest in range(self.num_cells):
            src = self.source_index(
                dest,
                query_step=query_step,
            )
            key = (src, dest)
            edges[key] = edges.get(key, 0.0) + 1.0

        return edges

    # ------------------------------------------------------------------
    # MISSPECIFIED PHYSICS USED ONLY BY THE PREDICTOR
    # ------------------------------------------------------------------

    def _model_amplitude_at(
        self,
        t: float,
        *,
        amplitude_scale: float = 1.0,
        state_delay_steps: int = 0,
        model_amplitudes=None,
        model_switch_fractions=None,
    ) -> float:
        """
        Returns the amplitude believed by the mapping model.

        state_delay_steps:
            The model sees the regime/state from t-delay rather than t.

        model_amplitudes:
            Optional tuple (A1_hat, A2_hat, A3_hat).

        model_switch_fractions:
            Optional tuple (s1_hat, s2_hat).
        """
        stale_t = max(
            0.0,
            t - state_delay_steps * self.dt,
        )

        frac = stale_t / self.total_time

        if model_amplitudes is None:
            amps = (
                self.amplitude_1,
                self.amplitude_2,
                self.amplitude_3,
            )
        else:
            amps = model_amplitudes

        if model_switch_fractions is None:
            s1, s2 = 1.0 / 3.0, 2.0 / 3.0
        else:
            s1, s2 = model_switch_fractions

        if frac < s1:
            amp = amps[0]
        elif frac < s2:
            amp = amps[1]
        else:
            amp = amps[2]

        return amplitude_scale * amp

    def model_velocity(
        self,
        x: float,
        t: float,
        *,
        c0_scale: float = 1.0,
        omega_scale: float = 1.0,
        amplitude_scale: float = 1.0,
        state_delay_steps: int = 0,
        include_spatial_variation: bool = True,
        model_amplitudes=None,
        model_switch_fractions=None,
    ) -> float:
        x = x % self.domain_length

        c0_hat = self.c0 * c0_scale
        omega_hat = self.omega * omega_scale

        if not include_spatial_variation:
            return c0_hat

        amp_hat = self._model_amplitude_at(
            t,
            amplitude_scale=amplitude_scale,
            state_delay_steps=state_delay_steps,
            model_amplitudes=model_amplitudes,
            model_switch_fractions=model_switch_fractions,
        )

        return (
            c0_hat
            + amp_hat
            * math.sin(2.0 * math.pi * x / self.domain_length)
            * math.cos(omega_hat * t)
        )

    def _rk4_step_model(
        self,
        x: float,
        t: float,
        h: float,
        **model_kwargs,
    ) -> float:
        def f(xx, tt):
            return self.model_velocity(
                xx,
                tt,
                **model_kwargs,
            )

        k1 = f(x, t)
        k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(x + h * k3, t + h)

        return (
            x
            + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        ) % self.domain_length

    def predicted_source_index(
        self,
        dest_cell: int,
        query_step: int,
        **model_kwargs,
    ) -> int:
        if query_step <= 0:
            return dest_cell % self.num_cells

        x = dest_cell * self.dx
        t = query_step * self.dt

        nsteps = max(
            1,
            query_step * self.rk4_substeps_per_dt,
        )
        h = -t / nsteps

        for _ in range(nsteps):
            x = self._rk4_step_model(
                x,
                t,
                h,
                **model_kwargs,
            )
            t += h

        return int(round(x / self.dx)) % self.num_cells

    def predicted_communication_graph(
        self,
        query_step: int,
        **model_kwargs,
    ) -> Dict[Tuple[int, int], float]:
        """
        Communication graph predicted using a potentially wrong physics model.
        """
        edges: Dict[Tuple[int, int], float] = {}

        for dest in range(self.num_cells):
            src = self.predicted_source_index(
                dest,
                query_step=query_step,
                **model_kwargs,
            )
            key = (src, dest)
            edges[key] = edges.get(key, 0.0) + 1.0

        return edges
