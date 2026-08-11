from __future__ import annotations

from graph.communication import linear_extrapolate_graph


class MappingPolicy:
    name = "base"

    def reset(self):
        pass

    def predicted_graph(
        self,
        *,
        workload,
        timestep,
        horizon,
        observed_graphs,
    ):
        raise NotImplementedError


class StaticPolicy(MappingPolicy):
    name = "static"

    def __init__(self):
        self.cached_graph = None

    def reset(self):
        self.cached_graph = None

    def predicted_graph(
        self,
        *,
        workload,
        timestep,
        horizon,
        observed_graphs,
    ):
        if self.cached_graph is None:
            self.cached_graph = workload.communication_graph(
                query_step=1
            )
        return self.cached_graph


class ReactivePolicy(MappingPolicy):
    name = "reactive"

    def predicted_graph(
        self,
        *,
        workload,
        timestep,
        horizon,
        observed_graphs,
    ):
        if observed_graphs:
            return observed_graphs[-1]

        return workload.communication_graph(
            query_step=timestep + 1
        )


class HistoryPolicy(MappingPolicy):
    name = "history"

    def predicted_graph(
        self,
        *,
        workload,
        timestep,
        horizon,
        observed_graphs,
    ):
        if len(observed_graphs) < 2:
            if observed_graphs:
                return observed_graphs[-1]

            return workload.communication_graph(
                query_step=timestep + 1
            )

        return linear_extrapolate_graph(
            observed_graphs[-2],
            observed_graphs[-1],
            workload.num_cells,
            horizon=horizon,
        )


class CharacteristicPolicy(MappingPolicy):
    """
    Backward-compatible physics policy.

    amplitude_error reproduces the previous scalar amplitude perturbation.
    """
    name = "characteristic"

    def __init__(
        self,
        amplitude_error: float = 0.0,
    ):
        self.amplitude_error = amplitude_error

    def predicted_graph(
        self,
        *,
        workload,
        timestep,
        horizon,
        observed_graphs,
    ):
        return workload.predicted_communication_graph(
            query_step=timestep + horizon + 1,
            amplitude_scale=1.0 + self.amplitude_error,
        )


class WrongPhysicsPolicy(MappingPolicy):
    """
    General misspecified-physics policy.

    All fields describe the model believed by the mapper.
    The actual workload remains unchanged.
    """

    def __init__(
        self,
        *,
        name: str,
        c0_scale: float = 1.0,
        omega_scale: float = 1.0,
        amplitude_scale: float = 1.0,
        state_delay_steps: int = 0,
        include_spatial_variation: bool = True,
        model_amplitudes=None,
        model_switch_fractions=None,
    ):
        self.name = name

        self.c0_scale = c0_scale
        self.omega_scale = omega_scale
        self.amplitude_scale = amplitude_scale
        self.state_delay_steps = state_delay_steps
        self.include_spatial_variation = include_spatial_variation
        self.model_amplitudes = model_amplitudes
        self.model_switch_fractions = model_switch_fractions

    def predicted_graph(
        self,
        *,
        workload,
        timestep,
        horizon,
        observed_graphs,
    ):
        return workload.predicted_communication_graph(
            query_step=timestep + horizon + 1,
            c0_scale=self.c0_scale,
            omega_scale=self.omega_scale,
            amplitude_scale=self.amplitude_scale,
            state_delay_steps=self.state_delay_steps,
            include_spatial_variation=self.include_spatial_variation,
            model_amplitudes=self.model_amplitudes,
            model_switch_fractions=self.model_switch_fractions,
        )


class OraclePolicy(MappingPolicy):
    name = "oracle"

    def predicted_graph(
        self,
        *,
        workload,
        timestep,
        horizon,
        observed_graphs,
    ):
        return workload.communication_graph(
            query_step=timestep + horizon + 1
        )
