class Burgers2DCharacteristicPolicy:
    def __init__(
        self,
        *,
        name="burgers2d_physics",
        advection_error=0.0,
        viscosity_error=0.0,
        u_state_error=0.0,
        v_state_error=0.0,
        remove_crossflow=False,
    ):
        self.name = name
        self.advection_error = advection_error
        self.viscosity_error = viscosity_error
        self.u_state_error = u_state_error
        self.v_state_error = v_state_error
        self.remove_crossflow = remove_crossflow

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
        current = min(
            timestep + 1,
            workload.timesteps,
        )

        target = min(
            timestep
            + horizon
            + 1,
            workload.timesteps,
        )

        effective_horizon = (
            target - current
        )

        return workload.predicted_communication_graph(
            timestep=current,
            horizon=effective_horizon,

            advection_scale=(
                workload.advection_scale
                * (
                    1.0
                    + self.advection_error
                )
            ),

            viscosity_scale=(
                1.0
                + self.viscosity_error
            ),

            u_state_scale=(
                1.0
                + self.u_state_error
            ),

            v_state_scale=(
                1.0
                + self.v_state_error
            ),

            remove_crossflow=(
                self.remove_crossflow
            ),
        )


class Burgers2DOraclePolicy:
    name = "burgers2d_oracle"

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
        return workload.communication_graph(
            query_step=(
                timestep
                + horizon
                + 1
            )
        )
