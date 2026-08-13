class BurgersCharacteristicPolicy:
    def __init__(self,name="burgers_characteristic",advection_error=0.0,
                 viscosity_error=0.0,state_scale_error=0.0):
        self.name=name
        self.advection_error=advection_error
        self.viscosity_error=viscosity_error
        self.state_scale_error=state_scale_error
    def reset(self): pass
    def predicted_graph(self,*,workload,timestep,horizon,observed_graphs):
        current=min(timestep+1,workload.timesteps)
        target=min(timestep+horizon+1,workload.timesteps)
        return workload.predicted_communication_graph(
            current,target-current,
            advection_scale=workload.advection_scale*(1+self.advection_error),
            viscosity_scale=1+self.viscosity_error,
            state_scale=1+self.state_scale_error)

class BurgersOraclePolicy:
    name="burgers_oracle"
    def reset(self): pass
    def predicted_graph(self,*,workload,timestep,horizon,observed_graphs):
        return workload.communication_graph(timestep+horizon+1)
