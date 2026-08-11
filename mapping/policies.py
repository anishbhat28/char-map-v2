from graph.communication import linear_extrapolate_graph
class StaticPolicy:
    name='static'
    def reset(self):self.g=None
    def predicted_graph(self,*,workload,timestep,horizon,observed_graphs):
        if self.g is None:self.g=workload.communication_graph(1)
        return self.g
class ReactivePolicy:
    name='reactive'
    def reset(self):pass
    def predicted_graph(self,*,workload,timestep,horizon,observed_graphs):
        return observed_graphs[-1] if observed_graphs else workload.communication_graph(timestep+1)
class HistoryPolicy:
    name='history'
    def reset(self):pass
    def predicted_graph(self,*,workload,timestep,horizon,observed_graphs):
        if len(observed_graphs)<2:return observed_graphs[-1] if observed_graphs else workload.communication_graph(timestep+1)
        return linear_extrapolate_graph(observed_graphs[-2],observed_graphs[-1],workload.num_cells,horizon)
class CharacteristicPolicy:
    name='characteristic'
    def __init__(self,amplitude_error=.02):self.amplitude_error=amplitude_error
    def reset(self):pass
    def predicted_graph(self,*,workload,timestep,horizon,observed_graphs):
        return workload.communication_graph(timestep+horizon+1,1+self.amplitude_error)
class OraclePolicy:
    name='oracle'
    def reset(self):pass
    def predicted_graph(self,*,workload,timestep,horizon,observed_graphs):
        return workload.communication_graph(timestep+horizon+1,1.0)
