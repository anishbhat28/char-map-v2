
class Characteristic2DPolicy:
    def __init__(self,name="physics2d",omega_error=0.0,shear_error=0.0,frequency_error=0.0,remove_shear=False):
        self.name=name; self.omega_error=omega_error; self.shear_error=shear_error
        self.frequency_error=frequency_error; self.remove_shear=remove_shear
    def reset(self): pass
    def predicted_graph(self,*,workload,timestep,horizon,observed_graphs):
        return workload.predicted_communication_graph(
            timestep+horizon+1,
            omega_scale=1+self.omega_error,
            shear_scale=1+self.shear_error,
            frequency_scale=1+self.frequency_error,
            remove_shear=self.remove_shear)

class Oracle2DPolicy:
    name="oracle2d"
    def reset(self): pass
    def predicted_graph(self,*,workload,timestep,horizon,observed_graphs):
        return workload.communication_graph(timestep+horizon+1)
