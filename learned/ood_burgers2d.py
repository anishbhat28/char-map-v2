from dataclasses import dataclass
from learned.burgers2d_data import make_randomized_workload

@dataclass(frozen=True)
class OODScenario:
    name:str
    viscosity_scale:float=1.0
    advection_scale:float=1.0
    u_initial_scale:float=1.0
    v_initial_scale:float=1.0

SCENARIOS={
    "id":OODScenario("id",1.0,1.0,1.0,1.0),
    "mild":OODScenario("mild",1.5,1.08,1.08,0.95),
    "moderate":OODScenario("moderate",2.0,1.15,1.15,0.90),
    "severe":OODScenario("severe",4.0,1.30,1.25,0.80),
}

def make_ood_workload(seed,scenario,nx=8,ny=8,timesteps=100,dt=.004,base_viscosity=.002):
    w=make_randomized_workload(seed=seed,nx=nx,ny=ny,timesteps=timesteps,dt=dt,
                               viscosity=base_viscosity*scenario.viscosity_scale)
    w.advection_scale=scenario.advection_scale
    w.initial_u=w.initial_u*scenario.u_initial_scale
    w.initial_v=w.initial_v*scenario.v_initial_scale
    w.state_history=w._precompute_true_history()
    return w
