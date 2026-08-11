import math
from dataclasses import dataclass
@dataclass
class RegimeSwitchingAdvection:
    num_cells:int=64; timesteps:int=60; dt:float=1/64; domain_length:float=1.0; c0:float=1.0; omega:float=4.0; amplitude_1:float=.25; amplitude_2:float=.75; amplitude_3:float=.40; rk4_substeps_per_dt:int=4
    @property
    def dx(self): return self.domain_length/self.num_cells
    def amplitude_at(self,t):
        f=t/max(self.dt,self.timesteps*self.dt)
        return self.amplitude_1 if f<1/3 else (self.amplitude_2 if f<2/3 else self.amplitude_3)
    def velocity(self,x,t,amplitude_scale=1.0):
        A=amplitude_scale*self.amplitude_at(t)
        return self.c0+A*math.sin(2*math.pi*(x%self.domain_length)/self.domain_length)*math.cos(self.omega*t)
    def _rk4(self,x,t,h,amplitude_scale):
        f=lambda xx,tt:self.velocity(xx,tt,amplitude_scale)
        k1=f(x,t); k2=f(x+.5*h*k1,t+.5*h); k3=f(x+.5*h*k2,t+.5*h); k4=f(x+h*k3,t+h)
        return (x+h*(k1+2*k2+2*k3+k4)/6)%self.domain_length
    def source_index(self,dest_cell,query_step,amplitude_scale=1.0):
        if query_step<=0:return dest_cell%self.num_cells
        x=dest_cell*self.dx; t=query_step*self.dt; n=max(1,query_step*self.rk4_substeps_per_dt); h=-t/n
        for _ in range(n): x=self._rk4(x,t,h,amplitude_scale); t+=h
        return int(round(x/self.dx))%self.num_cells
    def communication_graph(self,query_step,amplitude_scale=1.0):
        e={}
        for d in range(self.num_cells):
            s=self.source_index(d,query_step,amplitude_scale); e[(s,d)]=e.get((s,d),0)+1.0
        return e
