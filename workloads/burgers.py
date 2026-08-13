import numpy as np
from dataclasses import dataclass

@dataclass
class BurgersWorkload:
    num_cells:int=36
    timesteps:int=60
    domain_length:float=1.0
    dt:float=0.003
    viscosity:float=0.002
    advection_scale:float=1.0

    def __post_init__(self):
        self.dx=self.domain_length/self.num_cells
        self.x=np.arange(self.num_cells)*self.dx
        self.initial_state=(0.75+0.22*np.sin(2*np.pi*self.x/self.domain_length)
                            +0.10*np.sin(4*np.pi*self.x/self.domain_length+0.4))
        self.state_history=self._precompute()

    def _step(self,u,advection_scale,viscosity):
        up=np.roll(u,-1); um=np.roll(u,1)
        f=.5*advection_scale*u*u
        fp=.5*advection_scale*up*up
        fm=.5*advection_scale*um*um
        ap=np.maximum(np.abs(advection_scale*u),np.abs(advection_scale*up))
        am=np.maximum(np.abs(advection_scale*um),np.abs(advection_scale*u))
        Fp=.5*(f+fp)-.5*ap*(up-u)
        Fm=.5*(fm+f)-.5*am*(u-um)
        return u-(self.dt/self.dx)*(Fp-Fm)+self.dt*viscosity*(up-2*u+um)/(self.dx*self.dx)

    def _precompute(self):
        hist=[self.initial_state.copy()]
        u=self.initial_state.copy()
        for _ in range(self.timesteps):
            u=self._step(u,self.advection_scale,self.viscosity)
            hist.append(u.copy())
        return hist

    def forecast_from(self,timestep,horizon,advection_scale=1.0,viscosity_scale=1.0,state_scale=1.0):
        t=max(0,min(int(timestep),self.timesteps))
        u=self.state_history[t].copy()*state_scale
        states=[u.copy()]
        for _ in range(horizon):
            u=self._step(u,advection_scale,self.viscosity*viscosity_scale)
            states.append(u.copy())
        return states

    def _interp(self,state,xq):
        xq=xq%self.domain_length
        s=xq/self.dx; base=np.floor(s); i0=int(base)%self.num_cells
        frac=s-base; i1=(i0+1)%self.num_cells
        return float((1-frac)*state[i0]+frac*state[i1])

    def _backtrack(self,dest,states,advection_scale):
        x=dest*self.dx
        for k in range(len(states)-1,0,-1):
            u_now=self._interp(states[k],x)
            xmid=(x-.5*self.dt*advection_scale*u_now)%self.domain_length
            u_prev=self._interp(states[k-1],xmid)
            x=(x-self.dt*advection_scale*u_prev)%self.domain_length
        return int(round(x/self.dx))%self.num_cells

    def communication_graph(self,query_step,amplitude_scale=1.0):
        q=max(0,min(int(query_step),self.timesteps))
        states=self.state_history[:q+1]
        edges={}
        for d in range(self.num_cells):
            s=self._backtrack(d,states,self.advection_scale)
            edges[(s,d)]=edges.get((s,d),0.0)+1.0
        return edges

    def predicted_communication_graph(self,timestep,horizon,advection_scale=1.0,
                                      viscosity_scale=1.0,state_scale=1.0):
        t=max(0,min(int(timestep),self.timesteps))
        target=min(t+int(horizon),self.timesteps)
        future=self.forecast_from(t,target-t,advection_scale,viscosity_scale,state_scale)
        combined=[s.copy() for s in self.state_history[:t+1]]+[s.copy() for s in future[1:]]
        edges={}
        for d in range(self.num_cells):
            s=self._backtrack(d,combined,advection_scale)
            edges[(s,d)]=edges.get((s,d),0.0)+1.0
        return edges
