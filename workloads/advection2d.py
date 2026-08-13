
from dataclasses import dataclass
import math

@dataclass
class RotatingAdvection2D:
    nx:int=6; ny:int=6; timesteps:int=60; dt:float=0.01
    domain_x:float=1.0; domain_y:float=1.0
    omega:float=1.5; shear:float=0.35; temporal_frequency:float=3.0
    rk4_substeps_per_dt:int=4

    @property
    def num_cells(self): return self.nx*self.ny
    @property
    def dx(self): return self.domain_x/self.nx
    @property
    def dy(self): return self.domain_y/self.ny

    def cell_to_xy(self,c):
        iy,ix=divmod(c,self.nx)
        return (ix+.5)*self.dx,(iy+.5)*self.dy

    def xy_to_cell(self,x,y):
        ix=int(math.floor((x%self.domain_x)/self.dx))%self.nx
        iy=int(math.floor((y%self.domain_y)/self.dy))%self.ny
        return iy*self.nx+ix

    def velocity(self,x,y,t):
        xc=x/self.domain_x-.5; yc=y/self.domain_y-.5
        vx=self.omega*yc+self.shear*math.sin(2*math.pi*y/self.domain_y)*math.cos(self.temporal_frequency*t)
        vy=-self.omega*xc+self.shear*math.sin(2*math.pi*x/self.domain_x)*math.sin(self.temporal_frequency*t)
        return vx,vy

    def model_velocity(self,x,y,t,omega_scale=1.0,shear_scale=1.0,frequency_scale=1.0,remove_shear=False):
        om=self.omega*omega_scale
        sh=0.0 if remove_shear else self.shear*shear_scale
        fr=self.temporal_frequency*frequency_scale
        xc=x/self.domain_x-.5; yc=y/self.domain_y-.5
        vx=om*yc+sh*math.sin(2*math.pi*y/self.domain_y)*math.cos(fr*t)
        vy=-om*xc+sh*math.sin(2*math.pi*x/self.domain_x)*math.sin(fr*t)
        return vx,vy

    def _rk4(self,x,y,t,h,vf):
        k1=vf(x,y,t)
        k2=vf(x+.5*h*k1[0],y+.5*h*k1[1],t+.5*h)
        k3=vf(x+.5*h*k2[0],y+.5*h*k2[1],t+.5*h)
        k4=vf(x+h*k3[0],y+h*k3[1],t+h)
        xn=(x+h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6)%self.domain_x
        yn=(y+h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6)%self.domain_y
        return xn,yn

    def _source(self,dest,query_step,vf):
        if query_step<=0:return dest
        x,y=self.cell_to_xy(dest); t=query_step*self.dt
        n=max(1,query_step*self.rk4_substeps_per_dt); h=-t/n
        for _ in range(n):
            x,y=self._rk4(x,y,t,h,vf); t+=h
        return self.xy_to_cell(x,y)

    def source_index(self,dest,query_step):
        return self._source(dest,query_step,self.velocity)

    def predicted_source_index(self,dest,query_step,omega_scale=1.0,shear_scale=1.0,frequency_scale=1.0,remove_shear=False):
        def vf(x,y,t):
            return self.model_velocity(x,y,t,omega_scale,shear_scale,frequency_scale,remove_shear)
        return self._source(dest,query_step,vf)

    def communication_graph(self,query_step,amplitude_scale=1.0):
        e={}
        for d in range(self.num_cells):
            s=self.source_index(d,query_step); e[(s,d)]=e.get((s,d),0.0)+1.0
        return e

    def predicted_communication_graph(self,query_step,omega_scale=1.0,shear_scale=1.0,frequency_scale=1.0,remove_shear=False):
        e={}
        for d in range(self.num_cells):
            s=self.predicted_source_index(d,query_step,omega_scale,shear_scale,frequency_scale,remove_shear)
            e[(s,d)]=e.get((s,d),0.0)+1.0
        return e
