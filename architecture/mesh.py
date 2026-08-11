from dataclasses import dataclass
@dataclass
class Mesh:
    rows:int; cols:int
    @property
    def num_pes(self): return self.rows*self.cols
    def coord(self,pe): return pe//self.cols, pe%self.cols
    def distance(self,a,b):
        ra,ca=self.coord(a); rb,cb=self.coord(b); return abs(ra-rb)+abs(ca-cb)
    def weighted_manhattan_cost(self,edges,placement):
        return float(sum(w*self.distance(placement[s],placement[d]) for (s,d),w in edges.items()))
