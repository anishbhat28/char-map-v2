import numpy as np
def to_matrix(edges,n):
    m=np.zeros((n,n))
    for (s,d),w in edges.items():m[s,d]+=w
    return m
def from_matrix(m):
    out={}
    rows,cols=np.nonzero(m>1e-12)
    for i,j in zip(rows,cols): out[(int(i),int(j))]=float(m[i,j])
    return out
def linear_extrapolate_graph(prev,curr,n,horizon=1):
    p=to_matrix(prev,n); c=to_matrix(curr,n); x=np.maximum(c+horizon*(c-p),0)
    return from_matrix(x)
