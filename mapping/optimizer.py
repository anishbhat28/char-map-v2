import random

def greedy_swap_optimize(mesh,edges,initial_placement,max_passes=2,samples_per_pass=64,seed=0):
    p=dict(initial_placement); tasks=sorted(p); rng=random.Random(seed)
    best=mesh.weighted_manhattan_cost(edges,p)
    for _ in range(max_passes):
        improved=False
        for _ in range(samples_per_pass):
            a,b=rng.sample(tasks,2)
            pa,pb=p[a],p[b]; p[a],p[b]=pb,pa
            c=mesh.weighted_manhattan_cost(edges,p)
            if c+1e-12<best:
                best=c; improved=True
            else:
                p[a],p[b]=pa,pb
        if not improved: break
    return p,best
