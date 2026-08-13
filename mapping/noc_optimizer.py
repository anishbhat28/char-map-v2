
from architecture.noc import route_graph
from mapping.optimizer import migration_cost

def noc_objective(mesh, edges, ref, cand, *, noc_cfg, alpha=1.0, beta=1.0,
                  migration_lambda=0.0, state_size=1.0):
    m,_ = route_graph(mesh, edges, cand, noc_cfg)
    mig = migration_cost(mesh, ref, cand, state_size=state_size)
    obj = alpha*m["byte_hops"] + beta*m["max_link_load_bytes"] + migration_lambda*mig
    return obj,m,mig

def greedy_noc_swap_optimize(mesh, edges, initial_placement, *, noc_cfg,
                             max_passes=4, alpha=1.0, beta=1.0,
                             migration_lambda=0.0, state_size=1.0,
                             reference_placement=None):
    p=dict(initial_placement)
    ref=dict(reference_placement or initial_placement)
    best_obj,best_m,best_mig=noc_objective(
        mesh,edges,ref,p,noc_cfg=noc_cfg,alpha=alpha,beta=beta,
        migration_lambda=migration_lambda,state_size=state_size)
    tasks=sorted(p)
    for _ in range(max_passes):
        improved=False
        for i,a in enumerate(tasks):
            for b in tasks[i+1:]:
                pa,pb=p[a],p[b]; p[a],p[b]=pb,pa
                obj,m,mig=noc_objective(
                    mesh,edges,ref,p,noc_cfg=noc_cfg,alpha=alpha,beta=beta,
                    migration_lambda=migration_lambda,state_size=state_size)
                if obj+1e-12 < best_obj:
                    best_obj,best_m,best_mig=obj,m,mig; improved=True
                else:
                    p[a],p[b]=pa,pb
        if not improved: break
    return p,best_obj,best_m,best_mig
