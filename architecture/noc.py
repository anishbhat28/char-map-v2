
from dataclasses import dataclass
import math

@dataclass
class NoCConfig:
    bytes_per_unit: float = 256.0
    link_bandwidth_bytes_per_cycle: float = 64.0
    router_latency_cycles: float = 1.0

def xy_route(mesh, src_pe, dst_pe):
    if src_pe == dst_pe:
        return []
    sr,sc = mesh.coord(src_pe); dr,dc = mesh.coord(dst_pe)
    r,c = sr,sc; route=[]
    while c != dc:
        nc = c + (1 if dc > c else -1)
        a = r*mesh.cols+c; b = r*mesh.cols+nc
        route.append((a,b)); c=nc
    while r != dr:
        nr = r + (1 if dr > r else -1)
        a = r*mesh.cols+c; b = nr*mesh.cols+c
        route.append((a,b)); r=nr
    return route

def route_graph(mesh, edges, placement, cfg):
    loads={}; total_bytes=0.0; total_messages=0.0; max_hops=0
    for (src_task,dst_task),w in edges.items():
        if w <= 0: continue
        path = xy_route(mesh, placement[src_task], placement[dst_task])
        b = float(w)*cfg.bytes_per_unit
        total_messages += float(w); total_bytes += b; max_hops=max(max_hops,len(path))
        for link in path:
            loads[link]=loads.get(link,0.0)+b
    byte_hops=sum(loads.values())
    max_load=max(loads.values()) if loads else 0.0
    mean_load=(byte_hops/len(loads)) if loads else 0.0
    hot=sum(1 for x in loads.values() if max_load>0 and x>=0.8*max_load)
    ser=(max_load/cfg.link_bandwidth_bytes_per_cycle
         if cfg.link_bandwidth_bytes_per_cycle>0 else math.inf)
    lat=ser+cfg.router_latency_cycles*max_hops
    return {
        "total_messages":total_messages,
        "total_bytes":total_bytes,
        "byte_hops":byte_hops,
        "max_link_load_bytes":max_load,
        "mean_active_link_load_bytes":mean_load,
        "hot_link_count":hot,
        "max_hops":max_hops,
        "serialization_cycles":ser,
        "estimated_latency_cycles":lat,
    }, loads
