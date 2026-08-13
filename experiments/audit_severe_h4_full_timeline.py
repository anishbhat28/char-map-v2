from pathlib import Path
import sys, numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned.burgers2d_surrogate import Burgers2DSurrogate, rollout_surrogate
from learned.burgers2d_data import state_to_tensor
from learned.ood_burgers2d import SCENARIOS, make_ood_workload
from mapping.learned_burgers2d_policy import LearnedBurgers2DPolicy

CHECKPOINT = ROOT / "checkpoints" / "burgers2d_surrogate.pt"
SCENARIO = "severe"
SEED = 50000
HORIZON = 4

def graph_mismatch_fraction(gp, gt):
    keys = set(gp) | set(gt)
    diff = sum(abs(gp.get(k,0.0)-gt.get(k,0.0)) for k in keys)
    total = sum(gt.values())
    return 0.5 * diff / max(total, 1e-12)

def differing_destinations(gp, gt):
    ps, ts = {}, {}
    for (src,dst),w in gp.items():
        if w > 0: ps[dst] = src
    for (src,dst),w in gt.items():
        if w > 0: ts[dst] = src
    return [d for d in (set(ps)|set(ts)) if ps.get(d) != ts.get(d)]

def load_model(device):
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model = Burgers2DSurrogate(int(ckpt.get("hidden_channels",48))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(device)
    policy = LearnedBurgers2DPolicy(
        CHECKPOINT, device=device, name="learned_dynamics"
    )

    w = make_ood_workload(
        seed=SEED, scenario=SCENARIOS[SCENARIO],
        nx=8, ny=8, timesteps=100, dt=.004, base_viscosity=.002
    )

    rows = []
    for pt in range(0, w.timesteps - HORIZON):
        current = pt + 1
        target = pt + HORIZON + 1

        x = state_to_tensor(w.state_history[current]).unsqueeze(0).to(device)
        pred = rollout_surrogate(model, x, HORIZON)[-1][0].detach().cpu().numpy()
        true = state_to_tensor(w.state_history[target]).numpy()
        mse = float(np.mean((pred-true)**2))

        gp = policy.predicted_graph(
            workload=w, timestep=pt, horizon=HORIZON, observed_graphs=[]
        )
        gt = w.communication_graph(query_step=target)

        mismatch = graph_mismatch_fraction(gp, gt)
        diffs = differing_destinations(gp, gt)

        rows.append({
            "policy_timestep": pt,
            "current_step": current,
            "target_step": target,
            "rollout_mse": mse,
            "graph_mismatch": mismatch,
            "num_wrong_destinations": len(diffs),
            "fraction_wrong_destinations": len(diffs)/w.num_cells,
            "exact_graph": int(len(diffs)==0),
            "wrong_destinations": ",".join(map(str,diffs)),
        })

    df = pd.DataFrame(rows)
    (ROOT/"results").mkdir(exist_ok=True)
    out_path = ROOT/"results/severe_h4_full_timeline_audit.csv"
    df.to_csv(out_path,index=False)

    print("\n=== SEVERE OOD FULL-TIMELINE H=4 AUDIT ===")
    print(f"Timesteps audited: {len(df)}")
    print(f"Mean rollout MSE: {df.rollout_mse.mean():.8e}")
    print(f"Median rollout MSE: {df.rollout_mse.median():.8e}")
    print(f"P95 rollout MSE: {df.rollout_mse.quantile(.95):.8e}")
    print(f"Max rollout MSE: {df.rollout_mse.max():.8e}")
    print(f"Mean graph mismatch: {df.graph_mismatch.mean():.8f}")
    print(f"P95 graph mismatch: {df.graph_mismatch.quantile(.95):.8f}")
    print(f"Max graph mismatch: {df.graph_mismatch.max():.8f}")
    print(f"Exact-graph fraction: {df.exact_graph.mean():.4f}")
    print(f"Timesteps with graph error: {int((df.exact_graph==0).sum())}")

    bad = df[df.exact_graph==0].sort_values(
        ["graph_mismatch","rollout_mse"], ascending=False
    )
    if len(bad):
        print("\n=== TIMESTEPS WITH GRAPH MISMATCH ===")
        print(bad[[
            "policy_timestep","current_step","target_step","rollout_mse",
            "graph_mismatch","num_wrong_destinations","fraction_wrong_destinations"
        ]].to_string(index=False))
    else:
        print("\nNo graph mismatches occurred at any H=4 timestep.")

    print(f"\nSaved detailed audit to: {out_path}")

if __name__ == "__main__":
    main()
