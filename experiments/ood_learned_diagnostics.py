from pathlib import Path
import sys, numpy as np, pandas as pd, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

from learned.burgers2d_surrogate import Burgers2DSurrogate, rollout_surrogate
from learned.burgers2d_data import state_to_tensor
from learned.ood_burgers2d import SCENARIOS, make_ood_workload
from mapping.learned_burgers2d_policy import LearnedBurgers2DPolicy

CKPT=ROOT/"checkpoints/burgers2d_surrogate.pt"
SEEDS=[50000,50001,50002,50003]
STARTS=[10,30,50,70]
HORIZONS=[1,2,4,8]

def graph_mismatch(gp,gt):
    keys=set(gp)|set(gt)
    diff=sum(abs(gp.get(k,0)-gt.get(k,0)) for k in keys)
    return .5*diff/max(sum(gt.values()),1e-12)

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(CKPT,map_location=device)
    model=Burgers2DSurrogate(int(ckpt.get("hidden_channels",48))).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    policy=LearnedBurgers2DPolicy(CKPT,device=device,name="learned_dynamics")
    rows=[]
    for sname,scenario in SCENARIOS.items():
        print(f"Scenario: {sname}",flush=True)
        for seed in SEEDS:
            w=make_ood_workload(seed,scenario)
            for h in HORIZONS:
                mses=[]; mism=[]
                for t in STARTS:
                    x=state_to_tensor(w.state_history[t]).unsqueeze(0).to(device)
                    pred=rollout_surrogate(model,x,h)[-1][0].detach().cpu().numpy()
                    true=state_to_tensor(w.state_history[t+h]).numpy()
                    mses.append(float(np.mean((pred-true)**2)))
                    gp=policy.predicted_graph(workload=w,timestep=t-1,horizon=h,observed_graphs=[])
                    gt=w.communication_graph(t+h)
                    mism.append(graph_mismatch(gp,gt))
                rows.append({"scenario":sname,"seed":seed,"horizon":h,
                    "rollout_mse":np.mean(mses),"graph_mismatch":np.mean(mism)})
    out=pd.DataFrame(rows)
    (ROOT/"results").mkdir(exist_ok=True)
    out.to_csv(ROOT/"results/ood_rollout_graph_sweep.csv",index=False)
    s=out.groupby(["scenario","horizon"],as_index=False).agg(
        rollout_mse=("rollout_mse","mean"),graph_mismatch=("graph_mismatch","mean"))
    print("\n=== OOD LEARNED-DYNAMICS DIAGNOSTIC ===")
    print(s.to_string(index=False))
    print("\n=== H=4 DECISION-RELEVANT VIEW ===")
    print(s[s.horizon==4].to_string(index=False))
if __name__=="__main__": main()
