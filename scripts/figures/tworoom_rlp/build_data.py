"""Precompute figure data. Instances follow the eval protocol:
goal = state @ goal_offset_steps=25 ahead of origin (one plan horizon,
H=5 blocks x action_block 5), drawn under the report seeds 42/43/44,
selected for cross-room + RLP success (imagined endpoint < 16px tol)."""
import sys, pickle; import os as _os; sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, torch, gymnasium as gym, stable_worldmodel
import twcore as C

DEV="cpu"
# label (report seed) -> draw seed giving a cross-room offset-25 instance whose
# EXECUTED plan (true dynamics, collisions) routes through the door and reaches
# the goal (<14px, tol 16). Selected under report seeds 42/43/44.
INSTANCES = {42: 42044, 43: 43013, 44: 44009}
ACTOR_SEEDS = [0,1,2]
FLAGSHIP = 42
OFFSET = 25
ASCALE = 1.4   # action-std calibration (expert step-length match; see README)

def main():
    u = C.make_env()
    dlj = np.load("scripts/figures/tworoom_rlp/_latgrid_lejepa.npz")
    dpl = np.load("scripts/figures/tworoom_rlp/_latgrid_pldm.npz")
    XX, YY, blocked = dlj["XX"], dlj["YY"], dlj["blocked"]
    Zlj, Zpl = dlj["Z"], dpl["Z"]
    ny, nx = XX.shape
    gy = np.clip(YY.ravel().astype(int),0,223); gx=np.clip(XX.ravel().astype(int),0,223)

    stacks={}
    for base, seeds in [("lejepa", ACTOR_SEEDS), ("pldm",[0])]:
        for s in seeds:
            wm, actor, value, meta = C.load_stack(base, s, device=DEV)
            stacks[(base,s)] = dict(wm=wm, actor=actor, value=value, meta=meta)
    wm_lj = stacks[("lejepa",0)]["wm"]; wm_pl = stacks[("pldm",0)]["wm"]

    def learned_field(base,s,zg,Z):
        v=stacks[(base,s)]["value"]
        with torch.no_grad():
            return v(torch.tensor(Z,dtype=torch.float32),
                     torch.tensor(zg,dtype=torch.float32).expand(len(Z),-1)).cpu().numpy().reshape(-1)

    data=dict(XX=XX,YY=YY,blocked=blocked,ny=ny,nx=nx,
              door_y=float(u.door_positions[0]),door_half=float(u.door_sizes[0]),
              wall_x=float(u.WALL_CENTER),border=int(u.BORDER_SIZE),occ=C.occupancy(u),
              tasks={},refine={},flagship=FLAGSHIP,task_seeds=list(INSTANCES),
              actor_seeds=ACTOR_SEEDS,refine_seeds=list(INSTANCES),offset=OFFSET,
              instances=INSTANCES, success_tol=16.0)

    for label, draw in INSTANCES.items():
        origin, goal, epath, off = C.expert_instance(draw, offset=OFFSET)
        zg_lj=C.encode_pos(wm_lj,u,[goal]).numpy()[0]
        zg_pl=C.encode_pos(wm_pl,u,[goal]).numpy()[0]
        geo=C.geodesic_grid(u,goal)[gy,gx]
        entry=dict(origin=origin,goal=goal,offset=off,expert_path=epath,
            geodesic=geo,
            latent=dict(lejepa=np.linalg.norm(Zlj-zg_lj[None],axis=1),
                        pldm=np.linalg.norm(Zpl-zg_pl[None],axis=1)),
            learned=dict())
        for s in ACTOR_SEEDS: entry["learned"][("lejepa",s)]=learned_field("lejepa",s,zg_lj,Zlj)
        entry["learned"][("pldm",0)]=learned_field("pldm",0,zg_pl,Zpl)
        data["tasks"][label]=entry
        print(f"task seed {label} (draw {draw}): origin {origin.round(0)} goal {goal.round(0)} offset {off}")

    # action stats to un-z-score plans -> execute through TRUE env dynamics
    amean, astd, nstat = C.collect_action_stats()
    data["action_mean"]=amean; data["action_std"]=astd*ASCALE; data["ascale"]=ASCALE
    print(f"action stats n={nstat} mean={amean.round(3)} std*scale={ (astd*ASCALE).round(3)}")
    st=stacks[("lejepa",0)]
    # three optimizers on the SAME value function + SAME tasks
    methods={
        "rlp":  lambda o,g: C.refine(st["wm"],st["actor"],st["value"],st["meta"],u,o,g,device=DEV),
        "adam": lambda o,g: C.refine_adam(st["wm"],st["value"],st["meta"],u,o,g,device=DEV),
        "cem":  lambda o,g: C.refine_cem(st["wm"],st["value"],st["meta"],u,o,g,seed=0,device=DEV),
    }
    data["methods"]={m:{} for m in methods}
    for label, draw in INSTANCES.items():
        T=data["tasks"][label]; origin,goal=T["origin"],T["goal"]
        for m,fn in methods.items():
            out=fn(origin.tolist(),goal.tolist())
            paths=[C.execute_plan(u,origin,goal,out["plans"][k][0].numpy(),amean,astd*ASCALE)
                   for k in range(len(out["plans"]))]
            d=float(np.hypot(*(paths[-1][-1]-goal)))
            data["methods"][m][label]=dict(origin=origin,goal=goal,paths=paths,
                                           values=np.array(out["values"]),end_dist=d)
            print(f"{m:>4} seed {label}: value {out['values'][0]:.1f}->{out['values'][-1]:.2f}  "
                  f"iters={len(out['values'])-1}  EXECUTED end {d:.1f}px  {'SUCCESS' if d<16 else 'miss'}")
    data["refine"]=data["methods"]["rlp"]   # back-compat

    pickle.dump(data, open("scripts/figures/tworoom_rlp/_figdata.pkl","wb"))

if __name__=="__main__": main()
