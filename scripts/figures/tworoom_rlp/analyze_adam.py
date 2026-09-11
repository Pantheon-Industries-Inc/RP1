"""Value landscape J(A)=V(z_T(A),z_g) around the Adam-stall case (seed 43)."""
import sys; import os as _os; sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, torch, twcore as C
from rlp.core.rollout import rollout_traj
from rlp.core.temporal import trajectory_value

u=C.make_env(); wm,actor,value,meta=C.load_stack("lejepa",0)
o,g,_,off=C.expert_instance(43013,offset=25)
H,adim=meta["horizon"],meta["a_dim"]
z_hist,zg=C._encode_hist_goal(wm,u,o.tolist(),g.tolist()); a_hist=torch.zeros(1,2,adim); z0=z_hist[:,-1]
def J(Aflat):  # (N, H*adim) -> (N,)
    A=torch.as_tensor(Aflat,dtype=torch.float32).view(-1,H,adim); N=A.shape[0]
    with torch.no_grad():
        tr=rollout_traj(wm,z_hist.expand(N,-1,-1),a_hist.expand(N,-1,-1),A)
        return trajectory_value(value,tr,zg.expand(N,-1),z0.expand(N,-1),meta["objective"]).cpu().numpy()

adam=C.refine_adam(wm,value,meta,u,o.tolist(),g.tolist()); rlp=C.refine(wm,actor,value,meta,u,o.tolist(),g.tolist())
A0=np.zeros(H*adim); Aa=adam["plans"][-1].view(-1).numpy(); Ar=rlp["plans"][-1].view(-1).numpy()
# plane spanned by (toward Adam) and (toward RLP, orthogonalized)
du=Aa-A0; uhat=du/np.linalg.norm(du)
wv=Ar-A0; vperp=wv-(wv@uhat)*uhat; vhat=vperp/np.linalg.norm(vperp)
def coords(A): d=A-A0; return float(d@uhat), float(d@vhat)
ax_a=coords(Aa); ax_r=coords(Ar); ax_0=(0.0,0.0)
# grid
amax=max(ax_a[0],ax_r[0]); amin=min(0,ax_a[0],ax_r[0])
bmax=max(ax_a[1],ax_r[1]); bmin=min(0,ax_a[1],ax_r[1])
pad_a=0.35*(amax-amin+1e-6); pad_b=0.35*(bmax-bmin+1e-6)
al=np.linspace(amin-pad_a,amax+pad_a,71); be=np.linspace(bmin-pad_b,bmax+pad_b,71)
AA,BB=np.meshgrid(al,be)
pts=A0[None,:]+AA.ravel()[:,None]*uhat[None,:]+BB.ravel()[:,None]*vhat[None,:]
Jgrid=J(pts).reshape(BB.shape)
# optimization paths projected
adam_path=np.array([coords(p.view(-1).numpy()) for p in adam["plans"]])
rlp_path =np.array([coords(p.view(-1).numpy()) for p in rlp["plans"]])
# 1D profiles
ts=np.linspace(0,1,101)
prof_rlp=J(np.stack([(1-t)*A0+t*Ar for t in ts])); prof_adam=J(np.stack([(1-t)*A0+t*Aa for t in ts]))
prof_a2r=J(np.stack([(1-t)*Aa+t*Ar for t in ts]))
np.savez("scripts/figures/tworoom_rlp/_adamland.npz",
    AA=AA,BB=BB,J=Jgrid,al=al,be=be,ax0=ax_0,axa=ax_a,axr=ax_r,
    adam_path=adam_path,rlp_path=rlp_path,ts=ts,
    prof_rlp=prof_rlp,prof_adam=prof_adam,prof_a2r=prof_a2r,
    adam_vals=np.array(adam["values"]),rlp_vals=np.array(rlp["values"]),
    origin=o,goal=g)
print("J range on slice:",Jgrid.min().round(2),Jgrid.max().round(2))
print("A0",np.round(ax_0,2),"J",J(A0[None])[0].round(2))
print("adam",np.round(ax_a,2),"J",J(Aa[None])[0].round(2))
print("rlp ",np.round(ax_r,2),"J",J(Ar[None])[0].round(2))
print("ridge on A0->rlp: max",prof_rlp.max().round(1),"at t=",round(float(ts[prof_rlp.argmax()]),2))
print("saved _adamland.npz")
