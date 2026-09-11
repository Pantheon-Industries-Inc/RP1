"""Analysis: the plan-space value landscape where Adam stalls (seed 43). Viridis."""
import sys; import os as _os; sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import style as S
d=np.load("scripts/figures/tworoom_rlp/_adamland.npz")
OUT="docs/figures/tworoom_rlp"
BLUE=S.BLUE; ORANGE=S.ORANGE; INK=S.INK; MUTE=S.MUTE
J=d["J"]; al=d["al"]; be=d["be"]; ax0=d["ax0"]; axa=d["axa"]; axr=d["axr"]
adam_path=d["adam_path"]; rlp_path=d["rlp_path"]

fig=plt.figure(figsize=(13.6,4.7))
gs=GridSpec(1,3,width_ratios=[1.35,1,1],left=0.045,right=0.965,top=0.80,bottom=0.135,wspace=0.32)

# --- Panel A: 2D value landscape slice ---
axL=fig.add_subplot(gs[0,0])
ext=[al.min(),al.max(),be.min(),be.max()]
vmax=float(np.percentile(J,97))
im=axL.imshow(J,origin="lower",extent=ext,cmap="viridis",vmin=float(J.min()),vmax=vmax,
              aspect="auto",interpolation="bilinear")
cs=axL.contour(d["AA"],d["BB"],J,levels=np.linspace(J.min(),vmax,10),colors="white",
               linewidths=0.5,alpha=0.35)
# optimization paths (white halo so they read on the landscape)
for col,pth in [(BLUE,adam_path),(ORANGE,rlp_path)]:
    axL.plot(pth[:,0],pth[:,1],"-",color="white",lw=3.6,zorder=3.8,alpha=0.9,solid_capstyle="round")
    axL.plot(pth[:,0],pth[:,1],"-",color=col,lw=2.1,zorder=4,alpha=0.98,solid_capstyle="round")
    axL.plot(pth[:,0],pth[:,1],"o",color=col,ms=2.8,mec="white",mew=0.4,zorder=4.1)
# key points
axL.plot(*ax0,marker="o",ms=11,mfc="white",mec=INK,mew=1.6,zorder=6)
axL.plot(*axa,marker="X",ms=13,mfc=BLUE,mec="white",mew=1.4,zorder=6)
axL.plot(*axr,marker="*",ms=19,mfc=S.GOLD,mec=S.GOLD_EDGE,mew=1.0,zorder=6)
_bb=dict(boxstyle="round,pad=0.22",fc="white",ec="none",alpha=0.72)
axL.annotate("A₀ (init)  V=35",ax0,textcoords="offset points",xytext=(11,8),fontsize=9.5,color=INK,weight="bold",bbox=_bb)
axL.annotate("Adam stalls\nV=12.3",axa,textcoords="offset points",xytext=(-10,15),fontsize=9.5,
             color=BLUE,weight="bold",ha="right",bbox=_bb)
axL.annotate("RLP fθ\nV=0.3",axr,textcoords="offset points",xytext=(11,3),fontsize=9.5,color="#8a5a12",weight="bold",bbox=_bb)
axL.set_xlabel("plan-space direction → Adam",fontsize=10,color=BLUE)
axL.set_ylabel("direction → RLP solution",fontsize=10,color="#8a5a12")
axL.set_title("value landscape  J(A) = V(z_T(A), z_g)",fontsize=12,color=INK,weight="bold",pad=8)
axL.tick_params(labelsize=8)
cb=fig.colorbar(im,ax=axL,fraction=0.046,pad=0.03); cb.set_label("plan value",fontsize=9); cb.ax.tick_params(labelsize=8)

# --- Panel B: 1D value profiles ---
axM=fig.add_subplot(gs[0,1]); ts=d["ts"]
axM.plot(ts,d["prof_rlp"],"-",color=INK,lw=2.2,label="A₀ → RLP solution")
axM.plot(ts,d["prof_adam"],"-",color=BLUE,lw=2.0,label="A₀ → Adam stall")
axM.plot(ts,d["prof_a2r"],"--",color=ORANGE,lw=2.0,label="Adam → RLP")
imax=int(np.argmax(d["prof_rlp"]))
axM.annotate("ridge",(ts[imax],d["prof_rlp"][imax]),textcoords="offset points",xytext=(-6,-14),
             fontsize=9.5,color=INK,weight="bold",ha="right")
axM.set_xlabel("interpolation  t",fontsize=10,color=INK)
axM.set_ylabel("plan value V",fontsize=10,color=INK)
axM.set_title("a barrier blocks the straight path",fontsize=12,color=INK,weight="bold",pad=8)
axM.legend(fontsize=8.5,frameon=False,loc="center left")
axM.grid(alpha=0.18); axM.tick_params(labelsize=8)
for sp in ["top","right"]: axM.spines[sp].set_visible(False)

# --- Panel C: convergence ---
axR=fig.add_subplot(gs[0,2])
av=d["adam_vals"]; rv=d["rlp_vals"]
axR.plot(np.arange(len(av)),av,"-o",color=BLUE,lw=1.8,ms=2.5,label="Adam (30 it)")
axR.plot(np.arange(len(rv)),rv,"-o",color=ORANGE,lw=1.8,ms=3.5,label="RLP fθ (8 it)")
axR.axhline(0,color=MUTE,lw=0.8,ls=":")
axR.set_xlabel("optimizer iteration",fontsize=10,color=INK)
axR.set_ylabel("plan value V",fontsize=10,color=INK)
axR.set_title("Adam oscillates and plateaus",fontsize=12,color=INK,weight="bold",pad=8)
axR.legend(fontsize=8.5,frameon=False); axR.grid(alpha=0.18); axR.tick_params(labelsize=8)
for sp in ["top","right"]: axR.spines[sp].set_visible(False)

fig.suptitle("Why Adam stalls on task seed 43 — the plan objective V(z_T(A), z_g) is non-convex",
             x=0.045,ha="left",y=0.95,fontsize=13.5,color=INK,weight="bold")
fig.text(0.045,0.885,"same value function, same task; Adam descends into a shallow basin and oscillates, "
         "while the learned fθ reaches the deep goal basin across the ridge",
         ha="left",fontsize=9.5,color=MUTE)
p=f"{OUT}/adam_value_landscape.png"; fig.savefig(p,dpi=150); fig.savefig(p.replace('.png','.svg')); plt.close(fig)
print("wrote",p)
