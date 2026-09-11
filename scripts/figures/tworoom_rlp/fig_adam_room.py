"""The value landscape V(z(x,y),z_g) in the TwoRoom, with executed Adam vs RLP plans (seed 43). Viridis."""
import sys, pickle; import os as _os; sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import style as S, heat as Hm
from scipy.ndimage import distance_transform_edt

data=pickle.load(open("scripts/figures/tworoom_rlp/_figdata.pkl","rb"))
OUT="docs/twviz" if False else "docs/figures/tworoom_rlp"
SEED=43
T=data["tasks"][SEED]; o=T["origin"]; g=T["goal"]
XX,YY,blocked=data["XX"],data["YY"],data["blocked"]; ny,nx=data["ny"],data["nx"]
field=T["learned"][("lejepa",0)].astype(float)     # raw V over positions
adam_xy=data["methods"]["adam"][SEED]["paths"][-1]
rlp_xy =data["methods"]["rlp"][SEED]["paths"][-1]

def field_img():
    F=field.reshape(ny,nx).copy(); B=blocked.reshape(ny,nx)
    if B.any():
        idx=distance_transform_edt(B,return_distances=False,return_indices=True); F=F[tuple(idx)]
    return F
F=field_img()
vmax=float(np.percentile(field[~blocked],98)); vmin=float(field[~blocked].min())
xs,ys=XX[0,:],YY[:,0]; ext=[xs.min(),xs.max(),ys.max(),ys.min()]

def base(ax):
    im=ax.imshow(F,origin="upper",cmap="viridis",extent=ext,vmin=vmin,vmax=vmax,
                 interpolation="bilinear",zorder=0)
    Hm._barrier(ax,data); S.room_axes(ax)
    return im

def traj(ax,xy,color,label,end_marker):
    ax.plot(xy[:,0],xy[:,1],"-",color="white",lw=4.2,zorder=4,alpha=0.9,solid_capstyle="round")
    ax.plot(xy[:,0],xy[:,1],"-",color=color,lw=2.6,zorder=4.1,alpha=0.98,solid_capstyle="round")
    S.arrow_at_tip(ax,xy,color=color,lw=2.6,z=4.3)
    ax.plot(*o,marker="o",ms=8,mfc=S.RED,mec="#7a1a12",mew=0.8,zorder=6)
    ax.plot(*g,marker="*",ms=17,mfc=S.GOLD,mec=S.GOLD_EDGE,mew=0.8,zorder=6)

fig=plt.figure(figsize=(10.6,4.7))
gs=GridSpec(1,2,left=0.02,right=0.9,top=0.80,bottom=0.06,wspace=0.06)
axA=fig.add_subplot(gs[0,0]); base(axA); traj(axA,adam_xy,S.BLUE,"Adam",True)
end=adam_xy[-1]; axA.annotate("Adam plan ends here\n(still V≈12 from goal)",(end[0],end[1]),
    textcoords="offset points",xytext=(2,26),ha="center",fontsize=9.5,color=S.BLUE,weight="bold",
    bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="none",alpha=0.82))
axA.set_title("Adam — 30 steps",fontsize=12.5,color=S.BLUE,weight="bold",pad=6)
axR=fig.add_subplot(gs[0,1]); im=base(axR); traj(axR,rlp_xy,S.ORANGE,"RLP",True)
axR.annotate("RLP plan reaches\nthe goal (V≈0.3)",(g[0],g[1]),
    textcoords="offset points",xytext=(16,26),fontsize=9.5,color="#8a3a12",weight="bold",
    bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="none",alpha=0.8))
axR.set_title("RLP fθ — 8 steps",fontsize=12.5,color="#8a3a12",weight="bold",pad=6)

cax=fig.add_axes([0.915,0.10,0.016,0.62]); cb=fig.colorbar(im,cax=cax)
cb.set_label("value  V(·, z_g)  =  cost-to-go",fontsize=9.5); cb.ax.tick_params(labelsize=8)
cax.text(0.5,1.03,"far",ha="center",va="bottom",transform=cax.transAxes,fontsize=8.5,color=S.INK)
cax.text(0.5,-0.03,"near",ha="center",va="top",transform=cax.transAxes,fontsize=8.5,color=S.INK)

fig.suptitle("The value landscape over the room (task seed 43) — where each optimizer's plan ends",
             x=0.02,ha="left",y=0.945,fontsize=13,color=S.INK,weight="bold")
fig.text(0.02,0.875,"cost-to-go V(·, z_g) from every position · same critic, plans executed through true dynamics"
         "   ·   ● origin   ★ goal   ▬ barrier",
         ha="left",fontsize=9,color=S.MUTE)
p=f"{OUT}/adam_value_room.png"; fig.savefig(p,dpi=150); fig.savefig(p.replace('.png','.svg')); plt.close(fig)
print("wrote",p,"| adam end V-field=",round(float(field[np.argmin((XX.ravel()-end[0])**2+(YY.ravel()-end[1])**2)]),1))
