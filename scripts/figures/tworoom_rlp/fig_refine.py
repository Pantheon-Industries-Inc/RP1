"""Plan-refinement time-series filmstrips (RLP fθ / Adam / CEM). Executed paths, no headers."""
import sys, pickle; import os as _os; sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style as S

data=pickle.load(open("scripts/figures/tworoom_rlp/_figdata.pkl","rb"))
geom=dict(border=data["border"],wall_x=data["wall_x"],door_half=data["door_half"],door_y=data["door_y"])
OUT="docs/figures/tworoom_rlp"

# per-method display iterations (rlp K=8; adam/cem K=30) and output stem
METHODS={
    "rlp":  dict(iters=[0,1,2,3,4,6,8],      stem="refine_seed"),
    "adam": dict(iters=[0,2,4,8,14,20,30],   stem="refine_adam_seed"),
    "cem":  dict(iters=[0,1,2,4,8,15,30],    stem="refine_cem_seed"),
}

def filmstrip(R, iters, path):
    paths=R["paths"]; origin=R["origin"]; goal=R["goal"]; n=len(iters)
    fig,axes=plt.subplots(1,n, figsize=(2.15*n,2.35))
    fig.subplots_adjust(left=0.008,right=0.992,top=0.90,bottom=0.02,wspace=0.09)
    for j,(ax,k) in enumerate(zip(axes,iters)):
        k=min(k,len(paths)-1)
        S.room_axes(ax); S.draw_room(ax, geom)
        if j>0:
            gp=paths[min(iters[j-1],len(paths)-1)]
            ax.plot(gp[:,0],gp[:,1],"-",color=S.ORANGE,lw=1.5,alpha=0.15,zorder=3,solid_capstyle="round")
        S.draw_goal(ax, goal, style="ring", z=2.0)
        xy=paths[k]
        ax.plot(xy[:,0],xy[:,1],"-",color=S.ORANGE,lw=2.5,alpha=0.98,zorder=5,
                solid_capstyle="round",solid_joinstyle="round")
        ax.plot(xy[::5,0][1:],xy[::5,1][1:],"o",color=S.ORANGE,ms=3.4,mec="white",mew=0.5,zorder=5.1)
        if len(xy)>=2:
            S.arrow_at_tip(ax, xy, color=S.ORANGE, lw=2.5)
        S.draw_agent(ax, origin)
        ax.set_title(f"iteration {k}", fontsize=11, color=S.INK, pad=6)
    fig.savefig(path,dpi=150); fig.savefig(path.replace('.png','.svg')); plt.close(fig)

for method,cfg in METHODS.items():
    table=data["methods"][method]
    for rs in data["refine_seeds"]:
        R=table[rs]; p=f"{OUT}/{cfg['stem']}{rs}.png"
        filmstrip(R, cfg["iters"], p)
        print(f"wrote {p} | executed end {float(R['end_dist']):.1f}px")
