"""Cost-to-go heatmaps for TwoRoom (paper Fig-1 style). No prose headers."""
import sys, pickle; import os as _os; sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl
import style as S, heat as Hm

data=pickle.load(open("scripts/figures/tworoom_rlp/_figdata.pkl","rb"))
OUT="docs/figures/tworoom_rlp"; free=~data["blocked"]
def corr(a,b):
    m=free&np.isfinite(b); return np.corrcoef(a[m],b[m])[0,1]

def colorbar(fig,pos):
    cax=fig.add_axes(pos)
    mpl.colorbar.ColorbarBase(cax,cmap=Hm.VIRIDIS,orientation="vertical")
    cax.set_yticks([]); cax.tick_params(length=0)
    cax.text(0.5,1.04,"far",ha="center",va="bottom",transform=cax.transAxes,fontsize=9,color=S.INK)
    cax.text(0.5,-0.04,"near",ha="center",va="top",transform=cax.transAxes,fontsize=9,color=S.INK)
    for sp in cax.spines.values(): sp.set_edgecolor("#c9ced3")

def sublabel(ax,text,color):
    ax.text(0.5,-0.05,text,transform=ax.transAxes,ha="center",va="top",fontsize=12.5,color=color,style="italic")

def legend(fig,y=0.045):
    handles=[Line2D([0],[0],marker="*",ls="none",ms=13,mfc=S.GOLD,mec=S.GOLD_EDGE,label="goal"),
             Line2D([0],[0],marker="o",ls="none",ms=8,mfc=S.RED,mec="#7a1a12",label="origin"),
             Patch(fc=S.BARRIER,ec="none",label="barrier")]
    fig.legend(handles=[h for h in handles],loc="lower center",ncol=3,frameon=False,
               fontsize=10.5,bbox_to_anchor=(0.5,y),handletextpad=0.5,columnspacing=1.8)

# ---------- FIG 1: flagship (PLDM | LeWM | True) ----------
def flagship():
    ts=data["flagship"]; T=data["tasks"][ts]; o,g=T["origin"],T["goal"]
    fig=plt.figure(figsize=(13.4,4.0))
    gs=GridSpec(1,5,left=0.028,right=0.93,top=0.80,bottom=0.16,wspace=0.08)
    panels=[("PLDM","latent",T["latent"]["pldm"],S.ORANGE),("PLDM","learned",T["learned"][("pldm",0)],S.BLUE),
            ("LeWM","latent",T["latent"]["lejepa"],S.ORANGE),("LeWM","learned",T["learned"][("lejepa",0)],S.BLUE),
            ("True","geodesic",T["geodesic"],S.MUTE)]
    axes=[]
    for i,(grp,sub,f,c) in enumerate(panels):
        ax=fig.add_subplot(gs[0,i]); axes.append(ax); Hm.render_heat(ax,f,data,o,g); sublabel(ax,sub,c)
    def hdr(a0,a1,txt):
        x=(axes[a0].get_position().x0+axes[a1].get_position().x1)/2
        fig.text(x,0.855,txt,ha="center",fontsize=14.5,color=S.INK,weight="bold")
    hdr(0,1,"PLDM"); hdr(2,3,"LeWM"); hdr(4,4,"True distance")
    colorbar(fig,[0.945,0.16,0.016,0.64]); legend(fig,0.035)
    p=f"{OUT}/costtogo_main.png"; fig.savefig(p,dpi=150); fig.savefig(p.replace('.png','.svg')); plt.close(fig)
    print("wrote",p)

# ---------- FIG 2: task seeds x [latent, learned, geodesic] ----------
def seeds_grid():
    seeds=data["task_seeds"]
    cols=[("latent (LeWM)","lejepa_latent",S.ORANGE),("learned value (LeWM)","lejepa_learned",S.BLUE),
          ("true geodesic","geo",S.MUTE)]
    nr,nc=len(seeds),len(cols)
    fig=plt.figure(figsize=(3.05*nc+0.9,2.9*nr+0.4))
    gs=GridSpec(nr,nc,left=0.085,right=0.895,top=0.93,bottom=0.03,wspace=0.06,hspace=0.09)
    ax0=[]
    for r,ts in enumerate(seeds):
        T=data["tasks"][ts]; o,g=T["origin"],T["goal"]
        fields={"lejepa_latent":T["latent"]["lejepa"],"lejepa_learned":T["learned"][("lejepa",0)],"geo":T["geodesic"]}
        for c,(title,key,col) in enumerate(cols):
            ax=fig.add_subplot(gs[r,c]);
            if c==0: ax0.append(ax)
            Hm.render_heat(ax,fields[key],data,o,g)
            if r==0: ax.set_title(title,fontsize=12,color=col,style="italic",pad=8)
    for r,ts in enumerate(seeds):
        pos=ax0[r].get_position()
        fig.text(0.052,(pos.y0+pos.y1)/2,f"task seed {ts}",rotation=90,ha="center",va="center",
                 fontsize=12.5,color=S.INK,weight="bold")
    mid=ax0[len(seeds)//2].get_position()   # scale = height of one panel (middle row)
    colorbar(fig,[0.905,mid.y0,0.013,mid.height])
    p=f"{OUT}/costtogo_task_seeds.png"; fig.savefig(p,dpi=150); fig.savefig(p.replace('.png','.svg')); plt.close(fig)
    print("wrote",p)
    for ts in seeds:
        T=data["tasks"][ts]
        print(f"  seed {ts}: learned~geo {corr(T['learned'][('lejepa',0)],T['geodesic']):.3f}  latent~geo {corr(T['latent']['lejepa'],T['geodesic']):.3f}")

# ---------- FIG 3: actor seeds (fixed task) ----------
def actor_sweep():
    ts=data["flagship"]; T=data["tasks"][ts]; o,g=T["origin"],T["goal"]
    cols=[("true geodesic",T["geodesic"],S.MUTE)]+[(f"learned · actor seed {s}",T["learned"][("lejepa",s)],S.BLUE) for s in data["actor_seeds"]]
    nc=len(cols); fig=plt.figure(figsize=(3.05*nc+0.7,3.5))
    gs=GridSpec(1,nc,left=0.02,right=0.9,top=0.86,bottom=0.05,wspace=0.06)
    for c,(title,f,col) in enumerate(cols):
        ax=fig.add_subplot(gs[0,c]); Hm.render_heat(ax,f,data,o,g)
        ax.set_title(title,fontsize=12,color=col,style="italic",pad=8)
    colorbar(fig,[0.915,0.05,0.014,0.71])
    p=f"{OUT}/costtogo_actor_seeds.png"; fig.savefig(p,dpi=150); fig.savefig(p.replace('.png','.svg')); plt.close(fig)
    print("wrote",p)
    for s in data["actor_seeds"]:
        print(f"  actor {s}: learned~geo {corr(T['learned'][('lejepa',s)],T['geodesic']):.3f}")

flagship(); seeds_grid(); actor_sweep()
