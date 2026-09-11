"""One composite per task: 3xN grid (rows Adam/CEM/RP1, columns = shared refinement
iteration checkpoints). Iteration labels once on top, planner names once on the left."""
import sys, pickle; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style as S

data=pickle.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"_figdata.pkl"),"rb"))
geom=dict(border=data["border"],wall_x=data["wall_x"],door_half=data["door_half"],door_y=data["door_y"])
OUT="docs/figures/tworoom_rlp"

ROWS=[("Adam","adam"),("CEM","cem"),("RP1","rlp")]         # top -> bottom
COLS=[0,1,2,4,8,"final"]                                    # shared checkpoints
LABELS=[r"$k=0$",r"$k=1$",r"$k=2$",r"$k=4$",r"$k=8$","final"]
TASKS=[(1,42),(2,43),(3,44)]

def col_index(paths, c): return (len(paths)-1) if c=="final" else c

def panel(ax, R, k):
    S.room_axes(ax); S.draw_room(ax, geom)
    S.draw_goal(ax, R["goal"], style="ring", z=6)
    xy=R["paths"][k]
    ax.plot(xy[:,0], xy[:,1], "-", color=S.ORANGE, lw=1.9, alpha=0.98, zorder=5, solid_capstyle="round")
    if len(xy)>=2: S.arrow_at_tip(ax, xy, color=S.ORANGE, lw=1.9, back_px=8.0)
    S.draw_agent(ax, R["origin"])

for tno, seed in TASKS:
    fig,axes=plt.subplots(len(ROWS), len(COLS), figsize=(len(COLS)*1.62, len(ROWS)*1.62+0.15),
                          gridspec_kw=dict(wspace=0.05, hspace=0.06))
    for r,(name,key) in enumerate(ROWS):
        R=data["methods"][key][seed]; paths=R["paths"]
        for c,col in enumerate(COLS):
            ax=axes[r,c]; panel(ax, R, col_index(paths,col))
            if r==0: ax.set_title(LABELS[c], fontsize=12, color=S.INK, pad=5)
        axes[r,0].set_ylabel(name, fontsize=13, color=S.INK, rotation=90, labelpad=6)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.93, bottom=0.01)
    p=f"{OUT}/tworoom_refine_task{tno}.png"
    fig.savefig(p, dpi=170); fig.savefig(p.replace('.png','.svg')); plt.close(fig)
    print(f"wrote {p}  (seed {seed})  end-dist adam/cem/rp1 = "
          f"{data['methods']['adam'][seed]['end_dist']:.0f}/{data['methods']['cem'][seed]['end_dist']:.0f}/{data['methods']['rlp'][seed]['end_dist']:.0f} px")
