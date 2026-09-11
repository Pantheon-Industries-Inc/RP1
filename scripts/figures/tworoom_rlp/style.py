"""Shared palette + TwoRoom drawing helpers, matched to the paper's tones."""
import numpy as np
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection

# ---- palette (from the paper tex + shared mockups) ----
BG        = "#f6f8fa"   # near-white room background
WALL      = "#c2cdd5"   # light slate wall
WALL_EDGE = "#9aa7b1"
ORANGE    = "#c0491e"   # rust accent: plan + agent + "latent"
ORANGE_LT = "#e6a58c"
BLUE      = "#2f6f9f"   # steel blue: "learned"
TEAL      = "#3d6b7a"   # goal ring
GOLD      = "#f5c518"   # goal star
GOLD_EDGE = "#2b2b2b"
RED       = "#e2402f"   # origin dot
BARRIER   = "#828d95"   # heatmap barrier gray
INK       = "#2b2f33"
MUTE      = "#6b747c"

mpl.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "axes.edgecolor": "#d0d6db",
    "svg.fonttype": "none",
})

IMG = 224

def room_axes(ax):
    ax.set_xlim(0, IMG); ax.set_ylim(IMG, 0)  # y down (image coords)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def draw_room(ax, geom, bg=BG, wall=WALL, wedge=WALL_EDGE, lw=1.0, r=0.02):
    """Clean vector room: bg panel, border frame, vertical wall with door gap."""
    b = geom["border"]; wx = geom["wall_x"]; dh = geom["door_half"]; dy = geom["door_y"]
    half = 5.0  # wall half-thickness (WALL_WIDTH_DEFAULT//2)
    # background panel (rounded)
    ax.add_patch(FancyBboxPatch((b, b), IMG-2*b, IMG-2*b,
        boxstyle=f"round,pad=0,rounding_size={r*IMG}", fc=bg, ec="none", zorder=0))
    # border frame (4 bars)
    bt = 4.0
    for (x,y,w,h) in [(b-bt,b-bt,IMG-2*b+2*bt,bt),(b-bt,IMG-b,IMG-2*b+2*bt,bt),
                      (b-bt,b-bt,bt,IMG-2*b+2*bt),(IMG-b,b-bt,bt,IMG-2*b+2*bt)]:
        ax.add_patch(Rectangle((x,y),w,h, fc=wall, ec=wedge, lw=0.6, zorder=1,
                     joinstyle="round"))
    # vertical wall, split around the door
    segs = [(b-bt, dy-dh), (dy+dh, IMG-b+bt)]  # y ranges of wall (above/below door)
    for (y0,y1) in segs:
        if y1<=y0: continue
        ax.add_patch(FancyBboxPatch((wx-half, y0), 2*half, y1-y0,
            boxstyle="round,pad=0,rounding_size=2.5", fc=wall, ec=wedge, lw=0.6, zorder=1))

def draw_goal(ax, xy, style="ring", s=1.0, color=TEAL, z=2.0):
    """Shaded goal region, drawn BEHIND the plan (low z) so the path reads on top."""
    x,y = float(xy[0]), float(xy[1])
    if style=="ring":
        ax.add_patch(Circle((x,y), 16*s, fc=color, ec="none", alpha=0.16, zorder=z))
        ax.add_patch(Circle((x,y), 16*s, fill=False, ec=color, lw=1.3, ls=(0,(4,3)), alpha=0.55, zorder=z+0.1))
        ax.add_patch(Circle((x,y), 6.5*s, fc=color, ec="none", alpha=0.32, zorder=z+0.2))
        ax.add_patch(Circle((x,y), 6.5*s, fill=False, ec=color, lw=1.8, alpha=0.7, zorder=z+0.3))
    else:  # star
        ax.plot(x,y, marker="*", ms=17*s, color=GOLD, mec=GOLD_EDGE, mew=0.8, zorder=7)

def arrow_at_tip(ax, xy, color=ORANGE, lw=2.5, back_px=9.0, z=5.2):
    """Arrowhead placed exactly at the trajectory tip, direction from a stable lookback."""
    import numpy as np
    xy=np.asarray(xy,float); tip=xy[-1]; back=xy[0]
    for q in range(len(xy)-2,-1,-1):
        if float(np.hypot(*(xy[q]-tip)))>=back_px: back=xy[q]; break
    if float(np.hypot(*(tip-back)))<1e-6: return
    ax.add_patch(FancyArrowPatch(back, tip, arrowstyle="-|>", mutation_scale=13,
                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z))

def draw_agent(ax, xy, color=ORANGE, s=1.0):
    x,y = float(xy[0]), float(xy[1])
    ax.add_patch(Circle((x,y), 4.6*s, fc=color, ec="white", lw=1.0, zorder=8))

def draw_path(ax, xy, color=ORANGE, lw=2.4, alpha=1.0, dots=True, arrow=True, ms=4.2, z=5):
    xy = np.asarray(xy, float)
    ax.plot(xy[:,0], xy[:,1], "-", color=color, lw=lw, alpha=alpha, solid_capstyle="round",
            solid_joinstyle="round", zorder=z)
    if dots:
        ax.plot(xy[1:,0], xy[1:,1], "o", color=color, ms=ms, alpha=alpha, mec="white",
                mew=0.6, zorder=z+0.1)
    if arrow and len(xy)>=2:
        a,b = xy[-2], xy[-1]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                     color=color, lw=lw, alpha=alpha, zorder=z+0.2))
