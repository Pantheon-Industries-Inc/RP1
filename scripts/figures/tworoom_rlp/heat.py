"""Heatmap renderer: smooth cost-to-go field + clean vector barrier (paper style)."""
import sys; import os as _os; sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import style as S

# Original cost-to-go scale: viridis (near dark -> far bright).
CMAP = mpl.cm.viridis.copy()
VIRIDIS = CMAP  # name kept for callers

def _fill_blocked(field2d, blocked2d):
    """Nearest-free fill so imshow has no holes; barrier drawn on top."""
    from scipy.ndimage import distance_transform_edt
    f = field2d.copy()
    m = blocked2d
    if m.any():
        idx = distance_transform_edt(m, return_distances=False, return_indices=True)
        f = f[tuple(idx)]
    return f

def norm01(field, free):
    lo, hi = np.nanmin(field[free]), np.nanmax(field[free])
    return (field-lo)/max(1e-9,(hi-lo))

def render_heat(ax, field1d, data, origin, goal, goal_style="star", show_origin=True):
    ny,nx = data["ny"], data["nx"]
    XX,YY,blocked = data["XX"],data["YY"],data["blocked"]
    free = ~blocked
    v = norm01(field1d.astype(float), free)
    F = v.reshape(ny,nx); B = blocked.reshape(ny,nx)
    F = _fill_blocked(F, B)
    xs, ys = XX[0,:], YY[:,0]
    ext=[xs.min(), xs.max(), ys.max(), ys.min()]
    ax.imshow(F, origin="upper", cmap=VIRIDIS, extent=ext, vmin=0, vmax=1,
              interpolation="bilinear", zorder=0)
    _barrier(ax, data)
    S.room_axes(ax)
    if goal_style=="star":
        ax.plot(*goal, marker="*", ms=15, color=S.GOLD, mec=S.GOLD_EDGE, mew=0.7, zorder=6)
    if show_origin:
        ax.plot(*origin, marker="o", ms=8, color=S.RED, mec="#7a1a12", mew=0.7, zorder=6)

def _barrier(ax, data, color=S.BARRIER):
    b=data["border"]; wx=data["wall_x"]; dh=data["door_half"]; dy=data["door_y"]; half=5.0
    bt=4.0
    for (x,y,w,h) in [(b-bt,b-bt,S.IMG-2*b+2*bt,bt),(b-bt,S.IMG-b,S.IMG-2*b+2*bt,bt),
                      (b-bt,b-bt,bt,S.IMG-2*b+2*bt),(S.IMG-b,b-bt,bt,S.IMG-2*b+2*bt)]:
        ax.add_patch(Rectangle((x,y),w,h, fc=color, ec="none", zorder=4))
    for (y0,y1) in [(b-bt, dy-dh),(dy+dh, S.IMG-b+bt)]:
        if y1>y0:
            ax.add_patch(Rectangle((wx-half,y0),2*half,y1-y0, fc=color, ec="none", zorder=4))
