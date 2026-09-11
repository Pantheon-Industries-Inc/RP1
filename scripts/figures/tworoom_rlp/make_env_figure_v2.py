"""Polished 3-env illustration: TwoRoom in the RLP vector style (rounded border =
room border); Reacher & Cube re-rendered with natural colours, brightened.
Only TwoRoom marks the target; Reacher shows a faded goal-pose ghost, Cube none."""
import os; os.environ.setdefault("MUJOCO_GL","glfw")
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageEnhance
import mujoco, gymnasium as gym, stable_worldmodel  # noqa
import style as S
import importlib.util
spec=importlib.util.spec_from_file_location("mef", os.path.abspath("scripts/make_env_figure.py"))
mef=importlib.util.module_from_spec(spec); spec.loader.exec_module(mef)

OUT=Path("docs/figures/tworoom_rlp"); OUT.mkdir(parents=True, exist_ok=True)
BG="#f6f8fa"; FRAME_EC=S.WALL; ROUND=0.055

def enhance(frame, bright=1.10, contrast=1.06, color=1.04, gamma=0.95):
    a=np.asarray(frame).astype(np.float32)/255.0
    a=np.clip(a**gamma,0,1); im=Image.fromarray((a*255).astype("uint8"))
    im=ImageEnhance.Brightness(im).enhance(bright)
    im=ImageEnhance.Contrast(im).enhance(contrast)
    im=ImageEnhance.Color(im).enhance(color)
    return np.asarray(im)

def rounded(ax, W, H):
    bb=FancyBboxPatch((0,0),W,H,boxstyle=f"round,pad=0,rounding_size={ROUND*W}",
                      transform=ax.transData, fc="none", ec="none")
    ax.add_patch(bb); return bb

def frame_border(ax, W, H, x0=0, y0=0):
    ax.add_patch(FancyBboxPatch((x0,y0),W,H,boxstyle=f"round,pad=0,rounding_size={ROUND*W}",
                 transform=ax.transData, fc="none", ec=FRAME_EC, lw=1.8, zorder=9))

def panel_img(ax, img):
    H,W=img.shape[:2]
    im=ax.imshow(img, extent=[0,W,H,0], zorder=1); im.set_clip_path(rounded(ax,W,H))
    frame_border(ax,W,H)
    ax.set_xlim(-2,W+2); ax.set_ylim(H+2,-2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def panel_tworoom(ax, seed=5):
    env=gym.make("swm/TwoRoom-v1"); env.reset(seed=seed); u=env.unwrapped
    o=u.agent_position.numpy().copy(); g=u.target_position.numpy().copy()
    bs=int(u.BORDER_SIZE); wx=float(u.WALL_CENTER); dh=float(u.door_sizes[0]); dy=float(u.door_positions[0])
    env.close()
    half=5.0; lo=bs-2.0; hi=S.IMG-bs+2.0; r=ROUND*(hi-lo)
    ax.set_xlim(lo-1,hi+1); ax.set_ylim(hi+1,lo-1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    # rounded border == the room's outer wall (no gap, no double frame)
    ax.add_patch(FancyBboxPatch((lo,lo),hi-lo,hi-lo,boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=BG, ec=S.WALL, lw=3.4, zorder=1, joinstyle="round"))
    # central wall + door gap
    for (y0,y1) in [(lo, dy-dh),(dy+dh, hi)]:
        if y1>y0:
            ax.add_patch(FancyBboxPatch((wx-half,y0),2*half,y1-y0,boxstyle="round,pad=0,rounding_size=2.5",
                         fc=S.WALL, ec=S.WALL_EDGE, lw=0.6, zorder=2))
    S.draw_goal(ax, g, style="star", s=1.4, z=6)   # goal = star (no rings)
    S.draw_agent(ax, o)

# --------------------------------------------------------------- cube
def _render_cube_once(obj_xy, goal_xy, seed=2):
    """Single-cube scene: object_0 at obj_xy, goal cube (object_target_0 mocap) at
    goal_xy. Put goal_xy far off-screen to omit the goal cube entirely."""
    env=gym.make("swm/OGBCube-v0", render_mode="rgb_array", ob_type="pixels", width=mef.RES, height=mef.RES)
    env.reset(seed=seed, options={"variation_values": {"agent.ee_start_position": mef.EE_START}})
    unwrapped=env.unwrapped; model=unwrapped._model; data=unwrapped._data
    for name,rgb in mef.ARM_MATERIALS.items():
        mat=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if mat>=0: model.mat_rgba[mat,:3]=rgb; model.mat_rgba[mat,3]=1.0
    model.vis.headlight.ambient[:]=0.5; model.vis.headlight.diffuse[:]=0.9
    model.light_castshadow[:]=0; model.light_diffuse[:]=0.85
    body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_0")
    adr=model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[adr:adr+3]=[obj_xy[0], obj_xy[1], 0.02]
    data.mocap_pos[unwrapped._cube_target_mocap_ids[unwrapped._target_block]]=list(goal_xy)+[0.02]
    mujoco.mj_forward(model, data)
    frame=np.asarray(unwrapped.render(camera=mef.VIDEO_CAMERA)); env.close()
    return frame

def render_cube(seed=2, goal_opacity=0.20, obj_xy=(0.42, -0.16), goal_xy=(0.42, 0.16)):
    """Solid object cube + a heavily-shaded (faint) goal cube at a fixed distant
    target. Composite: render with the goal cube in place and with it moved
    off-screen, then blend the goal region back at ``goal_opacity`` (lower = fainter)."""
    full=_render_cube_once(obj_xy, goal_xy, seed)          # object + solid goal
    base=_render_cube_once(obj_xy, (5.0, 5.0), seed)       # goal cube off-screen
    mask=(np.abs(full.astype(int)-base.astype(int)).sum(-1) > 22)[...,None]
    out=np.where(mask, (1.0-goal_opacity)*base + goal_opacity*full, base)
    return out.astype(np.uint8)

# --------------------------------------------------------------- render
print("rendering reacher..."); R=mef.make_reacher()
arm=R["goal_frame"][...,0].astype(int) > R["goal_frame"][...,2].astype(int)+20
rea=enhance(mef.blend(R["frame"], R["goal_frame"], arm, 0.28), bright=1.11, contrast=1.06, color=1.05, gamma=0.95)
print("rendering cube..."); cub=enhance(render_cube(), bright=1.16, contrast=1.05, color=1.0, gamma=0.93)

fig,axes=plt.subplots(1,3, figsize=(10.2,3.7)); fig.patch.set_facecolor("white")
for ax in axes: ax.set_facecolor("white")
panel_img(axes[0], rea); panel_img(axes[1], cub); panel_tworoom(axes[2], seed=5)
for ax,lab in zip(axes,["Reacher","OGBench Cube","TwoRoom"]):
    ax.set_title(lab, fontsize=14, family="serif", color=S.INK, y=-0.11)
fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.08,wspace=0.07)
p=OUT/"env_panel.png"; fig.savefig(p,dpi=200,facecolor="white"); fig.savefig(str(p).replace('.png','.svg'))
plt.close(fig); print("wrote",p)
