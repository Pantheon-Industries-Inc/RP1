"""Near-goal resolution probe for PushT critics.

For each eval task (draw rule = the evaluator's), take the GOAL configuration
from the logged state, perturb it radially -- block-only, agent-only and joint
translation 0..60 px, block rotation 0..40 deg -- render from state, encode
with the frozen WM and read every critic's V(perturbed, goal) plus latent L2.
Reports, per critic and perturbation type: the mean curve V(r) - V(0), the
relative slope inside 30 px and the AUC separating "inside the 20 px / 20 deg
success tolerance" from "near miss" (20-40 px / 20-40 deg). A sharp critic has a
steep curve inside 30 px and AUC -> 1.

Env: H5, WM, OUT, DRAW (42), NT (50), CRITICS ("label=<value dir> ..."; default
base_ac / base_td from $ACTOR/train_checkpoints), GOAL_SRC (render|jpeg).
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import gymnasium as gym  # noqa: E402
import stable_worldmodel  # noqa: E402,F401
from PIL import Image  # noqa: E402

H5, WMD, OUT = os.environ["H5"], Path(os.environ["WM"]), Path(os.environ["OUT"])
DRAW = int(os.environ.get("DRAW", "42")); NT = int(os.environ.get("NT", "50")); OFF = 25
CRITICS = os.environ.get("CRITICS", "").split()
if not CRITICS:
    A = os.environ["ACTOR"]
    CRITICS = [f"base_ac={A}/train_checkpoints/value_ac", f"base_td={A}/train_checkpoints/value_td"]
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
RADII = list(range(0, 65, 5))
ANGLES = [0, 5, 10, 15, 20, 25, 30, 40]


def to_frame(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return np.asarray(Image.open(io.BytesIO(bytes(x))).convert("RGB"))
    x = np.asarray(x)
    return x.transpose(1, 2, 0) if (x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3) else x


with h5py.File(H5, "r") as h:
    epi = np.asarray(h["episode_idx"][:]).reshape(-1); stp = np.asarray(h["step_idx"][:]).reshape(-1)
    lo, hi, n = 16000, 18685, 50
    ep_ids = np.unique(epi); ep_ids = ep_ids[(ep_ids >= lo) & (ep_ids < hi)]
    maxlen = {int(e): int(stp[epi == e].max()) for e in ep_ids}
    keep = np.isin(epi, ep_ids); mx = np.zeros(len(epi), dtype=np.int64); mx[keep] = np.array([maxlen[int(e)] for e in epi[keep]])
    valid = np.nonzero(keep & (stp <= mx - OFF))[0]
    rows = np.sort(np.random.default_rng(DRAW).choice(valid, size=n, replace=False))[:NT]
    states = np.asarray(h["state"][:], dtype=np.float64); s_goal = states[rows + OFF]
    goal_jpeg = [to_frame(h["pixels"][int(r) + OFF]) for r in rows]

from rlp.core.value import load_metric  # noqa: E402
from rlp.core.world_model import load_pretrained  # noqa: E402
import stable_pretraining as spt  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

wm = load_pretrained(str(WMD)).to(dev).eval(); wm.requires_grad_(False)
stats = spt.data.dataset_stats.ImageNet
tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=stats["mean"], std=stats["std"]), T.Resize(224)])


@torch.no_grad()
def encode(frames, bs=128):
    zs = []
    for i in range(0, len(frames), bs):
        px = torch.stack([tf(f) for f in frames[i:i + bs]]).to(dev).unsqueeze(1)
        zs.append(wm.encode({"pixels": px})["emb"][:, 0].float())
    return torch.cat(zs)


critics = {}
for spec in CRITICS:
    label, path = spec.split("=", 1)
    m = load_metric(path, device=dev); critics[label] = (m, int(getattr(m, "latent_dim", 192)) // 192)


@torch.no_grad()
def V(label, z, zg):
    m, W = critics[label]
    return m(z.repeat(1, W), zg.expand(len(z), -1).repeat(1, W)).reshape(-1).cpu().numpy()


env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224).unwrapped; env.reset(seed=0)


def render(st):
    env._set_state(np.asarray(st[:5], dtype=np.float64)); return env.render()


rng = np.random.default_rng(DRAW + 1000)
TYPES = ["block", "agent", "joint", "angle"]
# curves[label][type] -> (NT, len(R)) of V(pert) - V(0); latent likewise
curves = {lab: {t: [] for t in TYPES} for lab in list(critics) + ["latent"]}
curves_jpeg = {lab: {t: [] for t in TYPES} for lab in list(critics)}
for e in range(len(rows)):
    g = s_goal[e]
    u = rng.normal(size=2); u /= np.linalg.norm(u)
    frames, index = [], []
    for r in RADII:
        for t in ("block", "agent", "joint"):
            s = g.copy()
            if t == "block": s[2:4] += r * u
            elif t == "agent": s[0:2] += r * u
            else: s[0:2] += r * u / np.sqrt(2); s[2:4] += r * u / np.sqrt(2)
            s[0:2] = np.clip(s[0:2], 20, 492); s[2:4] = np.clip(s[2:4], 40, 472)
            frames.append(render(s)); index.append((t, r))
    for a in ANGLES:
        s = g.copy(); s[4] = g[4] + np.radians(a); frames.append(render(s)); index.append(("angle", a))
    z = encode(frames); zg_render = encode([render(g)]); zg_jpeg = encode([goal_jpeg[e]])
    lat = (z - zg_render).norm(dim=-1).cpu().numpy()
    vals = {lab: V(lab, z, zg_render) for lab in critics}
    vals_j = {lab: V(lab, z, zg_jpeg) for lab in critics}
    for t in TYPES:
        sel = [i for i, (tt, _) in enumerate(index) if tt == t]
        base = [i for i, (tt, r) in enumerate(index) if tt == t and r == 0][0]
        curves["latent"][t].append(lat[sel] - lat[base])
        for lab in critics:
            curves[lab][t].append(vals[lab][sel] - vals[lab][base])
            curves_jpeg[lab][t].append(vals_j[lab][sel] - vals_j[lab][base])
    if e % 10 == 0:
        print(f"[res] task {e}: " + " ".join(f"{lab}: joint30={vals[lab][[i for i,(tt,r) in enumerate(index) if tt=='joint' and r==30][0]] - vals[lab][[i for i,(tt,r) in enumerate(index) if tt=='joint' and r==0][0]]:.2f}" for lab in critics), flush=True)


def auc(inside, outside):
    """P(V_inside < V_outside) over all pairs (lower V = closer)."""
    a = np.asarray(inside).ravel(); b = np.asarray(outside).ravel()
    if len(a) == 0 or len(b) == 0: return float("nan")
    return float((a[:, None] < b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def summarize(curve_dict):
    out = {}
    for lab, per_type in curve_dict.items():
        out[lab] = {}
        for t in TYPES:
            arr = np.stack(per_type[t])  # (NT, K)
            grid = ANGLES if t == "angle" else RADII
            mean = arr.mean(0)
            if t == "angle":
                inside = arr[:, [grid.index(a) for a in (0, 5, 10, 15)]]; outside = arr[:, [grid.index(a) for a in (25, 30, 40)]]
                span = mean[grid.index(40)]; near = mean[grid.index(20)]
            else:
                inside = arr[:, [grid.index(r) for r in (0, 5, 10, 15)]]; outside = arr[:, [grid.index(r) for r in (25, 30, 35, 40)]]
                span = mean[grid.index(60)]; near = mean[grid.index(30)]
            out[lab][t] = {"grid": grid, "mean_curve": [float(v) for v in mean], "std_curve": [float(v) for v in arr.std(0)],
                           "rel_slope_inside": float(near / span) if abs(span) > 1e-6 else float("nan"),
                           "auc_inside_vs_nearmiss": auc(inside, outside)}
    return out


res = {"draw": DRAW, "n_tasks": len(rows), "goal_src": "render", "critics": list(critics), "summary": summarize(curves),
       "summary_goal_jpeg": summarize(curves_jpeg)}
(OUT / "resolution_probe.json").write_text(json.dumps(res, indent=1))
print("[res] SUMMARY (goal = re-rendered goal config; rel_slope = fraction of the 60px/40deg rise reached at 30px/20deg)")
for lab in res["summary"]:
    print("[res] " + lab + " | " + " | ".join(f"{t}: slope {res['summary'][lab][t]['rel_slope_inside']:.2f} auc {res['summary'][lab][t]['auc_inside_vs_nearmiss']:.2f}" for t in TYPES), flush=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for k, t in enumerate(TYPES):
    for lab in res["summary"]:
        s = res["summary"][lab][t]; y = np.asarray(s["mean_curve"]); y = y / (abs(y[-1]) if abs(y[-1]) > 1e-6 else 1.0)
        ax[k].plot(s["grid"], y, marker="o", ms=3, label=lab)
    ax[k].axvline(20, color="k", ls="--", lw=0.8); ax[k].set_title(f"{t}: V(r)-V(0), normalised to the 60px/40deg rise", fontsize=8)
    ax[k].set_xlabel("px" if t != "angle" else "deg")
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "resolution_probe.png", dpi=110)
print("[res] DONE", flush=True)
