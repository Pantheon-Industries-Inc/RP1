"""PushT figures (2026-09-08): value-function heatmaps alongside BFS geodesics,
and CEM-vs-RLP contrast videos, from a finished diagnostic tag.

Env: DIAG (dir with results_*_s<draw>.txt + rec_{rlp,cem_latent}_s<draw>.lance),
     ACTOR (dir with train_checkpoints/planner.pt + value_ac), WM (world-model
     dir), H5 (expert h5), OUT (output dir), DRAW (eval draw seed, default 42).

Heatmap: for a task (start state, goal frame) the agent is swept over a GxG
grid with the block fixed at its start pose; each frame is encoded and the
config-B critic V(z, z_goal) is evaluated (w-frame window = static stack).
Alongside: latent L2 distance to the goal latent, and the BFS geodesic on the
same grid from every free cell to the expert's agent position at the goal
frame, with the block (dilated by the agent radius) as obstacle. The PushT
geodesic in agent space is the first phase of the task only (reach the
contact point); it is the closest analogue of the TwoRoom room-lattice BFS.
"""
from __future__ import annotations

import io
import json
import os
import re
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import gymnasium as gym  # noqa: E402
import stable_worldmodel  # noqa: E402,F401
from PIL import Image  # noqa: E402

DIAG, ACTOR, WMD, H5, OUT = (Path(os.environ[k]) for k in ("DIAG", "ACTOR", "WM", "H5", "OUT"))
DRAW = int(os.environ.get("DRAW", "42"))
G = int(os.environ.get("GRID", "40"))
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def successes(label):
    txt = (DIAG / f"results_{label}_s{DRAW}.txt").read_text()
    m = re.search(r"'episode_successes':\s*array\((\[.*?\])", txt, re.S)
    return np.array([t == "True" for t in re.findall(r"True|False", m.group(1))])


def to_frame(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return np.asarray(Image.open(io.BytesIO(bytes(x))).convert("RGB"))
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = x.transpose(1, 2, 0)
    return x


# ---------------------------------------------------------------- eval draw
with h5py.File(H5, "r") as h:
    epi = np.asarray(h["episode_idx"][:]).reshape(-1); stp = np.asarray(h["step_idx"][:]).reshape(-1)
    lo, hi, off, n = 16000, 18685, 25, 50
    ep_ids = np.unique(epi); ep_ids = ep_ids[(ep_ids >= lo) & (ep_ids < hi)]
    maxlen = {int(e): int(stp[epi == e].max()) for e in ep_ids}
    keep = np.isin(epi, ep_ids)
    mx = np.zeros(len(epi), dtype=np.int64); mx[keep] = np.array([maxlen[int(e)] for e in epi[keep]])
    valid = np.nonzero(keep & (stp <= mx - off))[0]
    rows = np.sort(np.random.default_rng(DRAW).choice(valid, size=n, replace=False))
    states = np.asarray(h["state"][:], dtype=np.float64)
    start_state = states[rows]; goal_state = states[rows + off]
    goal_frames = [to_frame(h["pixels"][int(r) + off]) for r in rows]
    start_frames_h5 = [to_frame(h["pixels"][int(r)]) for r in rows]
print(f"[figs] draw {DRAW}: {len(rows)} tasks, state dim {start_state.shape[1]}", flush=True)

r_ok, c_ok = successes("rlp"), successes("cem_latent")
modeA = np.nonzero(~r_ok & c_ok)[0]; both_ok = np.nonzero(r_ok & c_ok)[0]; both_fail = np.nonzero(~r_ok & ~c_ok)[0]
fail_eps = list(modeA[:5]) + list(both_fail[: max(0, 5 - len(modeA[:5]))])
succ_eps = list(both_ok[:5])
heat_eps = (list(modeA[:3]) + list(both_ok[:2]))[:5]
print(f"[figs] modeA {modeA.tolist()} both_ok {both_ok.tolist()[:10]} both_fail {both_fail.tolist()}", flush=True)

# ---------------------------------------------------------------- models
from rlp.core.value import load_metric  # noqa: E402
from rlp.core.world_model import load_pretrained  # noqa: E402
import stable_pretraining as spt  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

wm = load_pretrained(str(WMD)).to(dev).eval(); wm.requires_grad_(False)
ck = torch.load(ACTOR / "train_checkpoints" / "planner.pt", map_location="cpu", weights_only=False)
W = int(ck.get("window_frames") or 1)
vpath = ck["value"]
if not Path(vpath).exists():
    cand = [ACTOR / "train_checkpoints" / Path(vpath).name, ACTOR / "train_checkpoints" / Path(vpath).stem, ACTOR / "train_checkpoints" / "value_ac"]
    vpath = next(str(c) for c in cand if c.exists())
value = load_metric(vpath, device=dev)
print(f"[figs] critic {vpath} window {W}", flush=True)
stats = spt.data.dataset_stats.ImageNet
tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=stats["mean"], std=stats["std"]), T.Resize(224)])


@torch.no_grad()
def encode(frames, bs=256):
    zs = []
    for i in range(0, len(frames), bs):
        px = torch.stack([tf(f) for f in frames[i:i + bs]]).to(dev).unsqueeze(1)  # (B,1,3,H,W)
        zs.append(wm.encode({"pixels": px})["emb"][:, 0].float())
    return torch.cat(zs)


@torch.no_grad()
def V(z, zg):  # static W-frame window: repeat the frame, tile the goal
    return value(z.repeat(1, W), zg.repeat(1, W)).reshape(-1)


# ---------------------------------------------------------------- env renderer
env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224).unwrapped
env.reset(seed=0)
AGENT_R = 15.0
xs = np.linspace(8, 504, G)


def block_mask(state):
    env._set_state(state[:5])
    import shapely.geometry as sg
    polys = []
    for sh in env.block.shapes:
        vs = [tuple(env.block.local_to_world(v)) for v in sh.get_vertices()]
        polys.append(sg.Polygon(vs))
    blk = sg.MultiPolygon(polys).buffer(AGENT_R)
    mask = np.zeros((G, G), dtype=bool)
    for j, y in enumerate(xs):
        for i, x in enumerate(xs):
            mask[j, i] = blk.contains(sg.Point(x, y))
    return mask


def bfs(mask, target_xy):
    ti = int(np.argmin(np.abs(xs - target_xy[0]))); tj = int(np.argmin(np.abs(xs - target_xy[1])))
    dist = np.full((G, G), np.nan); step = (xs[1] - xs[0])
    if mask[tj, ti]:
        free = np.argwhere(~mask); k = np.argmin(((free - [tj, ti]) ** 2).sum(1)); tj, ti = free[k]
    dist[tj, ti] = 0.0; q = deque([(tj, ti)])
    while q:
        j, i = q.popleft()
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == di == 0: continue
                nj, ni = j + dj, i + di
                if 0 <= nj < G and 0 <= ni < G and not mask[nj, ni] and np.isnan(dist[nj, ni]):
                    dist[nj, ni] = dist[j, i] + step * (2 ** 0.5 if dj and di else 1.0); q.append((nj, ni))
    return dist


# ---------------------------------------------------------------- heatmaps
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

zg_all = encode(goal_frames)
summary = {}
for k, e in enumerate(heat_eps):
    s0 = start_state[e]; g = goal_state[e]
    frames = []
    for y in xs:
        for x in xs:
            st = s0.copy(); st[0], st[1] = x, y
            env._set_state(st[:5]); frames.append(env.render())
    z = encode(frames)
    v = V(z, zg_all[e:e + 1].expand(len(z), -1)).cpu().numpy().reshape(G, G)
    lat = (z - zg_all[e:e + 1]).norm(dim=-1).cpu().numpy().reshape(G, G)
    mask = block_mask(s0); geo = bfs(mask, g[:2])
    v_m = np.where(mask, np.nan, v); lat_m = np.where(mask, np.nan, lat)
    ok = ~np.isnan(geo) & ~mask
    from scipy.stats import spearmanr
    rho_v = float(spearmanr(v_m[ok], geo[ok]).statistic); rho_l = float(spearmanr(lat_m[ok], geo[ok]).statistic)
    tag = "modeA" if e in modeA else ("both_ok" if e in both_ok else "both_fail")
    summary[int(e)] = {"class": tag, "spearman_V_vs_bfs": rho_v, "spearman_latent_vs_bfs": rho_l}
    env._set_state(s0[:5]); start_img = env.render()
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    ax[0].imshow(start_img); ax[0].set_title(f"task {e} ({tag}) start frame"); ax[0].axis("off")
    ext = [0, 512, 512, 0]
    for a, arr, ttl in ((ax[1], v_m, f"critic V(agent xy | block at start, goal)  rho_bfs={rho_v:.2f}"),
                        (ax[2], lat_m, f"latent L2 to goal  rho_bfs={rho_l:.2f}"),
                        (ax[3], geo, "BFS geodesic to expert's goal-frame agent xy")):
        im = a.imshow(arr, extent=ext, cmap="viridis"); a.set_title(ttl, fontsize=9); plt.colorbar(im, ax=a, fraction=0.046)
        a.plot(s0[0], s0[1], "wo", ms=7, mec="k"); a.plot(g[0], g[1], "w*", ms=11, mec="k"); a.plot(s0[2], s0[3], "rs", ms=6); a.plot(g[2], g[3], "r*", ms=9)
        a.set_xlim(0, 512); a.set_ylim(512, 0); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"PushT draw {DRAW} task {e}: white o = agent start, white * = agent at goal frame, red s = block start, red * = block goal", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / f"heatmap_task{e:02d}_{tag}.png", dpi=110); plt.close(fig)
    print(f"[figs] heatmap task {e} {tag}: rho(V,bfs)={rho_v:.3f} rho(lat,bfs)={rho_l:.3f}", flush=True)

# ---------------------------------------------------------------- videos
import imageio  # noqa: E402
import lance  # noqa: E402


def load_rollouts(label):
    ds = lance.dataset(str(DIAG / f"rec_{label}_s{DRAW}.lance"))
    t = ds.to_table(columns=["episode_idx", "step_idx", "pixels"])
    ep = t.column(0).to_numpy().reshape(-1); st = t.column(1).to_numpy().reshape(-1)
    pix = np.asarray(t.column(2).to_pylist(), dtype=object)
    out = {}
    for e in np.unique(ep):
        m = ep == e; order = np.argsort(st[m])
        out[int(e)] = [to_frame(p) for p in pix[m][order]]
    return out


roll = {lab: load_rollouts(lab) for lab in ("rlp", "cem_latent")}
print(f"[figs] rollouts: rlp {len(roll['rlp'])} eps, cem {len(roll['cem_latent'])} eps", flush=True)


def border(img, ok, w=6):
    img = img.copy(); c = np.array([40, 200, 60] if ok else [220, 40, 40], dtype=np.uint8)
    img[:w] = c; img[-w:] = c; img[:, :w] = c; img[:, -w:] = c
    return img


index = ["| file | task | CEM | RLP |", "|---|---|---|---|"]
for kind, eps in (("fail", fail_eps), ("success", succ_eps)):
    for e in eps:
        e = int(e)
        keys = sorted(roll["rlp"].keys())
        ce, re_ = roll["cem_latent"][keys[e]], roll["rlp"][keys[e]]
        n_fr = max(len(ce), len(re_)); goal = goal_frames[e]
        frames = []
        for t in range(n_fr + 10):
            a = ce[min(t, len(ce) - 1)]; b = re_[min(t, len(re_) - 1)]
            frames.append(np.concatenate([border(a, bool(c_ok[e])), border(b, bool(r_ok[e])), goal], axis=1))
        name = f"{kind}_task{e:02d}_cem-{'OK' if c_ok[e] else 'FAIL'}_rlp-{'OK' if r_ok[e] else 'FAIL'}.mp4"
        imageio.mimwrite(OUT / name, frames, fps=10, codec="libx264", quality=7, macro_block_size=1)
        index.append(f"| {name} | {e} | {'OK' if c_ok[e] else 'FAIL'} | {'OK' if r_ok[e] else 'FAIL'} |")
(OUT / "INDEX.md").write_text(
    f"# PushT draw {DRAW}, actor {ACTOR.name}: CEM (left) | RLP (middle) | goal frame (right)\n\n"
    "Border colour = outcome of that planner on that task (green success, red failure). "
    "Failure videos = tasks RLP fails (CEM succeeds where available); success videos = both succeed.\n\n"
    + "\n".join(index) + "\n\nHeatmaps: " + json.dumps(summary) + "\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("[figs] DONE " + json.dumps(summary), flush=True)
