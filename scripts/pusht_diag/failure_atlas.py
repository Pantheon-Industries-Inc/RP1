"""Failure atlas for one eval draw: real path vs imagined path vs critic landscape.

For each task the refiner FAILS on, this builds the three views side by side:

1. the critic's landscape, as a heatmap of V over agent positions with the
   block held at its start pose, and over block positions with the agent held
   at its start, each overlaid with the real and imagined paths;
2. the ground-truth reference, the expert's own continuation from the start row
   (it reaches the goal by construction, so V must fall monotonically along it)
   and the BFS geodesic in agent space with the block as an obstacle;
3. the value and error curves along the expert path, the real path and the
   imagined path, so a failure is attributable to the critic misvaluing states
   (V wrong on the expert path), the world model fabricating motion (imagined
   and real diverge) or the optimiser stopping short (V flat, block error high).

Per task the JSON records the diagnosis inputs: V at the start, along the
expert path, at the real and imagined endpoints, decoded block and agent error
for each, the expert-path monotonicity, and the Spearman correlation between
the critic's agent-space landscape and the BFS geodesic.

Env: D, H5, CD, OUT (default $D/atlas), DRAW, PROBE, ACTOR_DIR (default
$D/train_checkpoints), CRITIC (default $ACTOR_DIR/value_ac), NT (max tasks),
GRID (heatmap resolution), LABELS.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch

with __import__("contextlib").suppress(ImportError):
    import hdf5plugin  # noqa: F401

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import gymnasium as gym  # noqa: E402
import stable_worldmodel  # noqa: E402,F401

D = Path(os.environ["D"])
H5 = os.environ["H5"]
OUT = Path(os.environ.get("OUT", str(D / "atlas")))
DRAW = int(os.environ.get("DRAW", "43"))
PROBE = os.environ["PROBE"]
ACTOR_DIR = Path(os.environ.get("ACTOR_DIR", str(D / "train_checkpoints")))
CRITIC = os.environ.get("CRITIC", str(ACTOR_DIR / "value_ac"))
NT = int(os.environ.get("NT", "6"))
G = int(os.environ.get("GRID", "32"))
LABELS = os.environ.get("LABELS", "rlp cem_value cem_latent").split()
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK, H, OFF = 5, 5, 25
AGENT_R = 15.0


def log(m: str) -> None:
    print(f"[atlas] {m}", flush=True)


def to_frame(x):
    from PIL import Image

    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return np.asarray(Image.open(io.BytesIO(bytes(x))).convert("RGB"))
    x = np.asarray(x)
    if x.dtype.kind in "SO":
        return to_frame(x.item() if x.shape == () else x.tolist())
    return x.transpose(1, 2, 0) if (x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3) else x


def successes(label: str) -> np.ndarray | None:
    p = D / f"results_{label}_s{DRAW}.txt"
    if not p.exists():
        return None
    m = re.search(r"episode_successes': array\(\[(.*?)\]", p.read_text(), re.S)
    return None if not m else np.array([v == "True" for v in re.findall(r"\b(True|False)\b", m.group(1))], dtype=bool)


# ---------------------------------------------------------------- the draw
with h5py.File(H5, "r") as h:
    epi = np.asarray(h["episode_idx"][:]).reshape(-1)
    stp = np.asarray(h["step_idx"][:]).reshape(-1)
    lo, hi, n = 16000, 18685, 50
    ep_ids = np.unique(epi)
    ep_ids = ep_ids[(ep_ids >= lo) & (ep_ids < hi)]
    maxlen = {int(e): int(stp[epi == e].max()) for e in ep_ids}
    keep = np.isin(epi, ep_ids)
    mx = np.zeros(len(epi), dtype=np.int64)
    mx[keep] = np.array([maxlen[int(e)] for e in epi[keep]])
    valid = np.nonzero(keep & (stp <= mx - OFF))[0]
    rows = np.sort(np.random.default_rng(DRAW).choice(valid, size=n, replace=False))
    states = np.asarray(h["state"][:], dtype=np.float64)
    s_start, s_goal = states[rows], states[rows + OFF]
    goal_frames = [to_frame(h["pixels"][int(r) + OFF]) for r in rows]
    # expert continuation: the logged states and frames for the 25 steps that DO reach the goal
    expert_states = np.stack([states[int(r) : int(r) + OFF + 1] for r in rows])  # (n, 26, 7)
    expert_frames = [[to_frame(h["pixels"][int(r) + k]) for k in range(0, OFF + 1, BLOCK)] for r in rows]
    act_all = np.asarray(h["action"][:], dtype=np.float64)
    ok_act = act_all[~np.isnan(act_all).any(1)]
    amu, asd = ok_act.mean(0), ok_act.std(0)

import stable_pretraining as spt  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

from rlp.core.rollout import rollout_traj  # noqa: E402
from rlp.core.value import load_metric  # noqa: E402
from rlp.core.value.contact import StateProbe  # noqa: E402
from rlp.core.world_model import load_pretrained  # noqa: E402

wm = load_pretrained(str(D / "lewm_pusht_official")).to(dev).eval()
wm.requires_grad_(False)
critic = load_metric(CRITIC, device=dev)
W = int(getattr(critic, "latent_dim", 192)) // 192
probe = StateProbe.load(PROBE, device=dev)
log(f"critic {CRITIC} window {W}; probe {probe.meta.get('r2')}")
stats = spt.data.dataset_stats.ImageNet
tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=stats["mean"], std=stats["std"]), T.Resize(224)])


@torch.no_grad()
def encode(frames, bs=128):
    zs = []
    for i in range(0, len(frames), bs):
        px = torch.stack([tf(to_frame(f)) for f in frames[i : i + bs]]).to(dev).unsqueeze(1)
        zs.append(wm.encode({"pixels": px})["emb"][:, 0].float())
    return torch.cat(zs)


@torch.no_grad()
def V(z, zg):
    return critic(z.repeat(1, W), zg.expand(len(z), -1).repeat(1, W)).reshape(-1).cpu().numpy()


@torch.no_grad()
def imagine(z_hist, actions_raw, n_blocks):
    a = torch.as_tensor((actions_raw - amu) / asd, dtype=torch.float32, device=dev).reshape(1, H, -1)
    zh = z_hist
    while zh.shape[0] < 3:
        zh = torch.cat([zh[:1], zh], dim=0)
    return rollout_traj(wm, zh[-3:].unsqueeze(0), torch.zeros(1, 2, a.shape[-1], device=dev), a)[0][:n_blocks]


def decode(z):
    return probe(z).cpu().numpy()


def err(dec, goal):
    ag = np.linalg.norm(dec[:, :2] - goal[:2], axis=1)
    bl = np.linalg.norm(dec[:, 2:4] - goal[2:4], axis=1)
    return ag, bl


# ---------------------------------------------------------------- env + geodesic
env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224).unwrapped
env.reset(seed=0)
xs = np.linspace(8, 504, G)


def render(state):
    env._set_state(np.asarray(state[:5], dtype=np.float64))
    return env.render()


def block_mask(state):
    import shapely.geometry as sg

    env._set_state(np.asarray(state[:5], dtype=np.float64))
    polys = [sg.Polygon([tuple(env.block.local_to_world(v)) for v in sh.get_vertices()]) for sh in env.block.shapes]
    blk = sg.MultiPolygon(polys).buffer(AGENT_R)
    mask = np.zeros((G, G), dtype=bool)
    for j, y in enumerate(xs):
        for i, x in enumerate(xs):
            mask[j, i] = blk.contains(sg.Point(x, y))
    return mask


def bfs(mask, target_xy):
    ti = int(np.argmin(np.abs(xs - target_xy[0])))
    tj = int(np.argmin(np.abs(xs - target_xy[1])))
    dist = np.full((G, G), np.nan)
    step = xs[1] - xs[0]
    if mask[tj, ti]:
        free = np.argwhere(~mask)
        k = np.argmin(((free - [tj, ti]) ** 2).sum(1))
        tj, ti = free[k]
    dist[tj, ti] = 0.0
    q = deque([(tj, ti)])
    while q:
        j, i = q.popleft()
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == di == 0:
                    continue
                nj, ni = j + dj, i + di
                if 0 <= nj < G and 0 <= ni < G and not mask[nj, ni] and np.isnan(dist[nj, ni]):
                    dist[nj, ni] = dist[j, i] + step * (2**0.5 if dj and di else 1.0)
                    q.append((nj, ni))
    return dist


# ---------------------------------------------------------------- recordings
import lance  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

start_imgs = np.stack([np.asarray(render(s), dtype=np.float32) for s in s_start])


def load(label):
    p = D / f"rec_{label}_s{DRAW}.lance"
    if not p.exists():
        return {}
    t = lance.dataset(str(p)).to_table(columns=["episode_idx", "step_idx", "pixels", "action"])
    ep = t.column("episode_idx").to_numpy().reshape(-1)
    st = t.column("step_idx").to_numpy().reshape(-1)
    pix = np.asarray(t.column("pixels").to_pylist(), dtype=object)
    act = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float64)
    eps = {}
    for e in np.unique(ep):
        m = ep == e
        o = np.argsort(st[m])
        eps[int(e)] = ([to_frame(x) for x in pix[m][o]], act[m][o])
    ok = successes(label)
    ks = sorted(eps)
    if not ks:
        return {}
    cost = np.zeros((len(ks), len(start_imgs)))
    for a, k in enumerate(ks):
        f0 = np.asarray(eps[k][0][0], dtype=np.float32)
        cost[a] = ((start_imgs - f0) ** 2).mean(axis=(1, 2, 3))
        if ok is not None and len(eps[k][0]) < 50:
            for e in range(len(start_imgs)):
                if not ok[e]:
                    cost[a, e] += 1e6
    r_, c_ = linear_sum_assignment(cost)
    return {int(e): eps[ks[a]] for a, e in zip(r_, c_) if cost[a, e] < 1e5}


roll = {lab: load(lab) for lab in LABELS}
ok = {lab: successes(lab) for lab in LABELS}
for lab in LABELS:
    log(f"{lab}: {len(roll[lab])} episodes mapped, success {None if ok[lab] is None else round(100 * ok[lab].mean(), 1)}")

r_ok = ok.get("rlp")
cmp_lab = "cem_value" if ok.get("cem_value") is not None else "cem_latent"
c_ok = ok.get(cmp_lab)
if r_ok is None:
    raise RuntimeError("no rlp results for this draw")
fails = [e for e in range(len(rows)) if not r_ok[e] and e in roll.get("rlp", {})]
modeA = [e for e in fails if c_ok is not None and c_ok[e]]
picks = (modeA + [e for e in fails if e not in modeA])[:NT]
log(f"draw {DRAW}: rlp fails {len(fails)} tasks; mode A (rlp fail, {cmp_lab} ok) {modeA}; atlas for {picks}")

# ---------------------------------------------------------------- per task
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

zg_all = encode(goal_frames)
report: dict[str, object] = {"draw": DRAW, "critic": CRITIC, "window": W, "tasks": {}}
for e in picks:
    s0, g = s_start[e], s_goal[e]
    zg = zg_all[e : e + 1]
    # expert reference path (ground truth: it reaches the goal)
    z_exp = encode(expert_frames[e])
    v_exp = V(z_exp, zg)
    d_exp = decode(z_exp)
    ex_state = expert_states[e][:: BLOCK]
    # real + imagined for each planner
    per: dict[str, object] = {}
    paths: dict[str, dict[str, np.ndarray]] = {}
    for lab in LABELS:
        if e not in roll.get(lab, {}):
            continue
        frames, actions = roll[lab][e]
        z = encode(frames)
        v_real_full = V(z, zg)
        dec_real_full = decode(z)
        nb = min(H, (len(frames) - 1) // BLOCK)
        z_imag = imagine(z[0:1], np.concatenate([actions[: nb * BLOCK], np.zeros((H * BLOCK - nb * BLOCK, actions.shape[1]))]), nb) if nb >= 1 else None
        idx = [min((b + 1) * BLOCK, len(frames) - 1) for b in range(nb)]
        v_imag = V(z_imag, zg) if z_imag is not None else np.zeros(0)
        dec_imag = decode(z_imag) if z_imag is not None else np.zeros((0, 6))
        ag_r, bl_r = err(dec_real_full[idx], g)
        ag_i, bl_i = err(dec_imag, g) if len(dec_imag) else (np.zeros(0), np.zeros(0))
        per[lab] = {
            "ok": bool(ok[lab][e]) if ok[lab] is not None else None,
            "steps": len(frames),
            "V_start": round(float(v_real_full[0]), 2),
            "V_real_first_plan": [round(float(x), 2) for x in v_real_full[idx]],
            "V_imag_first_plan": [round(float(x), 2) for x in v_imag],
            "V_real_final": round(float(v_real_full[-1]), 2),
            "block_err_real_first_plan": [round(float(x), 1) for x in bl_r],
            "block_err_imag_first_plan": [round(float(x), 1) for x in bl_i],
            "block_err_real_final": round(float(err(dec_real_full[-1:], g)[1][0]), 1),
            "agent_err_real_final": round(float(err(dec_real_full[-1:], g)[0][0]), 1),
        }
        paths[lab] = {"real": dec_real_full, "imag": dec_imag}
    # critic landscape over agent xy (block at start) and block xy (agent at start)
    frames_a, frames_b = [], []
    for y in xs:
        for x in xs:
            a_ = s0.copy(); a_[0], a_[1] = x, y; frames_a.append(render(a_))
            b_ = s0.copy(); b_[2], b_[3] = x, y; frames_b.append(render(b_))
    va = V(encode(frames_a), zg).reshape(G, G)
    vb = V(encode(frames_b), zg).reshape(G, G)
    mask = block_mask(s0)
    geo = bfs(mask, g[:2])
    valid_cells = ~np.isnan(geo) & ~mask
    va_m = np.where(mask, np.nan, va)
    rho = float(spearmanr(va_m[valid_cells], geo[valid_cells]).statistic)
    mono = float(np.mean(np.diff(v_exp) <= 0))
    report["tasks"][str(e)] = {
        "class": "modeA" if e in modeA else "both_fail",
        "block_move_required_px": round(float(np.linalg.norm(g[2:4] - s0[2:4])), 1),
        "block_rot_required_deg": round(float(np.degrees(abs((g[4] - s0[4] + np.pi) % (2 * np.pi) - np.pi))), 1),
        "agent_move_required_px": round(float(np.linalg.norm(g[:2] - s0[:2])), 1),
        "V_expert_path": [round(float(x), 2) for x in v_exp],
        "V_expert_monotone_fraction": round(mono, 2),
        "block_err_expert_path": [round(float(x), 1) for x in err(d_exp, g)[1]],
        "spearman_V_agentsweep_vs_bfs": round(rho, 3),
        "V_agent_sweep_range": round(float(np.nanmax(va_m) - np.nanmin(va_m)), 2),
        "V_block_sweep_range": round(float(vb.max() - vb.min()), 2),
        "planners": per,
    }
    log(f"task {e} [{report['tasks'][str(e)]['class']}]: block move needed {report['tasks'][str(e)]['block_move_required_px']} px; "
        f"V expert {[round(float(x),1) for x in v_exp]} (monotone {mono:.2f}); rho(V,bfs) {rho:.2f}")

    # figure: landscapes + curves
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.8))
    ext = [0, 512, 512, 0]
    ax[0].imshow(render(s0)); ax[0].set_title(f"task {e} start", fontsize=9); ax[0].axis("off")
    for a_, arr, ttl in (
        (ax[1], va_m, f"V over AGENT xy (block at start)  rho_bfs={rho:.2f}"),
        (ax[2], vb, "V over BLOCK xy (agent at start)"),
        (ax[3], geo, "BFS geodesic, agent space, block as obstacle"),
    ):
        im = a_.imshow(arr, extent=ext, cmap="viridis")
        plt.colorbar(im, ax=a_, fraction=0.046)
        a_.set_title(ttl, fontsize=8)
        a_.plot(s0[0], s0[1], "wo", ms=6, mec="k"); a_.plot(g[0], g[1], "w*", ms=10, mec="k")
        a_.plot(s0[2], s0[3], "rs", ms=6, mfc="none"); a_.plot(g[2], g[3], "r*", ms=9)
        a_.plot(ex_state[:, 0], ex_state[:, 1], "-", color="deepskyblue", lw=1.5)
        a_.plot(ex_state[:, 2], ex_state[:, 3], "-", color="orange", lw=1.5)
        if "rlp" in paths:
            pr = paths["rlp"]["real"]; pi = paths["rlp"]["imag"]
            a_.plot(pr[:, 0], pr[:, 1], "w-", lw=1.2)
            if len(pi):
                a_.plot(pi[:, 0], pi[:, 1], "w--", lw=1.0)
            a_.plot(pr[:, 2], pr[:, 3], "r-", lw=1.2)
            if len(pi):
                a_.plot(pi[:, 2], pi[:, 3], "r--", lw=1.0)
        a_.set_xlim(0, 512); a_.set_ylim(512, 0); a_.set_xticks([]); a_.set_yticks([])
    steps = np.arange(len(v_exp)) * BLOCK
    ax[4].plot(steps, v_exp, "-o", color="deepskyblue", ms=3, label="expert path (reaches goal)")
    for lab, col in (("rlp", "tab:blue"), (cmp_lab, "tab:orange")):
        if lab in per:
            vr = per[lab]["V_real_first_plan"]; vi = per[lab]["V_imag_first_plan"]
            ax[4].plot(np.arange(1, len(vr) + 1) * BLOCK, vr, "-s", color=col, ms=3, label=f"{lab} real")
            if vi:
                ax[4].plot(np.arange(1, len(vi) + 1) * BLOCK, vi, "--^", color=col, ms=3, alpha=0.7, label=f"{lab} imagined")
    ax[4].set_title("critic V along the three paths", fontsize=9); ax[4].set_xlabel("primitive step"); ax[4].legend(fontsize=6)
    fig.suptitle(
        f"draw {DRAW} task {e}: cyan = expert agent, orange = expert block, white = rlp agent (dashed imagined), "
        f"red = rlp block (dashed imagined); o/s start, * goal",
        fontsize=9,
    )
    fig.tight_layout(); fig.savefig(OUT / f"atlas_task{e:02d}.png", dpi=105); plt.close(fig)

(OUT / "failure_atlas.json").write_text(json.dumps(report, indent=1))
CHUNK = 3000
for name in ["failure_atlas.json"] + [f"atlas_task{e:02d}.png" for e in picks]:
    p = OUT / name
    if not p.exists():
        continue
    b = base64.b64encode(p.read_bytes()).decode()
    parts = [b[i : i + CHUNK] for i in range(0, len(b), CHUNK)]
    log(f"emitting {name}: {p.stat().st_size} bytes, {len(parts)} chunks")
    for k, part in enumerate(parts):
        print(f"[atlas-b64] {name} {k} {len(parts)} {part}", flush=True)
log("DONE")
