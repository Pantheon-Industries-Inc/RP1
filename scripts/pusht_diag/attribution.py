"""Attribute each failing episode to the world model, the critic, or the planner.

Uses only exact quantities where they exist: PushT's own recorded poses for the
truth, the deployed critic for what the planner believed, and the frozen WM to
re-imagine the executed plan. The probe is not used, so nothing here depends on
a decoder coarser than the success tolerance.

Per failing episode it reports

    true      the env's own test along the rollout (joint 20 px, 20 deg)
    belief    V at the executed endpoint, and V along the real path
    imagined  V at the terminal of each executed plan, re-rolled through the WM
    peers     whether the samplers on the SAME critic and WM solved it

and applies one stated rule, in order:

    3 planner   a sampler sharing the critic and the WM succeeds -> the objective
                admitted a solution the refiner did not find
    2 critic    the critic calls the executed endpoint nearly solved
                (V < v_done) while the true distance is outside tolerance
    1 world model  the imagined terminal was rated far better than the state
                reality delivered (V_real - V_imag > gap_eps)
    0 unattributed  none of the above (every planner failed and no signal fires)

Env: D (tag with recordings + train_checkpoints), H5, OUT, DRAW, RLP_LABEL,
PEERS (comma list), CRITIC, V_DONE, GAP_EPS, PEER_RESULT_DIRS (extra tag dirs
holding results_*.txt for peers not run in this job).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch

with __import__("contextlib").suppress(ImportError):
    import hdf5plugin  # noqa: F401

D = Path(os.environ["D"])
H5 = os.environ["H5"]
OUT = Path(os.environ.get("OUT", str(D / "attribution")))
DRAW = int(os.environ.get("DRAW", "43"))
RLP_LABEL = os.environ.get("RLP_LABEL", "rlp")
PEERS = [p for p in os.environ.get("PEERS", "cem_latent,cem_value").split(",") if p]
PEER_DIRS = [Path(p) for p in os.environ.get("PEER_RESULT_DIRS", "").split(",") if p]
CRITIC = os.environ.get("CRITIC", str(D / "train_checkpoints" / "value_ac"))
V_DONE = float(os.environ.get("V_DONE", "4.0"))
GAP_EPS = float(os.environ.get("GAP_EPS", "3.0"))
OFF, TOL_POS, TOL_ANG, BLOCK, H = 25, 20.0, np.pi / 9, 5, 5
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def log(m: str) -> None:
    print(f"[attr] {m}", flush=True)


def to_frame(x):
    from PIL import Image

    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return np.asarray(Image.open(io.BytesIO(bytes(x))).convert("RGB"))
    x = np.asarray(x)
    if x.dtype.kind in "SO":
        return to_frame(x.item() if x.shape == () else x.tolist())
    return x


def successes(label: str):
    for base in [D, *PEER_DIRS]:
        p = base / f"results_{label}_s{DRAW}.txt"
        if p.exists():
            m = re.search(r"episode_successes': array\(\[(.*?)\]", p.read_text(), re.S)
            if m:
                return np.array([v == "True" for v in re.findall(r"\b(True|False)\b", m.group(1))], dtype=bool)
    return None


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
    start_frames = [to_frame(h["pixels"][int(r)]) for r in rows]
    act_all = np.asarray(h["action"][:], dtype=np.float64)
    ok_act = act_all[~np.isnan(act_all).any(1)]
    amu, asd = ok_act.mean(0), ok_act.std(0)

import stable_pretraining as spt  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

from rlp.core.rollout import rollout_traj  # noqa: E402
from rlp.core.value import load_metric  # noqa: E402
from rlp.core.world_model import load_pretrained  # noqa: E402

wm = load_pretrained(str(D / "lewm_pusht_official")).to(dev).eval()
wm.requires_grad_(False)
critic = load_metric(CRITIC, device=dev)
W = int(getattr(critic, "latent_dim", 192)) // 192
stats = spt.data.dataset_stats.ImageNet
tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=stats["mean"], std=stats["std"]), T.Resize(224)])
log(f"critic {Path(CRITIC).name} window {W}; v_done {V_DONE}; gap_eps {GAP_EPS}")


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
def imagine(z_hist, actions_raw, nb):
    a = torch.as_tensor((actions_raw - amu) / asd, dtype=torch.float32, device=dev).reshape(1, H, -1)
    zh = z_hist
    while zh.shape[0] < 3:
        zh = torch.cat([zh[:1], zh], dim=0)
    return rollout_traj(wm, zh[-3:].unsqueeze(0), torch.zeros(1, 2, a.shape[-1], device=dev), a)[0][:nb]


import lance  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

starts = np.stack([np.asarray(f, dtype=np.float32) for f in start_frames])


def load(label):
    p = D / f"rec_{label}_s{DRAW}.lance"
    if not p.exists():
        return {}
    t = lance.dataset(str(p)).to_table(columns=["episode_idx", "step_idx", "pixels", "action", "pos_agent", "block_pose"])
    ep = t.column("episode_idx").to_numpy().reshape(-1)
    st = t.column("step_idx").to_numpy().reshape(-1)
    pix = np.asarray(t.column("pixels").to_pylist(), dtype=object)
    act = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float64)
    ag = np.stack(t.column("pos_agent").to_numpy(zero_copy_only=False)).astype(np.float64)
    bp = np.stack(t.column("block_pose").to_numpy(zero_copy_only=False)).astype(np.float64)
    eps = {}
    for e in np.unique(ep):
        m = ep == e
        o = np.argsort(st[m])
        eps[int(e)] = ([to_frame(x) for x in pix[m][o]], act[m][o], np.column_stack([ag[m][o], bp[m][o]]))
    ok = successes(label)
    ks = sorted(eps)
    cost = np.zeros((len(ks), len(starts)))
    for a_, k in enumerate(ks):
        f0 = np.asarray(eps[k][0][0], dtype=np.float32)
        cost[a_] = ((starts - f0) ** 2).mean(axis=(1, 2, 3))
        if ok is not None and len(eps[k][0]) < 50:
            for e in range(len(starts)):
                if not ok[e]:
                    cost[a_, e] += 1e6
    r_, c_ = linear_sum_assignment(cost)
    return {int(e): eps[ks[a_]] for a_, e in zip(r_, c_) if cost[a_, e] < 1e5}


roll = load(RLP_LABEL)
ok_rlp = successes(RLP_LABEL)
peer_ok = {p: successes(p) for p in PEERS}
peer_roll = {p: load(p) for p in PEERS}
log(f"{RLP_LABEL}: {len(roll)} episodes; peers { {p: (None if v is None else round(100 * v.mean(), 1)) for p, v in peer_ok.items()} }")

zg_all = encode(goal_frames)
report: dict[str, object] = {"draw": DRAW, "rule": {"v_done": V_DONE, "gap_eps": GAP_EPS}, "episodes": {}}
counts = {1: 0, 2: 0, 3: 0, 0: 0}
fails = [e for e in range(len(rows)) if ok_rlp is not None and not ok_rlp[e] and e in roll]
for e in fails:
    frames, actions, poses = roll[e]
    g = s_goal[e]
    dist = np.linalg.norm(poses[:, :4] - g[:4], axis=1)
    ang = np.abs(poses[:, 4] - g[4])
    z = encode(frames)
    zg = zg_all[e : e + 1]
    v_real = V(z, zg)
    # per executed plan: V at the imagined terminal vs V at the real state reached
    plans = []
    for t0 in range(0, len(frames) - 1, H * BLOCK):
        nb = min(H, (len(frames) - 1 - t0) // BLOCK)
        if nb < 1:
            continue
        a_exec = actions[t0 : t0 + nb * BLOCK]
        padded = np.concatenate([a_exec, np.zeros((H * BLOCK - len(a_exec), a_exec.shape[1]))])
        z_im = imagine(z[t0 : t0 + 1], padded, nb)
        v_im = V(z_im, zg)
        real_idx = min(t0 + nb * BLOCK, len(frames) - 1)
        plans.append({"t0": t0, "blocks": nb, "V_imag_terminal": float(v_im[-1]), "V_real_reached": float(v_real[real_idx])})
    gap = max((p["V_real_reached"] - p["V_imag_terminal"]) for p in plans) if plans else 0.0
    peers_ok = {p: (None if peer_ok[p] is None else bool(peer_ok[p][e])) for p in PEERS}
    peer_min = {}
    for p in PEERS:
        if e in peer_roll.get(p, {}):
            pp = peer_roll[p][e][2]
            peer_min[p] = round(float(np.linalg.norm(pp[:, :4] - g[:4], axis=1).min()), 1)
    v_end = float(v_real[-1])
    d_end, d_min = float(dist[-1]), float(dist.min())
    if any(v for v in peers_ok.values() if v):
        label, why = 3, "a sampler on the same critic and world model solved it"
    elif v_end < V_DONE and d_end > TOL_POS:
        label, why = 2, f"critic calls the endpoint nearly solved (V {v_end:.1f}) at {d_end:.0f} px true error"
    elif gap > GAP_EPS:
        label, why = 1, f"imagined terminal rated {gap:.1f} better than the state reality delivered"
    else:
        label, why = 0, "no planner solved it and no signal fires"
    counts[label] += 1
    report["episodes"][str(e)] = {
        "label": label,
        "why": why,
        "true_min_distance_px": round(d_min, 1),
        "true_final_distance_px": round(d_end, 1),
        "angle_err_at_min_deg": round(float(np.degrees(ang[int(np.argmin(dist))])), 1),
        "V_at_endpoint": round(v_end, 2),
        "V_at_start": round(float(v_real[0]), 2),
        "max_imagination_gap": round(float(gap), 2),
        "plans": plans,
        "peers_ok": peers_ok,
        "peers_min_distance_px": peer_min,
        "block_move_required_px": round(float(np.linalg.norm(g[2:4] - s_start[e][2:4])), 1),
        "block_rot_required_deg": round(float(np.degrees(abs((g[4] - s_start[e][4] + np.pi) % (2 * np.pi) - np.pi))), 1),
    }
    log(
        f"task {e:2d} -> {label} | true min {d_min:5.1f} final {d_end:5.1f} px | V end {v_end:5.2f} | "
        f"imag gap {gap:5.2f} | peers {peers_ok} {peer_min} | {why}"
    )

report["counts"] = {"world_model": counts[1], "critic": counts[2], "planner": counts[3], "unattributed": counts[0]}
log(f"counts {report['counts']}")
(OUT / "attribution.json").write_text(json.dumps(report, indent=1))
b = base64.b64encode((OUT / "attribution.json").read_bytes()).decode()
for k in range(0, len(b), 3000):
    print(f"[attr-b64] attribution.json {k // 3000} {(len(b) + 2999) // 3000} {b[k : k + 3000]}", flush=True)
log("DONE")
