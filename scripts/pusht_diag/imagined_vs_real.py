"""Imagined vs real: why does the refiner convert a latent-parity critic into ~70?

For every recorded eval rollout (RLP, CEM on the same critic, CEM on latent
L2; draw DRAW) and every executed plan (the deployment executes the full
5-block plan, so the executed 25 actions ARE the plan), re-imagine the plan
through the frozen WM from the real history and compare, block by block:

  * the critic's distance-to-goal V along the IMAGINED latents vs the REAL
    latents (optimism gap = what the planner believed minus what happened),
  * latent L2 to the goal along both,
  * decoded agent / block positions (ridge probe latent -> state fitted on the
    training cache) along both, i.e. the imagined path vs the real path.

Outputs (OUT): ivr_summary.json, ivr_distance_curves.png (V / latent L2 /
decoded block+agent error vs block, imagined dashed vs real solid, successes
vs failures, RLP vs CEM), ivr_plans_task*.png (arena: imagined vs real paths
for RLP and CEM on the same tasks). Env: D, H5, CD, OUT (default $D/ivr),
STATE_CACHE (row-aligned logged state; built from H5 if missing), DRAW (42),
LABELS ("rlp cem_value cem_latent"), CRITIC (default $D/train_checkpoints/value_ac).
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
    import hdf5plugin  # noqa: F401  (registers the HDF5 filters the PushT h5 pixels use)

D = Path(os.environ["D"])
H5 = os.environ["H5"]
CD = Path(os.environ.get("CD", "/checkpoints/armin@pantheon.inc/counterstrike/caches"))
OUT = Path(os.environ.get("OUT", str(D / "ivr")))
DRAW = int(os.environ.get("DRAW", "42"))
LABELS = os.environ.get("LABELS", "rlp cem_value cem_latent").split()
CRITIC = os.environ.get("CRITIC", str(D / "train_checkpoints" / "value_ac"))
STATE_CACHE = os.environ.get("STATE_CACHE", "")
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK, H, OFF = 5, 5, 25


def log(msg: str) -> None:
    print(f"[ivr] {msg}", flush=True)


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
    txt = p.read_text()
    m = re.search(r"episode_successes': array\(\[(.*?)\]", txt, re.S)
    if not m:
        return None
    return np.array([v == "True" for v in re.findall(r"\b(True|False)\b", m.group(1))], dtype=bool)


# ---------------------------------------------------------------- tasks of the draw (evaluator's rule)
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
    states_all = np.asarray(h["state"][:], dtype=np.float64)
    s_start, s_goal = states_all[rows], states_all[rows + OFF]
    start_frames = [to_frame(h["pixels"][int(r)]) for r in rows]
    goal_frames = [to_frame(h["pixels"][int(r) + OFF]) for r in rows]
    act_all = np.asarray(h["action"][:], dtype=np.float64)
    act_ok = act_all[~np.isnan(act_all).any(1)]
    amu, asd = act_ok.mean(0), act_ok.std(0)
log(f"draw {DRAW}: {n} tasks; block already within tolerance at start on "
    f"{int(((np.linalg.norm(s_goal[:, 2:4] - s_start[:, 2:4], axis=1) < 20) & (np.abs(s_goal[:, 4] - s_start[:, 4]) < np.pi / 9)).sum())}")

# ---------------------------------------------------------------- WM, critic, encoder
import stable_pretraining as spt  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

from rlp.core.rollout import rollout_traj  # noqa: E402
from rlp.core.value import load_metric  # noqa: E402
from rlp.core.world_model import load_pretrained  # noqa: E402
from rlp.data import LatentCache  # noqa: E402

wm = load_pretrained(str(D / "lewm_pusht_official")).to(dev).eval()
wm.requires_grad_(False)
critic = load_metric(CRITIC, device=dev)
W = int(getattr(critic, "latent_dim", 192)) // 192
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
def V(z, zg):  # z (T,D); zg (1,D). Window critics get a static window (only W=1 is deploy-exact).
    return critic(z.repeat(1, W), zg.expand(len(z), -1).repeat(1, W)).reshape(-1).cpu().numpy()


@torch.no_grad()
def imagine(z_hist, actions_raw):  # z_hist (k<=3, D) real history newest last; raw actions (H*BLOCK, a)
    a = torch.as_tensor((actions_raw - amu) / asd, dtype=torch.float32, device=dev).reshape(1, H, -1)
    zh = z_hist
    while zh.shape[0] < 3:
        zh = torch.cat([zh[:1], zh], dim=0)
    zh = zh[-3:].unsqueeze(0)
    ah = torch.zeros(1, 2, a.shape[-1], device=dev)
    return rollout_traj(wm, zh, ah, a)[0]  # (H, D) latents at t+5..t+25


# ---------------------------------------------------------------- ridge probe latent -> state
def load_state_rows():
    if STATE_CACHE and Path(STATE_CACHE).exists():
        sc = LatentCache.load(STATE_CACHE, mmap=False)
        return sc.z.numpy(), sc.episode_idx.numpy(), sc.step_idx.numpy()
    with h5py.File(H5, "r") as h:
        e = np.asarray(h["episode_idx"][:]).reshape(-1)
        n16 = int(np.count_nonzero(e < 16000))
        return np.asarray(h["state"][:n16], dtype=np.float32), e[:n16], np.asarray(h["step_idx"][:n16]).reshape(-1)


cache = LatentCache.load(str(CD / "counterstrike_fs1.pt"), mmap=True)
st_z, st_e, st_s = load_state_rows()
if len(st_z) != len(cache.z) or not np.array_equal(st_e, cache.episode_idx.numpy()):
    raise RuntimeError(f"state rows ({len(st_z)}) do not align with the fs1 cache ({len(cache.z)})")
rng = np.random.default_rng(0)
idx = rng.choice(len(cache.z), size=min(200_000, len(cache.z)), replace=False)
X = cache.z[torch.as_tensor(np.sort(idx))].float().numpy().astype(np.float64)
S = st_z[np.sort(idx)].astype(np.float64)
Y = np.column_stack([S[:, 0], S[:, 1], S[:, 2], S[:, 3], np.cos(S[:, 4]), np.sin(S[:, 4])])
ntr = int(0.9 * len(X))
Xb = np.column_stack([X, np.ones(len(X))])
lam = 1.0
Wr = np.linalg.solve(Xb[:ntr].T @ Xb[:ntr] + lam * np.eye(Xb.shape[1]), Xb[:ntr].T @ Y[:ntr])
pred = Xb[ntr:] @ Wr
r2 = 1 - ((pred - Y[ntr:]) ** 2).sum(0) / ((Y[ntr:] - Y[ntr:].mean(0)) ** 2).sum(0)
err_agent = np.linalg.norm(pred[:, :2] - Y[ntr:, :2], axis=1)
err_block = np.linalg.norm(pred[:, 2:4] - Y[ntr:, 2:4], axis=1)
log(f"ridge probe (held-out): R2 agent x/y {r2[0]:.3f}/{r2[1]:.3f} block x/y {r2[2]:.3f}/{r2[3]:.3f} "
    f"angle cos/sin {r2[4]:.3f}/{r2[5]:.3f}; median px error agent {np.median(err_agent):.1f} block {np.median(err_block):.1f}")
Wt = torch.as_tensor(Wr, dtype=torch.float32, device=dev)


@torch.no_grad()
def decode(z):  # (T, D) -> (T, 6)
    return (torch.cat([z, torch.ones(len(z), 1, device=dev)], dim=1) @ Wt).cpu().numpy()


def pos_err(dec, goal_state):
    """agent px error, block px error, block angle error (deg) of decoded states vs the goal state."""
    ag = np.linalg.norm(dec[:, :2] - goal_state[:2], axis=1)
    bl = np.linalg.norm(dec[:, 2:4] - goal_state[2:4], axis=1)
    ang = np.degrees(np.abs(np.arctan2(dec[:, 5], dec[:, 4]) - goal_state[4] + np.pi) % (2 * np.pi) - np.pi)
    return ag, bl, np.abs(ang)


# ---------------------------------------------------------------- recorded rollouts -> tasks
import lance  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

starts = np.stack([np.asarray(f, dtype=np.float32) for f in start_frames])


def load_rollouts(label):
    p = D / f"rec_{label}_s{DRAW}.lance"
    if not p.exists():
        log(f"no recording for {label}")
        return {}
    t = lance.dataset(str(p)).to_table(columns=["episode_idx", "step_idx", "pixels", "action"])
    ep = t.column("episode_idx").to_numpy().reshape(-1)
    st = t.column("step_idx").to_numpy().reshape(-1)
    pix = np.asarray(t.column("pixels").to_pylist(), dtype=object)
    act = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float64)
    out = {}
    for e in np.unique(ep):
        m = ep == e
        order = np.argsort(st[m])
        out[int(e)] = ([to_frame(p) for p in pix[m][order]], act[m][order])
    return out


def assign(eps, ok):
    ks = sorted(eps)
    if not ks:
        return {}
    cost = np.zeros((len(ks), len(starts)))
    for a, k in enumerate(ks):
        f0 = np.asarray(eps[k][0][0], dtype=np.float32)
        cost[a] = ((starts - f0) ** 2).mean(axis=(1, 2, 3))
        full = len(eps[k][0]) >= 50
        if ok is not None:
            for e in range(len(starts)):
                if not full and not ok[e]:  # stopped early => success (terminate_at_goal)
                    cost[a, e] += 1e6
    r_, c_ = linear_sum_assignment(cost)
    return {int(e): eps[ks[a]] for a, e in zip(r_, c_) if cost[a, e] < 1e5}


zg_all = encode(goal_frames)
results = {}
per_task = {}
for lab in LABELS:
    ok = successes(lab)
    roll = assign(load_rollouts(lab), ok)
    log(f"{lab}: {len(roll)} recorded episodes mapped to tasks; success rate from results {None if ok is None else round(100 * ok.mean(), 1)}")
    rec = []
    for e, (frames, act) in sorted(roll.items()):
        zg = zg_all[e : e + 1]
        z_frames = encode(frames)  # (T, D)
        plans = []
        for t0 in range(0, len(frames) - 1, H * BLOCK):
            if t0 + H * BLOCK > len(frames) - 1 + BLOCK:  # need at least one full block after t0
                break
            hist = z_frames[max(0, t0 - 2 * BLOCK) : t0 + 1 : BLOCK] if t0 > 0 else z_frames[t0 : t0 + 1]
            a_exec = act[t0 : t0 + H * BLOCK]
            nb = len(a_exec) // BLOCK
            if nb < 1:
                break
            a_exec = a_exec[: nb * BLOCK]
            if nb < H:  # episode ended early (success): pad the plan with zeros for the WM, mark n blocks
                a_exec = np.concatenate([a_exec, np.zeros((H * BLOCK - len(a_exec), a_exec.shape[1]))])
            z_imag = imagine(hist, a_exec)[:nb]
            real_idx = [min(t0 + (b + 1) * BLOCK, len(frames) - 1) for b in range(nb)]
            z_real = z_frames[real_idx]
            v_imag, v_real = V(z_imag, zg), V(z_real, zg)
            l_imag = (z_imag - zg).norm(dim=-1).cpu().numpy()
            l_real = (z_real - zg).norm(dim=-1).cpu().numpy()
            d_imag, d_real = decode(z_imag), decode(z_real)
            plans.append(
                {
                    "t0": t0,
                    "n_blocks": nb,
                    "v0": float(V(z_frames[t0 : t0 + 1], zg)[0]),
                    "v_imag": v_imag.tolist(),
                    "v_real": v_real.tolist(),
                    "l_imag": l_imag.tolist(),
                    "l_real": l_real.tolist(),
                    "div": (z_imag - z_real).norm(dim=-1).cpu().numpy().tolist(),
                    "dec_imag": d_imag.tolist(),
                    "dec_real": d_real.tolist(),
                    "dec_start": decode(z_frames[t0 : t0 + 1])[0].tolist(),
                }
            )
        rec.append({"task": e, "ok": bool(ok[e]) if ok is not None else None, "steps": len(frames), "plans": plans})
    per_task[lab] = rec
    # aggregates over FIRST plans (t0 = 0, the h25 plan that decides most tasks)
    firsts = [(r["ok"], r["plans"][0]) for r in rec if r["plans"] and r["plans"][0]["n_blocks"] == H]
    agg = {}
    for name, sel in (("ok", [p for o, p in firsts if o]), ("fail", [p for o, p in firsts if o is False])):
        if not sel:
            continue
        vi = np.array([p["v_imag"] for p in sel]); vr = np.array([p["v_real"] for p in sel])
        li = np.array([p["l_imag"] for p in sel]); lr = np.array([p["l_real"] for p in sel])
        dv = np.array([p["div"] for p in sel])
        gs = np.array([s_goal[r["task"]] for r in rec if r["plans"] and r["plans"][0] in sel])
        eb_i, eb_r, ea_i, ea_r = [], [], [], []
        for p, g in zip(sel, gs, strict=True):
            ai, bi, _ = pos_err(np.array(p["dec_imag"]), g); ar, br, _ = pos_err(np.array(p["dec_real"]), g)
            ea_i.append(ai); eb_i.append(bi); ea_r.append(ar); eb_r.append(br)
        agg[name] = {
            "n": len(sel),
            "v0_mean": float(np.mean([p["v0"] for p in sel])),
            "v_imag_by_block": vi.mean(0).round(2).tolist(),
            "v_real_by_block": vr.mean(0).round(2).tolist(),
            "optimism_gap_final(real-imag)": float((vr[:, -1] - vi[:, -1]).mean()),
            "latent_imag_by_block": li.mean(0).round(2).tolist(),
            "latent_real_by_block": lr.mean(0).round(2).tolist(),
            "divergence_by_block": dv.mean(0).round(2).tolist(),
            "decoded_agent_err_final_imag/real_px": [float(np.mean([a[-1] for a in ea_i])), float(np.mean([a[-1] for a in ea_r]))],
            "decoded_block_err_final_imag/real_px": [float(np.mean([b[-1] for b in eb_i])), float(np.mean([b[-1] for b in eb_r]))],
        }
    results[lab] = agg
    log(f"{lab} first-plan aggregates: " + json.dumps(agg))

(OUT / "ivr_summary.json").write_text(json.dumps({"draw": DRAW, "critic": CRITIC, "ridge_r2": r2.round(4).tolist(), "aggregates": results}, indent=1))
(OUT / "ivr_per_task.json").write_text(json.dumps(per_task))

# ---------------------------------------------------------------- figures
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

labs = [lab for lab in LABELS if results.get(lab)]
blocks = np.arange(1, H + 1) * BLOCK
fig, ax = plt.subplots(len(labs), 4, figsize=(19, 4.2 * len(labs)), squeeze=False)
for i, lab in enumerate(labs):
    rec = per_task[lab]
    for cls, col in (("ok", "tab:green"), ("fail", "tab:red")):
        sel = [r for r in rec if r["plans"] and r["plans"][0]["n_blocks"] == H and (r["ok"] if cls == "ok" else r["ok"] is False)]
        if not sel:
            continue
        P = [r["plans"][0] for r in sel]
        G = [s_goal[r["task"]] for r in sel]
        vi = np.array([p["v_imag"] for p in P]); vr = np.array([p["v_real"] for p in P])
        li = np.array([p["l_imag"] for p in P]); lr = np.array([p["l_real"] for p in P])
        ai = np.array([pos_err(np.array(p["dec_imag"]), g)[0] for p, g in zip(P, G, strict=True)])
        ar = np.array([pos_err(np.array(p["dec_real"]), g)[0] for p, g in zip(P, G, strict=True)])
        bi = np.array([pos_err(np.array(p["dec_imag"]), g)[1] for p, g in zip(P, G, strict=True)])
        br = np.array([pos_err(np.array(p["dec_real"]), g)[1] for p, g in zip(P, G, strict=True)])
        for k, (yi, yr, ttl) in enumerate(((vi, vr, "critic V to goal"), (li, lr, "latent L2 to goal"), (ai, ar, "decoded AGENT error to goal (px)"), (bi, br, "decoded BLOCK error to goal (px)"))):
            ax[i, k].plot(blocks, yr.mean(0), "-o", color=col, ms=3, label=f"real ({cls}, n={len(P)})")
            ax[i, k].plot(blocks, yi.mean(0), "--s", color=col, ms=3, alpha=0.7, label=f"imagined ({cls})")
            ax[i, k].set_title(f"{lab}: {ttl}", fontsize=9); ax[i, k].set_xlabel("primitive step of the first plan")
            if k >= 2:
                ax[i, k].axhline(20, color="k", ls=":", lw=0.8)
    ax[i, 0].legend(fontsize=7)
fig.suptitle(f"PushT draw {DRAW}, first 25-step plan: imagined (WM rollout of the executed plan, dashed) vs real (solid); critic = {Path(CRITIC).name}", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "ivr_distance_curves.png", dpi=110)

# arena plots: tasks where RLP failed and CEM(critic) succeeded, and tasks both succeeded
def by_task(lab):
    return {r["task"]: r for r in per_task.get(lab, [])}

R, C = by_task("rlp"), by_task("cem_value") if "cem_value" in per_task else by_task("cem_latent")
clab = "cem_value" if "cem_value" in per_task else "cem_latent"
modeA = [e for e in sorted(set(R) & set(C)) if R[e]["ok"] is False and C[e]["ok"]]
both_ok = [e for e in sorted(set(R) & set(C)) if R[e]["ok"] and C[e]["ok"]]
picks = modeA[:4] + both_ok[:2]
log(f"arena tasks: mode A {modeA[:4]} both-ok {both_ok[:2]}")
if picks:
    fig, ax = plt.subplots(len(picks), 2, figsize=(10, 5 * len(picks)), squeeze=False)
    for i, e in enumerate(picks):
        for j, (lab, rec) in enumerate((("rlp", R[e]), (clab, C[e]))):
            a_ = ax[i, j]; g = s_goal[e]; s0 = s_start[e]
            a_.set_xlim(0, 512); a_.set_ylim(512, 0); a_.set_aspect("equal")
            a_.plot(g[0], g[1], "b*", ms=12, label="agent goal"); a_.plot(g[2], g[3], "r*", ms=12, label="block goal")
            a_.plot(s0[0], s0[1], "bo", ms=7, mfc="none", label="agent start"); a_.plot(s0[2], s0[3], "rs", ms=7, mfc="none", label="block start")
            for p in rec["plans"]:
                di, dr = np.array(p["dec_imag"]), np.array(p["dec_real"]); d0 = np.array(p["dec_start"])
                a_.plot(np.r_[d0[0], dr[:, 0]], np.r_[d0[1], dr[:, 1]], "-o", color="tab:blue", ms=3, lw=1.5)
                a_.plot(np.r_[d0[0], di[:, 0]], np.r_[d0[1], di[:, 1]], "--s", color="tab:blue", ms=3, lw=1, alpha=0.6)
                a_.plot(np.r_[d0[2], dr[:, 2]], np.r_[d0[3], dr[:, 3]], "-o", color="tab:red", ms=3, lw=1.5)
                a_.plot(np.r_[d0[2], di[:, 2]], np.r_[d0[3], di[:, 3]], "--s", color="tab:red", ms=3, lw=1, alpha=0.6)
            p0 = rec["plans"][0] if rec["plans"] else None
            vtxt = f" V0 {p0['v0']:.1f} -> imag {p0['v_imag'][-1]:.1f} / real {p0['v_real'][-1]:.1f}" if p0 else ""
            a_.set_title(f"task {e} {lab}: {'OK' if rec['ok'] else 'FAIL'} ({rec['steps']} steps){vtxt}", fontsize=8)
            if i == 0 and j == 0:
                a_.legend(fontsize=6, loc="lower right")
    fig.suptitle("solid = real path (decoded from real frames), dashed = imagined path (decoded from WM latents of the executed plan); blue agent, red block", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "ivr_plans.png", dpi=100)

for name in ("ivr_summary.json", "ivr_distance_curves.png", "ivr_plans.png"):
    p = OUT / name
    if p.exists():
        print(f"[ivr-b64] {name} {base64.b64encode(p.read_bytes()).decode()}", flush=True)
log("DONE")
