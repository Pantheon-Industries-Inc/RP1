"""Failure videos with the world model's imagination overlaid (E31 rendering).

For selected tasks of one eval draw: a three-panel MP4 per task --
  [ RLP real rollout + overlays | CEM-latent real rollout | goal frame ]
The RLP panel draws, per executed plan, the WORLD MODEL'S imagined block
position (ridge-decoded from the re-imagined latents of the plan actually
executed, from the real history at the plan start) as a hollow marker, next to
the real block (exact pose). The fake-basin story predicts a real block that
sits still while the imagined marker drifts toward the goal. Captions carry
V_imag / V_real per block and the E22 label.
Env: D, H5, CD, DRAW (43), TASKS ("0,8,9,15,25,3,31,43" + auto 2 successes),
LABELS_NOTE (json task->label), OUT (default $D/videos_diag), FPS (8).
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
CD = Path(os.environ.get("CD", "/checkpoints/armin@pantheon.inc/counterstrike/caches"))
OUT = Path(os.environ.get("OUT", str(D / "videos_diag")))
DRAW = int(os.environ.get("DRAW", "43"))
TASKS = [int(t) for t in os.environ.get("TASKS", "0,8,9,15,25,3,31,43").split(",") if t]
NOTE = json.loads(os.environ.get("LABELS_NOTE", '{"0":"planner","8":"planner","9":"planner","15":"planner","25":"planner","3":"world model","31":"critic","43":"critic"}'))
FPS = int(os.environ.get("FPS", "8"))
CRITIC = os.environ.get("CRITIC", str(D / "train_checkpoints" / "value_ac"))
H, BLOCK, OFF = 5, 5, 25
ARENA = 512.0  # PushT state space is 512x512 px; frames are rendered at 224
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def log(m: str) -> None:
    print(f"[vid] {m}", flush=True)


def to_frame(x):
    from PIL import Image

    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return np.asarray(Image.open(io.BytesIO(bytes(x))).convert("RGB"))
    x = np.asarray(x)
    if x.dtype.kind in "SO":
        return to_frame(x.item() if x.shape == () else x.tolist())
    return x


def successes(label: str):
    p = D / f"results_{label}_s{DRAW}.txt"
    if not p.exists():
        return None
    m = re.search(r"episode_successes': array\(\[(.*?)\]", p.read_text(), re.S)
    return None if not m else np.array([v == "True" for v in re.findall(r"\b(True|False)\b", m.group(1))], dtype=bool)


with h5py.File(H5, "r") as h:
    epi = np.asarray(h["episode_idx"][:]).reshape(-1)
    stp = np.asarray(h["step_idx"][:]).reshape(-1)
    n16 = int(np.count_nonzero(epi < 16000))
    act_train = np.asarray(h["action"][:n16], dtype=np.float32)
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
    s_goal = states[rows + OFF]
    start_frames = [to_frame(h["pixels"][int(r)]) for r in rows]
    goal_frames = [to_frame(h["pixels"][int(r) + OFF]) for r in rows]
    st_z = np.asarray(h["state"][:n16], dtype=np.float32)
amu, asd = np.nanmean(act_train, 0), np.nanstd(act_train, 0) + 1e-6
starts = np.stack([np.asarray(f, dtype=np.float32) for f in start_frames])

import imageio  # noqa: E402
import lance  # noqa: E402
import stable_pretraining as spt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
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
def V(z, zg):
    return critic(z.repeat(1, W), zg.expand(len(z), -1).repeat(1, W)).reshape(-1).cpu().numpy()


@torch.no_grad()
def imagine(z_hist, actions_raw):
    a = torch.as_tensor((actions_raw - amu) / asd, dtype=torch.float32, device=dev).reshape(1, H, -1)
    zh = z_hist
    while zh.shape[0] < 3:
        zh = torch.cat([zh[:1], zh], dim=0)
    zh = zh[-3:].unsqueeze(0)
    ah = torch.zeros(1, 2, a.shape[-1], device=dev)
    return rollout_traj(wm, zh, ah, a)[0]  # (H, D)


cache = LatentCache.load(str(CD / "counterstrike_fs1.pt"), mmap=True)
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(cache.z), size=min(200_000, len(cache.z)), replace=False))
X = cache.z[torch.as_tensor(idx)].float().numpy().astype(np.float64)
S = st_z[idx].astype(np.float64)
Y = np.column_stack([S[:, 0], S[:, 1], S[:, 2], S[:, 3], np.cos(S[:, 4]), np.sin(S[:, 4])])
Xb = np.column_stack([X, np.ones(len(X))])
Wt = torch.as_tensor(np.linalg.solve(Xb.T @ Xb + np.eye(Xb.shape[1]), Xb.T @ Y), dtype=torch.float32, device=dev)


@torch.no_grad()
def decode(z):
    return (torch.cat([z, torch.ones(len(z), 1, device=dev)], dim=1) @ Wt).cpu().numpy()


def load(label):
    p = D / f"rec_{label}_s{DRAW}.lance"
    if not p.exists():
        log(f"no recording for {label}")
        return {}
    t = lance.dataset(str(p)).to_table(columns=["episode_idx", "step_idx", "pixels", "action", "pos_agent", "block_pose"])
    ep = t.column("episode_idx").to_numpy().reshape(-1)
    st = t.column("step_idx").to_numpy().reshape(-1)
    pix = np.asarray(t.column("pixels").to_pylist(), dtype=object)
    act = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
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


def px(xy):  # state coords (512 space) -> frame pixels (224)
    return (float(xy[0]) * 224 / ARENA, float(xy[1]) * 224 / ARENA)


def panel(frame, ok, caption, agent=None, block=None, imag_block=None, goal_block=None):
    im = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    col = (40, 200, 90) if ok else (230, 60, 60)
    d.rectangle([0, 0, im.width - 1, im.height - 1], outline=col, width=5)
    if goal_block is not None:  # goal block centre: thin white cross
        x, y = px(goal_block)
        d.line([x - 6, y, x + 6, y], fill=(255, 255, 255), width=1)
        d.line([x, y - 6, x, y + 6], fill=(255, 255, 255), width=1)
    if block is not None:  # real block centre: filled yellow dot
        x, y = px(block)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 220, 0))
    if imag_block is not None:  # imagined block centre: hollow magenta ring
        x, y = px(imag_block)
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 0, 220), width=3)
        if block is not None:
            bx, by = px(block)
            d.line([bx, by, x, y], fill=(255, 0, 220), width=1)
    if agent is not None:
        x, y = px(agent)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(0, 200, 255), width=2)
    d.rectangle([0, im.height - 26, im.width, im.height], fill=(0, 0, 0))
    d.text((6, im.height - 22), caption, fill=(255, 255, 255))
    return np.asarray(im)


z_goal_all = encode(goal_frames)
rlp, cem = load("rlp"), load("cem_latent")
ok_r, ok_c = successes("rlp"), successes("cem_latent")
# two successes for contrast (RLP ok, long enough to have a full plan)
extra = [e for e in sorted(rlp) if ok_r is not None and ok_r[e] and e not in TASKS][:2]
sel = TASKS + extra
log(f"rendering tasks {sel}")

manifest = []
for e in sel:
    if e not in rlp:
        log(f"task {e}: no RLP recording matched"); continue
    fr_r, act_r, pose_r = rlp[e]
    fr_c, _, pose_c = cem.get(e, (None, None, None))
    zg = z_goal_all[e : e + 1]
    goal_block = s_goal[e, 2:4]
    # imagined block per step, from each executed plan's real start history
    imag = {}
    vtxt = {}
    for t0 in range(0, len(fr_r) - 1, H * BLOCK):
        nb = min(H, (len(fr_r) - 1 - t0 + BLOCK - 1) // BLOCK)
        if nb < 1:
            break
        a_exec = act_r[t0 : t0 + H * BLOCK]
        if len(a_exec) < H * BLOCK:
            a_exec = np.concatenate([a_exec, np.zeros((H * BLOCK - len(a_exec), a_exec.shape[1]), np.float32)])
        hist = encode(fr_r[max(0, t0 - 2 * BLOCK) : t0 + 1 : BLOCK])
        zi = imagine(hist, a_exec)[:nb]
        dec = decode(zi)
        real_idx = [min(t0 + (b + 1) * BLOCK, len(fr_r) - 1) for b in range(nb)]
        zr = encode([fr_r[i] for i in real_idx])
        vi, vr = V(zi, zg), V(zr, zg)
        for b in range(nb):
            for t in range(t0 + b * BLOCK + 1, min(t0 + (b + 1) * BLOCK, len(fr_r) - 1) + 1):
                imag[t] = dec[b, 2:4]
                vtxt[t] = f"V imag {vi[b]:.1f}  real {vr[b]:.1f}"
    n_frames = max(len(fr_r), len(fr_c) if fr_c is not None else 0)
    frames = []
    lab = NOTE.get(str(e), "ok" if (ok_r is not None and ok_r[e]) else "fail")
    for t in range(n_frames):
        tr = min(t, len(fr_r) - 1)
        p1 = panel(fr_r[tr], bool(ok_r[e]) if ok_r is not None else True,
                   f"RLP t={tr:02d} {vtxt.get(tr, '')}", agent=pose_r[tr, :2], block=pose_r[tr, 2:4],
                   imag_block=imag.get(tr), goal_block=goal_block)
        if fr_c is not None:
            tc = min(t, len(fr_c) - 1)
            p2 = panel(fr_c[tc], bool(ok_c[e]) if ok_c is not None else True, f"CEM-latent t={tc:02d}",
                       agent=pose_c[tc, :2], block=pose_c[tc, 2:4], goal_block=goal_block)
        else:
            p2 = np.zeros_like(p1)
        p3 = panel(goal_frames[e], True, f"goal  task {e}  [{lab}]", goal_block=goal_block)
        frames.append(np.concatenate([p1, p2, p3], axis=1))
    frames += [frames[-1]] * FPS  # hold the last frame a second
    name = f"task{e:02d}_{lab.replace(' ', '')}_rlp-{'OK' if ok_r is not None and ok_r[e] else 'FAIL'}_cem-{'OK' if (ok_c is not None and e < len(ok_c) and ok_c[e]) else 'FAIL'}.mp4"
    imageio.mimwrite(OUT / name, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=1)
    manifest.append(name)
    log(f"wrote {name} ({len(frames)} frames)")

for name in manifest:
    b = base64.b64encode((OUT / name).read_bytes()).decode()
    total = (len(b) + 2999) // 3000
    for k in range(total):
        print(f"[vid-b64] {name} {k} {total} {b[k * 3000 : (k + 1) * 3000]}", flush=True)
log(f"DONE {len(manifest)} videos")
