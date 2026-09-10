"""Can any contact rule separate hallucinated block motion from real pushes?

E17's penalty assumed agent-block SEPARATION discriminates: legitimate pushes
happen close, hallucinated ones far. At the expert-calibrated radius (p99 =
135 px) the term never fired for the refiner (bit-identical success); at 60 px
it taxed half of all real pushes. This script measures the joint distribution
instead of guessing a third threshold.

For every recorded rollout, every executed plan is re-imagined through the WM.
Each imagined step gets:

    move_imag   decoded block pose change in imagination (px)
    move_real   decoded block pose change in reality (px)
    gap         decoded agent-block centre distance (px), min over the step
    align       cos angle between the agent's displacement and the block's
                imagined displacement (a real push moves the block roughly the
                way the agent travels)
    approach    agent displacement toward the block centre (px)

A step is HALLUCINATED when the world model moved the block much more than
reality did, and LEGITIMATE when reality moved it. The script reports each
candidate feature's AUC for telling the two apart, so a discriminator is chosen
from data rather than assumed.

Env: D, H5, CD, OUT (default $D/contact), DRAW (42), LABELS, PROBE (the
tools/fit_state_probe artifact), HALL_EPS (20), MOVE_EPS (15).
"""

from __future__ import annotations

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
OUT = Path(os.environ.get("OUT", str(D / "contact")))
DRAW = int(os.environ.get("DRAW", "42"))
LABELS = os.environ.get("LABELS", "rlp cem_value cem_latent").split()
PROBE = os.environ["PROBE"]
HALL_EPS = float(os.environ.get("HALL_EPS", "20"))
MOVE_EPS = float(os.environ.get("MOVE_EPS", "15"))
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK, H, OFF = 5, 5, 25


def log(m: str) -> None:
    print(f"[contact] {m}", flush=True)


def to_frame(x):
    from PIL import Image

    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return np.asarray(Image.open(io.BytesIO(bytes(x))).convert("RGB"))
    x = np.asarray(x)
    if x.dtype.kind in "SO":
        return to_frame(x.item() if x.shape == () else x.tolist())
    return x.transpose(1, 2, 0) if (x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3) else x


with h5py.File(H5, "r") as h:
    act_all = np.asarray(h["action"][:], dtype=np.float64)
    ok = act_all[~np.isnan(act_all).any(1)]
    amu, asd = ok.mean(0), ok.std(0)

import stable_pretraining as spt  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

from rlp.core.rollout import rollout_traj  # noqa: E402
from rlp.core.value.contact import StateProbe  # noqa: E402
from rlp.core.world_model import load_pretrained  # noqa: E402

wm = load_pretrained(str(D / "lewm_pusht_official")).to(dev).eval()
wm.requires_grad_(False)
probe = StateProbe.load(PROBE, device=dev)
log(f"probe meta: {probe.meta}")
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
def imagine(z_hist, actions_raw, n):
    a = torch.as_tensor((actions_raw - amu) / asd, dtype=torch.float32, device=dev).reshape(1, H, -1)
    zh = z_hist
    while zh.shape[0] < 3:
        zh = torch.cat([zh[:1], zh], dim=0)
    return rollout_traj(wm, zh[-3:].unsqueeze(0), torch.zeros(1, 2, a.shape[-1], device=dev), a)[0][:n]


def pose(states):  # (T,6) -> agent xy, block xy, angle
    return states[:, :2], states[:, 2:4], np.arctan2(states[:, 5], states[:, 4])


def move_px(block, angle):
    d = np.linalg.norm(block[1:] - block[:-1], axis=1)
    da = np.abs((angle[1:] - angle[:-1] + np.pi) % (2 * np.pi) - np.pi)
    return d + 40.0 * da


import lance  # noqa: E402

rows: list[dict[str, float]] = []
for lab in LABELS:
    p = D / f"rec_{lab}_s{DRAW}.lance"
    if not p.exists():
        log(f"no recording for {lab}")
        continue
    t = lance.dataset(str(p)).to_table(columns=["episode_idx", "step_idx", "pixels", "action"])
    ep = t.column("episode_idx").to_numpy().reshape(-1)
    st = t.column("step_idx").to_numpy().reshape(-1)
    pix = np.asarray(t.column("pixels").to_pylist(), dtype=object)
    act = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float64)
    for e in np.unique(ep):
        m = ep == e
        order = np.argsort(st[m])
        frames = [to_frame(x) for x in pix[m][order]]
        actions = act[m][order]
        z = encode(frames)
        s_real_all = probe(z).cpu().numpy()
        for t0 in range(0, len(frames) - 1, H * BLOCK):
            a_exec = actions[t0 : t0 + H * BLOCK]
            nb = len(a_exec) // BLOCK
            if nb < 1:
                continue
            padded = np.concatenate([a_exec[: nb * BLOCK], np.zeros((H * BLOCK - nb * BLOCK, a_exec.shape[1]))])
            hist = z[max(0, t0 - 2 * BLOCK) : t0 + 1 : BLOCK] if t0 > 0 else z[t0 : t0 + 1]
            z_imag = imagine(hist, padded, nb)
            s_imag = probe(z_imag).cpu().numpy()
            idx_real = [min(t0 + (b + 1) * BLOCK, len(frames) - 1) for b in range(nb)]
            s_real = np.concatenate([s_real_all[t0 : t0 + 1], s_real_all[idx_real]])
            s_im = np.concatenate([s_real_all[t0 : t0 + 1], s_imag])
            ag_i, bl_i, an_i = pose(s_im)
            ag_r, bl_r, an_r = pose(s_real)
            mv_i, mv_r = move_px(bl_i, an_i), move_px(bl_r, an_r)
            sep = np.linalg.norm(ag_i - bl_i, axis=1)
            gap = np.minimum(sep[1:], sep[:-1])
            d_agent = ag_i[1:] - ag_i[:-1]
            d_block = bl_i[1:] - bl_i[:-1]
            na, nb_ = np.linalg.norm(d_agent, axis=1), np.linalg.norm(d_block, axis=1)
            align = (d_agent * d_block).sum(1) / np.maximum(na * nb_, 1e-6)
            to_block = bl_i[:-1] - ag_i[:-1]
            approach = (d_agent * to_block).sum(1) / np.maximum(np.linalg.norm(to_block, axis=1), 1e-6)
            for k in range(nb):
                rows.append(
                    {
                        "label": lab,
                        "task": int(e),
                        "t0": t0,
                        "step": k,
                        "move_imag": float(mv_i[k]),
                        "move_real": float(mv_r[k]),
                        "gap": float(gap[k]),
                        "sep_start": float(sep[k]),
                        "align": float(align[k]),
                        "approach": float(approach[k]),
                        "agent_step_px": float(na[k]),
                    }
                )
    log(f"{lab}: {sum(1 for r in rows if r['label'] == lab)} imagined steps")

R = rows
hall = [r for r in R if r["move_imag"] - r["move_real"] > HALL_EPS and r["move_imag"] > MOVE_EPS]
legit = [r for r in R if r["move_real"] > MOVE_EPS and r["move_imag"] - r["move_real"] <= HALL_EPS]
log(f"{len(R)} imagined steps: {len(hall)} hallucinated, {len(legit)} legitimate")


def auc(pos, neg):
    a, b = np.asarray(pos), np.asarray(neg)
    if not len(a) or not len(b):
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


summary: dict[str, object] = {"n_steps": len(R), "n_hallucinated": len(hall), "n_legitimate": len(legit)}
for feat, sign in (("gap", 1), ("sep_start", 1), ("align", -1), ("approach", -1), ("agent_step_px", 1)):
    hv = [sign * r[feat] for r in hall]
    lv = [sign * r[feat] for r in legit]
    summary[feat] = {
        "auc_hallucinated_over_legitimate": round(auc(hv, lv), 3),
        "hallucinated_p10_p50_p90": [round(pct([r[feat] for r in hall], q), 1) for q in (10, 50, 90)],
        "legitimate_p10_p50_p90": [round(pct([r[feat] for r in legit], q), 1) for q in (10, 50, 90)],
    }
    log(f"{feat:14s} AUC {summary[feat]['auc_hallucinated_over_legitimate']:.3f}  "
        f"hall {summary[feat]['hallucinated_p10_p50_p90']}  legit {summary[feat]['legitimate_p10_p50_p90']}")

# what fraction of hallucinated / legitimate motion would a gap threshold catch?
gates = {}
for radius in (60, 80, 100, 120, 135, 160):
    caught = sum(1 for r in hall if r["gap"] > radius - 0.0) / max(len(hall), 1)
    taxed = sum(1 for r in legit if r["gap"] > radius - 0.0) / max(len(legit), 1)
    gates[radius] = {"hallucinated_caught": round(caught, 3), "legitimate_taxed": round(taxed, 3)}
    log(f"gap > {radius:3d} px: catches {caught:.1%} of hallucination, taxes {taxed:.1%} of real pushes")
summary["gap_gates"] = gates
(OUT / "contact_discriminator.json").write_text(json.dumps({"summary": summary, "rows": R[:5000]}, indent=1))
log("DONE")
