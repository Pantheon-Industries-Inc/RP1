"""PushT diagnostics E1/E2/E3 (2026-09-08), run inside a counterstrike job after
its evals (POST_SCRIPT_B64). Env: D (tag dir), H5 (expert h5). Reads
results_*/evallog_*/probes_*/rec_* written by the job, prints '[diag] ...'
JSON lines and writes $D/diag.json.

E1  oracle initialisation: rlp vs rlp_ceminit vs rlp_ceminit_k0 vs cem_latent,
    per episode (same draws), paired.
E2  refinement audit: per-iteration critic energy E_k and imagined latent
    distance from the rlp probes (t=0 replan), grouped by outcome class
    (RLP ok / mode A: RLP fail & CEM ok / both fail), plateau statistics, and
    the K=64 continuation.
E3  WM-error attribution: imagined-vs-real latent divergence over the first
    plan (5 blocks x 5 steps) under RLP's actions, CEM's actions and the
    expert's actions from the same start (recorded lances + the h5).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

D = Path(os.environ["D"])
H5 = os.environ.get("H5", "")
OUT: dict = {}


def log(k, v):
    OUT[k] = v
    print(f"[diag] {k} {json.dumps(v)}", flush=True)


def successes(label, seed):
    f = D / f"results_{label}_s{seed}.txt"
    if not f.exists():
        return None
    txt = f.read_text()
    m = re.search(r"'episode_successes':\s*array\((\[.*?\])", txt, re.S)
    return np.array([t == "True" for t in re.findall(r"True|False", m.group(1))]) if m else None


SEEDS = sorted({int(m.group(1)) for f in D.glob("results_rlp_s*.txt") for m in [re.search(r"_s(\d+)\.txt$", f.name)] if m})
LABELS = ["rlp", "rlp_ceminit", "rlp_ceminit_k0", "rlp_k64", "cem_latent", "noop"]

# ------------------------------------------------------------------ E1
e1 = {}
for lab in LABELS:
    rates = []
    for s in SEEDS:
        a = successes(lab, s)
        if a is not None:
            rates.append(float(a.mean() * 100))
    if rates:
        e1[lab] = {"per_draw": rates, "mean": float(np.mean(rates))}
log("E1_success", e1)
pairs = {}
for s in SEEDS:
    r, c, ci, k0 = (successes(x, s) for x in ("rlp", "cem_latent", "rlp_ceminit", "rlp_ceminit_k0"))
    if r is None or c is None:
        continue
    d = {"n": int(len(r)), "rlp_ok": int(r.sum()), "cem_ok": int(c.sum()),
         "modeA_rlp_fail_cem_ok": int((~r & c).sum()), "rlp_ok_cem_fail": int((r & ~c).sum()),
         "both_fail": int((~r & ~c).sum()), "union": int((r | c).sum())}
    if ci is not None:
        d["ceminit_ok"] = int(ci.sum()); d["ceminit_fixes_modeA"] = int((~r & c & ci).sum())
        d["ceminit_breaks_rlp_ok"] = int((r & ~ci).sum()); d["ceminit_vs_cem_lost"] = int((c & ~ci).sum())
    if k0 is not None:
        d["k0_ok"] = int(k0.sum()); d["k0_agrees_cem"] = int((k0 == c).sum())
    pairs[f"s{s}"] = d
log("E1_paired", pairs)

# ------------------------------------------------------------------ E2
def load_probes(label, seed):
    files = sorted(glob.glob(str(D / f"probes_{label}_s{seed}" / "probe_*.pt")))
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


e2 = {}
for lab in ("rlp", "rlp_k64", "rlp_ceminit"):
    curves = {"rlp_ok": [], "modeA": [], "both_fail": []}
    plateau = []
    for s in SEEDS:
        pr = load_probes(lab, s)
        r, c = successes("rlp", s), successes("cem_latent", s)
        if not pr or r is None or c is None:
            continue
        p0 = pr[0]  # first replan (t=0) for all envs
        E = p0["E_iters"].numpy()  # (K, B)
        if E.ndim != 2 or E.shape[0] == 0:
            continue
        Ef = p0["E"].numpy().reshape(-1)  # E after the last update
        full = np.concatenate([E, Ef[None]], 0)  # (K+1, B)
        cls = np.where(r, "rlp_ok", np.where(c, "modeA", "both_fail"))
        for i in range(full.shape[1]):
            curves[str(cls[i])].append(full[:, i].tolist())
        drop = full[0] - full[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac2 = np.where(np.abs(drop) > 1e-6, (full[0] - full[min(2, len(full) - 1)]) / drop, np.nan)
        mono = (np.diff(full, axis=0) <= 1e-6).mean(0)
        plateau.append({"seed": s, "E0_mean": float(full[0].mean()), "Ef_mean": float(full[-1].mean()),
                        "frac_drop_by_k2_median": float(np.nanmedian(frac2)), "monotone_frac_mean": float(mono.mean()),
                        "Ef_modeA_mean": float(full[-1][cls == "modeA"].mean()) if (cls == "modeA").any() else None,
                        "Ef_rlpok_mean": float(full[-1][cls == "rlp_ok"].mean()) if (cls == "rlp_ok").any() else None,
                        "lat_final_modeA": float(p0["lat_traj"][:, -1].numpy()[cls == "modeA"].mean()) if (cls == "modeA").any() else None,
                        "lat_final_rlpok": float(p0["lat_traj"][:, -1].numpy()[cls == "rlp_ok"].mean()) if (cls == "rlp_ok").any() else None})
    if plateau:
        e2[lab] = {"plateau": plateau, "mean_curve": {k: (np.mean(np.array(v), 0).round(4).tolist() if v else None) for k, v in curves.items()},
                   "n": {k: len(v) for k, v in curves.items()}}
log("E2_refinement", e2)

# ------------------------------------------------------------------ E3
def wm_divergence():
    try:
        import lance
        from rlp.core.world_model import load_pretrained
        from rlp.core.rollout import rollout_traj
        import stable_pretraining as spt
        from torchvision.transforms import v2 as T
        import h5py
    except Exception as e:  # noqa: BLE001
        return {"error": f"imports: {e}"}
    wmdir = D / "lewm_pusht_official"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wm = load_pretrained(str(wmdir)).to(dev).eval(); wm.requires_grad_(False)
    stats = spt.data.dataset_stats.ImageNet
    tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=stats["mean"], std=stats["std"]), T.Resize(224)])
    BLOCK, H = 5, 5

    def to_frame(x):  # recorded lances hold JPEG bytes; the h5 holds uint8 arrays
        if isinstance(x, (bytes, bytearray, np.bytes_)):
            import io
            from PIL import Image
            return np.asarray(Image.open(io.BytesIO(bytes(x))).convert("RGB"))
        x = np.asarray(x)
        if x.dtype.kind in "SO":
            return to_frame(x.item() if x.shape == () else x.tolist())
        return x

    def enc(frames):  # (T,H,W,3) uint8 -> (T,D)
        px = torch.stack([tf(to_frame(f)) for f in frames]).to(dev)
        with torch.no_grad():
            return wm.encode({"pixels": px.unsqueeze(0)})["emb"][0].float()

    # action normalisation used by the eval (StandardScaler on the h5 actions)
    amu = asd = None
    if H5 and os.path.exists(H5):
        with h5py.File(H5, "r") as h:
            act = np.asarray(h["action"][:], dtype=np.float64); act = act[~np.isnan(act).any(1)]
            amu, asd = act.mean(0), act.std(0)

    def imagine(z_frames, actions_raw):  # real block-boundary latents (H+1,D), raw actions (H*BLOCK,a)
        a = (actions_raw - amu) / asd if amu is not None else actions_raw
        A = torch.as_tensor(a, dtype=torch.float32, device=dev).reshape(1, H, -1)
        z0 = z_frames[0:1].unsqueeze(0).expand(1, 3, -1)
        a_hist = torch.zeros(1, 2, A.shape[-1], device=dev)
        with torch.no_grad():
            return rollout_traj(wm, z0, a_hist, A)[0]  # (H,D)

    out = {}
    for lab in ("rlp", "cem_latent"):
        for s in SEEDS:
            p = D / f"rec_{lab}_s{s}.lance"
            if not p.exists():
                continue
            ds = lance.dataset(str(p))
            t = ds.to_table(columns=["episode_idx", "step_idx", "pixels", "action"])
            ep = t.column("episode_idx").to_numpy().reshape(-1); st = t.column("step_idx").to_numpy().reshape(-1)
            eps = np.unique(ep)
            div, dgoal_imag, dgoal_real = [], [], []
            r = successes(lab, s)
            for i, e in enumerate(eps):
                m = ep == e
                order = np.argsort(st[m])
                pix_all = np.asarray(t.column("pixels").to_pylist(), dtype=object)
                pix = [pix_all[m][order][j] for j in range(int(m.sum()))]
                act = np.stack(t.column("action").to_numpy(zero_copy_only=False)[m][order]).astype(np.float64)
                if len(pix) < H * BLOCK + 1:
                    continue
                frames = [pix[j] for j in range(0, H * BLOCK + 1, BLOCK)]  # t = 0,5,...,25
                z_real = enc(frames)  # (H+1, D)
                z_imag = imagine(z_real, act[: H * BLOCK])  # (H, D)
                d = (z_imag - z_real[1:]).norm(dim=-1).cpu().numpy()
                div.append(d.tolist())
            if div:
                dv = np.array(div)
                out[f"{lab}_s{s}"] = {"n": int(len(dv)), "divergence_by_block_mean": dv.mean(0).round(4).tolist(),
                                     "divergence_final_mean": float(dv[:, -1].mean())}
                if r is not None and len(r) == len(dv):
                    out[f"{lab}_s{s}"]["divergence_final_fail_vs_ok"] = [float(dv[~r, -1].mean()) if (~r).any() else None, float(dv[r, -1].mean()) if r.any() else None]
    # expert arm: same start rows as the eval draw (replicate the draw)
    try:
        import h5py
        from rlp.eval.world_model import get_episodes_length
        with h5py.File(H5, "r") as h:
            epi = np.asarray(h["episode_idx"][:]).reshape(-1); stp = np.asarray(h["step_idx"][:]).reshape(-1)
            lo, hi, off, n = 16000, 18685, 25, 50
            ep_ids = np.unique(epi); ep_ids = ep_ids[(ep_ids >= lo) & (ep_ids < hi)]
            maxlen = {e: int(stp[epi == e].max()) for e in ep_ids}
            keep = np.isin(epi, ep_ids)
            mx = np.zeros(len(epi), dtype=np.int64); mx[keep] = np.array([maxlen[e] for e in epi[keep]])
            valid = np.nonzero(keep & (stp <= mx - off))[0]
            exp_out = {}
            for s in SEEDS:
                g = np.random.default_rng(s)
                rows = np.sort(g.choice(valid, size=n, replace=False))
                div = []
                for r0 in rows:
                    pix = np.asarray(h["pixels"][r0 : r0 + H * BLOCK + 1 : BLOCK])
                    act = np.asarray(h["action"][r0 : r0 + H * BLOCK], dtype=np.float64)
                    if np.isnan(act).any():
                        continue
                    z_real = enc(pix); z_imag = imagine(z_real, act)
                    div.append((z_imag - z_real[1:]).norm(dim=-1).cpu().numpy().tolist())
                dv = np.array(div)
                exp_out[f"expert_s{s}"] = {"n": int(len(dv)), "divergence_by_block_mean": dv.mean(0).round(4).tolist(), "divergence_final_mean": float(dv[:, -1].mean())}
            out.update(exp_out)
    except Exception as e:  # noqa: BLE001
        out["expert_error"] = str(e)
    return out


try:
    log("E3_wm_divergence", wm_divergence())
except Exception as e:  # noqa: BLE001
    log("E3_error", str(e))
(D / "diag.json").write_text(json.dumps(OUT, indent=1))
print("[diag] DONE", flush=True)
