"""E5/E8 critic probes (2026-09-08), POST_ONLY on a diagnostic tag. Env: D, H5.

E5  calibration real vs imagined: for each recorded RLP/CEM episode, the
    critic energy of the REAL window at the end of the first plan (frames at
    t=5..25 encoded) vs the IMAGINED window (WM rollout on the executed
    actions); optimism gap = E_imag - E_real by outcome class; AUC of E_real
    and E_imag for predicting real success. Co-trained critic (value_ac) and
    offline teacher (value_td) side by side.
E8  velocity channel: on the rlp probes (t=0 plan, z_traj), energy with the
    true 4-frame imagined window vs a static stack of the terminal frame
    (velocity zeroed). If mode-A plans gain much more energy under the static
    stack than successes, the refiner's advantage lives in the velocity
    channel of the window.
"""
from __future__ import annotations
import glob, io, json, os, re
from pathlib import Path
import numpy as np, torch
from PIL import Image

D = Path(os.environ["D"]); H5 = os.environ.get("H5", "")
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = {}
def log(k, v): OUT[k] = v; print(f"[probe] {k} {json.dumps(v)}", flush=True)

def successes(label, seed):
    f = D / f"results_{label}_s{seed}.txt"
    if not f.exists(): return None
    m = re.search(r"'episode_successes':\s*array\((\[.*?\])", f.read_text(), re.S)
    return np.array([t == "True" for t in re.findall(r"True|False", m.group(1))]) if m else None

def to_frame(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)): return np.asarray(Image.open(io.BytesIO(bytes(x))).convert("RGB"))
    x = np.asarray(x); return x.transpose(1, 2, 0) if (x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3) else x

def auc(score, ok):  # AUC of "low score predicts success"
    s, o = np.asarray(score, float), np.asarray(ok, bool)
    if o.all() or (~o).all(): return None
    pos, neg = s[o], s[~o]
    return float((pos[:, None] < neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

from rlp.core.value import load_metric
from rlp.core.world_model import load_pretrained
from rlp.core.rollout import rollout_traj
import stable_pretraining as spt
from torchvision.transforms import v2 as T
wm = load_pretrained(str(D / "lewm_pusht_official")).to(dev).eval(); wm.requires_grad_(False)
ck = torch.load(D / "train_checkpoints" / "planner.pt", map_location="cpu", weights_only=False)
W = int(ck.get("window_frames") or 1)
crit = {"ac": load_metric(str(D / "train_checkpoints" / "value_ac"), device=dev), "td": load_metric(str(D / "train_checkpoints" / "value_td"), device=dev)}
stats = spt.data.dataset_stats.ImageNet
tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=stats["mean"], std=stats["std"]), T.Resize(224)])
BLOCK, H = 5, 5
SEEDS = sorted({int(re.search(r"_s(\d+)\.txt$", f.name).group(1)) for f in D.glob("results_rlp_s*.txt")})

@torch.no_grad()
def enc(frames):
    px = torch.stack([tf(to_frame(f)) for f in frames]).to(dev).unsqueeze(1)
    return wm.encode({"pixels": px})["emb"][:, 0].float()

@torch.no_grad()
def window_E(name, traj, zg):  # traj (H,D) newest last -> W-frame window energy
    cols = [traj[max(len(traj) - 1 - k, 0)] for k in range(W - 1, -1, -1)]
    return float(crit[name](torch.cat(cols)[None], zg[None].repeat(1, W)).reshape(-1)[0])

@torch.no_grad()
def static_E(name, z_last, zg):
    return float(crit[name](z_last[None].repeat(1, W), zg[None].repeat(1, W)).reshape(-1)[0])

# action normalisation (StandardScaler on the h5, as the eval does)
amu = asd = None
if H5 and os.path.exists(H5):
    import h5py
    with h5py.File(H5, "r") as h:
        act = np.asarray(h["action"][:], dtype=np.float64); act = act[~np.isnan(act).any(1)]; amu, asd = act.mean(0), act.std(0)
        epi = np.asarray(h["episode_idx"][:]).reshape(-1); stp = np.asarray(h["step_idx"][:]).reshape(-1)
        lo, hi, off, n = 16000, 18685, 25, 50
        ep_ids = np.unique(epi); ep_ids = ep_ids[(ep_ids >= lo) & (ep_ids < hi)]
        maxlen = {int(e): int(stp[epi == e].max()) for e in ep_ids}
        keep = np.isin(epi, ep_ids); mx = np.zeros(len(epi), dtype=np.int64); mx[keep] = np.array([maxlen[int(e)] for e in epi[keep]])
        valid = np.nonzero(keep & (stp <= mx - off))[0]
        GOALS, STARTS = {}, {}
        for s in SEEDS:
            rows = np.sort(np.random.default_rng(s).choice(valid, size=n, replace=False))
            GOALS[s] = [to_frame(h["pixels"][int(r) + off]) for r in rows]; STARTS[s] = [to_frame(h["pixels"][int(r)]) for r in rows]

# ---------------------------------------------------------------- E5
import lance
e5 = {}
for lab in ("rlp", "cem_latent"):
    for s in SEEDS:
        p = D / f"rec_{lab}_s{s}.lance"
        if not p.exists() or s not in GOALS: continue
        ok = successes(lab, s); zg_all = enc(GOALS[s]); starts = np.stack([np.asarray(f, np.float32) for f in STARTS[s]])
        ds = lance.dataset(str(p)); t = ds.to_table(columns=["episode_idx", "step_idx", "pixels", "action"])
        ep = t.column(0).to_numpy().reshape(-1); st = t.column(1).to_numpy().reshape(-1)
        pix = np.asarray(t.column(2).to_pylist(), dtype=object); act = t.column(3).to_numpy(zero_copy_only=False)
        rows_out = []
        for k in np.unique(ep):
            m = ep == k; order = np.argsort(st[m]); frames = [pix[m][order][j] for j in range(int(m.sum()))]
            if len(frames) < H * BLOCK + 1: continue
            f0 = np.asarray(to_frame(frames[0]), np.float32); e = int(np.argmin(((starts - f0) ** 2).mean(axis=(1, 2, 3))))
            a = np.stack(act[m][order]).astype(np.float64)[: H * BLOCK]
            a = (a - amu) / asd if amu is not None else a
            z_real = enc([frames[j] for j in range(0, H * BLOCK + 1, BLOCK)])  # (H+1,D)
            A = torch.as_tensor(a, dtype=torch.float32, device=dev).reshape(1, H, -1)
            z_imag = rollout_traj(wm, z_real[0:1].unsqueeze(0).expand(1, 3, -1), torch.zeros(1, 2, A.shape[-1], device=dev), A)[0]
            zg = zg_all[e]
            r = {"task": e, "ok": bool(ok[e]) if ok is not None else None}
            for nm in ("ac", "td"):
                r[f"E_real_{nm}"] = window_E(nm, z_real[1:], zg); r[f"E_imag_{nm}"] = window_E(nm, z_imag, zg)
                r[f"E_static_imag_{nm}"] = static_E(nm, z_imag[-1], zg)
            rows_out.append(r)
        if not rows_out: continue
        okv = np.array([r["ok"] for r in rows_out], bool)
        summ = {"n": len(rows_out), "n_ok": int(okv.sum())}
        for nm in ("ac", "td"):
            Er = np.array([r[f"E_real_{nm}"] for r in rows_out]); Ei = np.array([r[f"E_imag_{nm}"] for r in rows_out]); Es = np.array([r[f"E_static_imag_{nm}"] for r in rows_out])
            summ[nm] = {"E_real_ok": float(Er[okv].mean()) if okv.any() else None, "E_real_fail": float(Er[~okv].mean()) if (~okv).any() else None,
                        "E_imag_ok": float(Ei[okv].mean()) if okv.any() else None, "E_imag_fail": float(Ei[~okv].mean()) if (~okv).any() else None,
                        "optimism_gap_imag_minus_real_ok": float((Ei - Er)[okv].mean()) if okv.any() else None,
                        "optimism_gap_imag_minus_real_fail": float((Ei - Er)[~okv].mean()) if (~okv).any() else None,
                        "static_minus_window_imag_ok": float((Es - Ei)[okv].mean()) if okv.any() else None,
                        "static_minus_window_imag_fail": float((Es - Ei)[~okv].mean()) if (~okv).any() else None,
                        "auc_E_real_predicts_success": auc(Er, okv), "auc_E_imag_predicts_success": auc(Ei, okv)}
        e5[f"{lab}_s{s}"] = summ
log("E5_calibration", e5)

# ---------------------------------------------------------------- E8 (probes: t=0 plan of the rlp arm)
e8 = {}
for s in SEEDS:
    files = sorted(glob.glob(str(D / f"probes_rlp_s{s}" / "probe_*.pt")))
    r, c = successes("rlp", s), successes("cem_latent", s)
    if not files or r is None or c is None: continue
    p0 = torch.load(files[0], map_location="cpu", weights_only=False)
    zt, zg = p0["z_traj"].float().to(dev), p0["zg"].float().to(dev)
    cls = np.where(r, "rlp_ok", np.where(c, "modeA", "both_fail"))
    out = {}
    for nm in ("ac", "td"):
        Ew = np.array([window_E(nm, zt[i], zg[i]) for i in range(len(zt))]); Es = np.array([static_E(nm, zt[i, -1], zg[i]) for i in range(len(zt))])
        out[nm] = {k: {"E_window": float(Ew[cls == k].mean()), "E_static": float(Es[cls == k].mean()), "static_minus_window": float((Es - Ew)[cls == k].mean()), "n": int((cls == k).sum())} for k in ("rlp_ok", "modeA", "both_fail") if (cls == k).any()}
        out[nm]["auc_window_predicts_rlp_success"] = auc(Ew, r); out[nm]["auc_static_predicts_rlp_success"] = auc(Es, r)
    out["teacher_vs_cotrained_corr"] = float(np.corrcoef([window_E("ac", zt[i], zg[i]) for i in range(len(zt))], [window_E("td", zt[i], zg[i]) for i in range(len(zt))])[0, 1])
    e8[f"s{s}"] = out
log("E8_velocity_channel", e8)
(D / "critic_probe.json").write_text(json.dumps(OUT, indent=1)); print("[probe] DONE", flush=True)
