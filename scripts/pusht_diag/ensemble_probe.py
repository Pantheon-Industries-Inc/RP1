"""E7: epistemic disagreement of TD teachers at RLP's imagined terminal states.
Env: D (diag tag with probes_rlp_s*/ and results), TEACHERS (space-separated
value_td dirs), optional H5. For every t=0 plan of the rlp arm: window energy
under each teacher; std across teachers by class (rlp_ok / modeA / both_fail);
AUC of the std for separating mode-A from RLP-ok; and the same for the mean.
"""
from __future__ import annotations
import glob, json, os, re
from pathlib import Path
import numpy as np, torch
from rlp.core.value import load_metric
D = Path(os.environ["D"]); dev = "cuda" if torch.cuda.is_available() else "cpu"
teachers = [load_metric(p, device=dev) for p in os.environ["TEACHERS"].replace(":", " ").split()]
ck = torch.load(D / "train_checkpoints" / "planner.pt", map_location="cpu", weights_only=False); W = int(ck.get("window_frames") or 1)
D_LAT = int(ck["z_dim"])
# teachers may differ in window width (latent_dim = W_t * D); each scores with
# its own window, disagreement is computed WITHIN groups of equal width
def width(v): return max(int(getattr(v, "latent_dim", D_LAT)) // D_LAT, 1)
groups = {}
for v in teachers: groups.setdefault(width(v), []).append(v)
print(f"[e7] teacher window widths: {[width(v) for v in teachers]}", flush=True)
def successes(label, seed):
    f = D / f"results_{label}_s{seed}.txt"
    m = re.search(r"'episode_successes':\s*array\((\[.*?\])", f.read_text(), re.S) if f.exists() else None
    return np.array([t == "True" for t in re.findall(r"True|False", m.group(1))]) if m else None
def auc(score, pos):
    s, o = np.asarray(score, float), np.asarray(pos, bool)
    if o.all() or (~o).all(): return None
    return float((s[o][:, None] > s[~o][None, :]).mean() + 0.5 * (s[o][:, None] == s[~o][None, :]).mean())
@torch.no_grad()
def wE(v, traj, zg):
    Wt = width(v)
    cols = [traj[max(len(traj) - 1 - k, 0)] for k in range(Wt - 1, -1, -1)]
    return float(v(torch.cat(cols)[None], zg[None].repeat(1, Wt)).reshape(-1)[0])
out = {"n_teachers": len(teachers)}
for s in sorted({int(re.search(r"_s(\d+)$", p).group(1)) for p in glob.glob(str(D / "probes_rlp_s*"))}):
    files = sorted(glob.glob(str(D / f"probes_rlp_s{s}" / "probe_*.pt"))); r, c = successes("rlp", s), successes("cem_latent", s)
    if not files or r is None or c is None: continue
    p0 = torch.load(files[0], map_location="cpu", weights_only=False); zt, zg = p0["z_traj"].float().to(dev), p0["zg"].float().to(dev)
    cls = np.where(r, "rlp_ok", np.where(c, "modeA", "both_fail")); m = (cls == "rlp_ok") | (cls == "modeA")
    for Wt, vs in groups.items():
        if len(vs) < 2: continue
        E = np.array([[wE(v, zt[i], zg[i]) for i in range(len(zt))] for v in vs])  # (T,B)
        sd, mu = E.std(0), E.mean(0)
        res = {k: {"std_mean": float(sd[cls == k].mean()), "E_mean": float(mu[cls == k].mean()), "n": int((cls == k).sum())} for k in ("rlp_ok", "modeA", "both_fail") if (cls == k).any()}
        res["n_teachers"] = len(vs); res["window"] = Wt
        res["auc_std_modeA_vs_ok"] = auc(sd[m], cls[m] == "modeA"); res["auc_mean_modeA_vs_ok"] = auc(mu[m], cls[m] == "modeA")
        res["corr_std_vs_meanE"] = float(np.corrcoef(sd, mu)[0, 1])
        out[f"s{s}_w{Wt}"] = res
        print(f"[e7] draw {s} window {Wt}: " + json.dumps(res), flush=True)
(D / "ensemble_probe.json").write_text(json.dumps(out, indent=1)); print("[e7] DONE", flush=True)
