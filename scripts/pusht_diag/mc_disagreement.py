"""Does the world model DISAGREE with itself where it hallucinates? (E30 precheck)

Pessimism-by-disagreement can only work if the fake basin -- imagined block
motion with no real push -- is where the model is unsure of its own
prediction. E7 found critic ensembles AGREE on the shortcut (shared target
bias). This asks the same of the world model: for every recorded first plan
(RLP and CEM-latent, draw DRAW), re-imagine the executed plan once
deterministically and K times with the predictor's dropout on, and relate the
spread across passes to what actually went wrong:

  * hallucination  = || decoded imagined block (terminal) - REAL block (exact pose) ||  (px)
  * optimism gap   = V_real - V_imag at the terminal block
  * pushed         = real block displacement over the plan > 10 px
  * contact        = min agent-block gap over the executed steps (exact poses)

and reports Spearman correlations, AUC of disagreement for hallucination > 20 px,
and disagreement split by pushed / not pushed. Env: D, H5, CD, K (8), DRAW (43),
LABELS ("rlp,cem_latent"), CRITIC.
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
OUT = Path(os.environ.get("OUT", str(D / "mcd")))
K = int(os.environ.get("K", "8"))
DRAWS = [int(d) for d in os.environ.get("DRAWS", os.environ.get("DRAW", "43")).split(",")]
DRAW = DRAWS[0]  # rebound per draw below
LABELS = os.environ.get("LABELS", "rlp,cem_latent").split(",")
CRITIC = os.environ.get("CRITIC", str(D / "train_checkpoints" / "value_ac"))
H, BLOCK, OFF = 5, 5, 25
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def log(m: str) -> None:
    print(f"[mcd] {m}", flush=True)


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


def run_draw(draw: int) -> None:
    # rebind the MODULE-level DRAW: helpers defined above (successes) read it as a global
    global DRAW, s_goal, starts, amu, asd, wm, predictor, critic, W, tf, cache, Wt, z_goal_all
    DRAW = draw
    # ---------------------------------------------------------------- eval tasks (same draw rule as the eval)
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
    amu, asd = np.nanmean(act_train, 0), np.nanstd(act_train, 0) + 1e-6
    starts = np.stack([np.asarray(f, dtype=np.float32) for f in start_frames])

    import lance  # noqa: E402
    import stable_pretraining as spt  # noqa: E402
    from scipy.optimize import linear_sum_assignment  # noqa: E402
    from scipy.stats import spearmanr  # noqa: E402
    from torchvision.transforms import v2 as T  # noqa: E402

    from rlp.core.rollout import rollout_traj  # noqa: E402
    from rlp.core.value import load_metric  # noqa: E402
    from rlp.core.world_model import load_pretrained  # noqa: E402
    from rlp.data import LatentCache  # noqa: E402

    wm = load_pretrained(str(D / "lewm_pusht_official")).to(dev).eval()
    wm.requires_grad_(False)
    predictor = wm.predictor
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
    def imagine(z_hist, actions_raw, passes: int = 1):
        """(passes, H, D): pass 0 deterministic; passes > 1 add K-1 dropout rollouts of the SAME plan."""
        a = torch.as_tensor((actions_raw - amu) / asd, dtype=torch.float32, device=dev).reshape(1, H, -1)
        zh = z_hist
        while zh.shape[0] < 3:
            zh = torch.cat([zh[:1], zh], dim=0)
        zh = zh[-3:].unsqueeze(0)
        ah = torch.zeros(1, 2, a.shape[-1], device=dev)
        outs = [rollout_traj(wm, zh, ah, a)[0]]
        if passes > 1:
            predictor.train()
            try:
                outs += [rollout_traj(wm, zh, ah, a)[0] for _ in range(passes - 1)]
            finally:
                predictor.eval()
        return torch.stack(outs)


    # ---------------------------------------------------------------- ridge probe latent -> [ax, ay, bx, by, cos, sin]
    cache = LatentCache.load(str(CD / "counterstrike_fs1.pt"), mmap=True)
    with h5py.File(H5, "r") as h:
        st_z = np.asarray(h["state"][:n16], dtype=np.float32)
    if len(st_z) != len(cache.z):
        raise RuntimeError(f"state rows ({len(st_z)}) do not align with the fs1 cache ({len(cache.z)})")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(cache.z), size=min(200_000, len(cache.z)), replace=False))
    X = cache.z[torch.as_tensor(idx)].float().numpy().astype(np.float64)
    S = st_z[idx].astype(np.float64)
    Y = np.column_stack([S[:, 0], S[:, 1], S[:, 2], S[:, 3], np.cos(S[:, 4]), np.sin(S[:, 4])])
    Xb = np.column_stack([X, np.ones(len(X))])
    Wr = np.linalg.solve(Xb.T @ Xb + 1.0 * np.eye(Xb.shape[1]), Xb.T @ Y)
    Wt = torch.as_tensor(Wr, dtype=torch.float32, device=dev)


    @torch.no_grad()
    def decode(z):  # (T, D) -> (T, 6)
        return (torch.cat([z, torch.ones(len(z), 1, device=dev)], dim=1) @ Wt).cpu().numpy()


    # ---------------------------------------------------------------- recordings with exact poses, matched to tasks
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


    def auc(score, positive):
        """Rank AUC of `score` for the boolean `positive`."""
        score, positive = np.asarray(score, float), np.asarray(positive, bool)
        if positive.all() or not positive.any():
            return float("nan")
        order = np.argsort(score)
        ranks = np.empty(len(score))
        ranks[order] = np.arange(1, len(score) + 1)
        n1, n0 = positive.sum(), (~positive).sum()
        return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


    z_goal_all = encode(goal_frames)
    report: dict[str, object] = {"draw": DRAW, "K": K, "labels": {}}
    for lab in LABELS:
        roll = load(lab)
        ok = successes(lab)
        if not roll:
            continue
        log(f"{lab}: {len(roll)} episodes matched")
        per = []
        for e, (frames, act, pose) in sorted(roll.items()):
            if len(frames) < H * BLOCK + 1:
                continue  # first plan must be fully executed
            z0 = encode(frames[:1])
            zg = z_goal_all[e : e + 1]
            a_exec = act[: H * BLOCK]
            zi = imagine(z0, a_exec, passes=K + 1)  # (K+1, H, D)
            det, drop = zi[0], zi[1:]
            real_idx = [(b + 1) * BLOCK for b in range(H)]
            z_real = encode([frames[i] for i in real_idx])
            # hallucination: decoded imagined terminal block vs the REAL terminal block (exact pose)
            dec_det = decode(det)
            halluc = float(np.linalg.norm(dec_det[-1, 2:4] - pose[real_idx[-1], 2:4]))
            # disagreement: spread of the K dropout passes at the terminal block
            term = drop[:, -1]  # (K, D)
            dis_latent = float(term.std(0).norm())
            dis_v = float(np.std(V(term, zg)))
            dec_k = np.stack([decode(drop[k])[-1] for k in range(K)])  # (K, 6)
            dis_block = float(np.linalg.norm(dec_k[:, 2:4].std(0)))
            v_gap = float(V(z_real[-1:], zg)[0] - V(det[-1:], zg)[0])  # real minus imagined (optimism > 0)
            disp = float(np.linalg.norm(pose[real_idx[-1], 2:4] - pose[0, 2:4]))
            gap = float(np.min(np.linalg.norm(pose[: real_idx[-1] + 1, :2] - pose[: real_idx[-1] + 1, 2:4], axis=1)))
            per.append(
                dict(task=e, ok=(bool(ok[e]) if ok is not None else None), halluc_px=halluc, v_gap=v_gap,
                     dis_latent=dis_latent, dis_v=dis_v, dis_block_px=dis_block, real_disp_px=disp, min_gap_px=gap)
            )
        if not per:
            continue
        A = {k: np.array([p[k] for p in per], dtype=float) for k in per[0] if k not in ("task", "ok")}
        pushed = A["real_disp_px"] > 10
        bad = A["halluc_px"] > 20
        summ = {
            "n": len(per),
            "halluc_px_median": float(np.median(A["halluc_px"])),
            "frac_hallucinating(>20px)": float(bad.mean()),
            "frac_pushed(>10px)": float(pushed.mean()),
            "spearman(dis_block, halluc)": float(spearmanr(A["dis_block_px"], A["halluc_px"]).correlation),
            "spearman(dis_v, v_gap)": float(spearmanr(A["dis_v"], A["v_gap"]).correlation),
            "spearman(dis_latent, halluc)": float(spearmanr(A["dis_latent"], A["halluc_px"]).correlation),
            "auc(dis_block -> halluc>20px)": auc(A["dis_block_px"], bad),
            "auc(dis_v -> halluc>20px)": auc(A["dis_v"], bad),
            "auc(dis_block -> NOT pushed)": auc(A["dis_block_px"], ~pushed),
            "dis_block_px mean | not pushed / pushed": [float(A["dis_block_px"][~pushed].mean()) if (~pushed).any() else None,
                                                        float(A["dis_block_px"][pushed].mean()) if pushed.any() else None],
            "halluc_px mean | not pushed / pushed": [float(A["halluc_px"][~pushed].mean()) if (~pushed).any() else None,
                                                     float(A["halluc_px"][pushed].mean()) if pushed.any() else None],
        }
        report["labels"][lab] = {"summary": summ, "plans": per}
        log(f"{lab}: {json.dumps(summ)}")

    (OUT / f"mc_disagreement_d{DRAW}.json").write_text(json.dumps(report, indent=1))
    b = base64.b64encode((OUT / f"mc_disagreement_d{DRAW}.json").read_bytes()).decode()
    for k in range(0, len(b), 3000):
        print(f"[mcd-b64] mc_disagreement_d{DRAW}.json {k // 3000} {(len(b) + 2999) // 3000} {b[k : k + 3000]}", flush=True)
    log("DONE")


for _d in DRAWS:
    run_draw(_d)
