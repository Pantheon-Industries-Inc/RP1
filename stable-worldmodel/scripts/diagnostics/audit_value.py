"""Planner-free audit of a learned TRM value (reacher).

Turns a 62-minute planning verdict into a ~1-minute screen, in the spirit of the
TwoRoom SCSA audit (scripts/trm/eval_trm.py).

AUDIT 1 -- distance fidelity vs true steps-to-go.
    Within an episode the ground-truth cost-to-go between two frames is just the
    index gap, so no env oracle is needed. We score V(z_a, z_{t+k}) over a wide
    k range and report Spearman(V, k) plus the monotonicity curve, in TWO
    conditions that differ ONLY in where the first argument comes from:
      (a) enc->enc  : z_a = encoder latent            <- the value's TRAINING distribution
      (b) pred->enc : z_a = predictor rollout output  <- the DEPLOYMENT distribution
    The metric hook at plan time calls cost(predicted_emb, goal_emb), i.e. (b).
    If (a) is healthy and (b) collapses, the value is fine but is being evaluated
    off-distribution -- which is a property of the world model, not the value.
    Also reports ||pred - enc|| (the predictor offset) as the mechanism.

AUDIT 2 -- agreement with the cost we KNOW plans well.
    Plain CEM on the raw flat-latent terminal MSE scores 72.0/94.0 on this WM, so
    that cost is a working reference ranker. On a shared candidate set (random
    action plans rolled through the WM exactly as the planner does) we rank the
    candidates by (a) raw-latent terminal MSE and (b) the learned value, and
    report Spearman + top-1 agreement.

Held-out: episodes are drawn from the LAST --heldout-eps episode ids. The old
DINO value was fit on a 200k-row cache = the FIRST 1000 episodes, so that slice
is genuinely unseen for it; the lejepa/pldm values were fit on all 2M rows, so
for those the audit is in-sample (stated in the output).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

import stable_worldmodel as swm
from stable_worldmodel.trm import load_metric
from stable_worldmodel.wm.utils import load_pretrained

K_LIST = [1, 2, 3, 5, 8, 12, 20, 30, 50]  # fs1 primitive steps


def build_featurizer(wm, device, img_size=224):
    """Reuse the EXACT transform the latent cache was built with, so the audit
    feeds the value the same encoder distribution it was trained on."""
    import sys

    sys.path.insert(0, "/workspace/stable-worldmodel/scripts/trm")
    from _common import build_featurizer as _bf

    base = _bf(wm, device=device, img_size=img_size)
    wants_proprio = bool(getattr(wm, "wants_proprio", False))

    @torch.no_grad()
    def featurize(rows):
        if wants_proprio and "proprio" not in rows:
            n = len(rows["pixels"])
            rows = dict(rows)
            rows["proprio"] = np.concatenate(
                [np.asarray(rows["qpos"], np.float32).reshape(n, -1),
                 np.asarray(rows["qvel"], np.float32).reshape(n, -1)], axis=1)
        return base(rows)

    return featurize


def encode_rows(ds, featurize, rows, batch=128):
    out = []
    for s in range(0, len(rows), batch):
        chunk = [int(r) for r in rows[s:s + batch]]
        out.append(featurize(ds.get_row_data(chunk)))
    return torch.cat(out, 0)


def rollout(wm, z_hist, plan, a_hist):
    if hasattr(wm, "rollout_traj"):
        return wm.rollout_traj(z_hist, plan, a_hist)
    embs, acts, outs = list(z_hist.unbind(1)), list(a_hist.unbind(1)), []
    hs = z_hist.shape[1]
    for t in range(plan.shape[1]):
        acts.append(plan[:, t])
        nxt = wm.predict(torch.stack(embs[-hs:], 1),
                         wm.action_encoder(torch.stack(acts[-hs:], 1)))[:, -1]
        embs.append(nxt)
        outs.append(nxt)
    return torch.stack(outs, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--dataset", default="dmc/reacher_random.lance")
    p.add_argument("--n-start", type=int, default=192)
    p.add_argument("--n-start-audit2", type=int, default=48)
    p.add_argument("--n-cand", type=int, default=64)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--heldout-eps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    label = args.label or args.metric.split("/")[-1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    wm = load_pretrained(args.wm).to(dev).eval()
    wm.requires_grad_(False)
    metric = load_metric(args.metric, device=dev)
    featurize = build_featurizer(wm, dev)

    ae = wm.action_encoder
    has_ext = hasattr(ae, "ext_mu")
    if has_ext:
        a_dim = int(ae.ext_mu.shape[0])
    else:  # bare Embedder (lejepa/pldm bases): infer block width from its input layer
        a_dim = int(next(m.in_features for m in ae.modules() if isinstance(m, torch.nn.Linear)))
    ds = swm.data.load_dataset(args.dataset)
    ep = np.asarray(ds.get_col_data("episode_idx")).reshape(-1).astype(np.int64)
    st = np.asarray(ds.get_col_data("step_idx")).reshape(-1).astype(np.int64)
    act = np.asarray(ds.get_col_data("action"), dtype=np.float32).reshape(len(ep), -1)
    raw_a_dim = act.shape[1]
    FS = a_dim // raw_a_dim
    H = args.horizon
    hs = int(getattr(wm, "history_size", 3))

    uniq = np.unique(ep)
    held = uniq[-args.heldout_eps:]
    ep_start = {int(e): int(np.nonzero(ep == e)[0][0]) for e in held}
    ep_len = {int(e): int((ep == e).sum()) for e in held}
    L = min(ep_len.values())
    print(f"== {label}  wm={args.wm.split('/')[-1]}  metric_dim_in="
          f"{next(m.in_features for m in metric.modules() if isinstance(m, torch.nn.Linear))}")
    print(f"   held-out episodes {held[0]}..{held[-1]} (n={len(held)}), len={L}, "
          f"fs={FS}, history={hs}, a_dim={a_dim}")

    if has_ext:
        # _BlockActionEncoder undoes exactly this affine, so the WM sees the raw
        # action regardless of what convention ext_* encodes
        ext_mu = ae.ext_mu.detach().cpu().numpy()
        ext_std = ae.ext_std.detach().cpu().numpy()
    else:
        # bare Embedder: caller must z-score with the dataset's action stats
        # (same recipe as scripts/plan/train_lip_ac.py)
        amu, astd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-6
        ext_mu = np.tile(amu, FS).astype(np.float32)
        ext_std = np.tile(astd, FS).astype(np.float32)
    print(f"   action norm: {'ckpt ext_*' if has_ext else 'dataset stats'} "
          f"mu[:2]={np.round(ext_mu[:2], 4).tolist()} std[:2]={np.round(ext_std[:2], 4).tolist()}")

    def block(e, t):  # z-scored action block starting at primitive step t
        s = ep_start[e] + t
        raw = act[s:s + FS].reshape(-1)
        return (raw - ext_mu) / ext_std

    # ---- sample start states -------------------------------------------------
    t_lo, t_hi = hs * FS, L - (FS + max(K_LIST) + 2)
    starts = []
    for _ in range(args.n_start):
        e = int(held[rng.integers(len(held))])
        t = int(rng.integers(t_lo // FS, t_hi // FS)) * FS
        starts.append((e, t))

    # rows we need
    need = []
    for e, t in starts:
        for j in range(hs):
            need.append(ep_start[e] + t - (hs - 1 - j) * FS)
        need.append(ep_start[e] + t + FS)                       # encoder counterpart
        for k in K_LIST:
            need.append(ep_start[e] + t + FS + k)               # goals
    need = np.array(need)
    uniq_rows, inv = np.unique(need, return_inverse=True)
    print(f"   encoding {len(uniq_rows)} frames ...")
    Z = encode_rows(ds, featurize, uniq_rows)
    row_of = {int(r): i for i, r in enumerate(uniq_rows)}

    def zi(e, t):
        return Z[row_of[ep_start[e] + t]]

    # ---- AUDIT 1 -------------------------------------------------------------
    z_hist = torch.stack([torch.stack([zi(e, t - (hs - 1 - j) * FS) for j in range(hs)])
                          for e, t in starts])                                  # (N,hs,D)
    a_hist = torch.tensor(np.stack([[block(e, t - (hs - 1 - j) * FS) for j in range(hs - 1)]
                                    for e, t in starts]), dtype=torch.float32, device=dev)
    plan1 = torch.tensor(np.stack([[block(e, t)] for e, t in starts]),
                         dtype=torch.float32, device=dev)
    with torch.no_grad():
        pred = rollout(wm, z_hist, plan1, a_hist)[:, -1]        # (N, D) predictor space
    enc = torch.stack([zi(e, t + FS) for e, t in starts])        # (N, D) encoder space

    off = (pred - enc)
    print(f"   ||pred - enc|| mean={off.norm(dim=-1).mean():.2f}   "
          f"||enc||={enc.norm(dim=-1).mean():.2f}   "
          f"mean-offset norm ||E[pred-enc]||={off.mean(0).norm():.2f}   "
          f"cos(off_i, mean_off)={torch.nn.functional.cosine_similarity(off, off.mean(0, keepdim=True)).mean():.3f}")

    res = {}
    for cond, first in (("a_enc->enc", enc), ("b_pred->enc", pred)):
        vals, ks = [], []
        curve = []
        with torch.no_grad():
            for k in K_LIST:
                goals = torch.stack([zi(e, t + FS + k) for e, t in starts])
                v = metric.cost(first, goals).float().cpu().numpy()
                vals.append(v)
                ks.append(np.full(len(v), k))
                curve.append(float(np.mean(v)))
        v_all, k_all = np.concatenate(vals), np.concatenate(ks)
        rho = float(spearmanr(v_all, k_all).statistic)
        res[cond] = (rho, curve)
        print(f"   AUDIT1 {cond:12s} Spearman(V, true k) = {rho:+.3f}   "
              f"V(k): " + " ".join(f"{c:.2f}" for c in curve))

    # cross-episode control (should be LARGER than same-episode far pairs)
    other = []
    for i, (e, t) in enumerate(starts):
        e2 = int(held[rng.integers(len(held))])
        while e2 == e:
            e2 = int(held[rng.integers(len(held))])
        other.append(e2)
    far_rows = np.array([ep_start[e] + t + FS + max(K_LIST) for e, t in starts])
    cross_rows = np.array([ep_start[e2] + int(rng.integers(0, L)) for e2 in other])
    extra = np.unique(np.concatenate([far_rows, cross_rows]))
    Ze = encode_rows(ds, featurize, extra)
    erow = {int(r): i for i, r in enumerate(extra)}
    with torch.no_grad():
        v_far = metric.cost(enc, torch.stack([Ze[erow[int(r)]] for r in far_rows])).float().cpu().numpy()
        v_x = metric.cost(enc, torch.stack([Ze[erow[int(r)]] for r in cross_rows])).float().cpu().numpy()
    sep = float(v_x.mean() - v_far.mean())
    print(f"   AUDIT1 cross-episode: V(cross)={v_x.mean():.2f} vs V(same,k={max(K_LIST)})="
          f"{v_far.mean():.2f}  separation={sep:+.2f}  P[cross>far]={float((v_x > v_far).mean()):.3f}")
    del Ze

    # ---- AUDIT 2 -------------------------------------------------------------
    n2 = min(args.n_start_audit2, len(starts))
    C = args.n_cand
    rhos, top1 = [], []
    pr_v, pr_m, best_v = [], [], []
    for i in range(n2):
        e, t = starts[i]
        goal_row = ep_start[e] + t + FS * H
        with torch.no_grad():
            goal = featurize(ds.get_row_data([int(goal_row)]))[0]
            raw = rng.uniform(-1.0, 1.0, size=(C, H, a_dim)).astype(np.float32)
            cand = (raw - ext_mu) / ext_std
            # AUDIT 3: candidate 0 is the data's TRUE plan, which by construction
            # is the one that actually arrives at the goal frame -> ceiling-free
            # ground truth for "does the value pick the right plan?"
            true_plan = np.stack([block(e, t + k * FS) for k in range(H)])[None]
            plan = torch.tensor(np.concatenate([true_plan, cand], 0),
                                dtype=torch.float32, device=dev)
            zh = z_hist[i:i + 1].expand(C + 1, -1, -1)
            ah = a_hist[i:i + 1].expand(C + 1, -1, -1)
            term = rollout(wm, zh, plan, ah)[:, -1]              # (C+1, D)
            g = goal.unsqueeze(0).expand_as(term)
            mse = ((term - g) ** 2).mean(-1).float().cpu().numpy()
            vv = metric.cost(term, g).float().cpu().numpy()
        rhos.append(float(spearmanr(mse[1:], vv[1:]).statistic))
        top1.append(float(int(np.argmin(mse[1:])) == int(np.argmin(vv[1:]))))
        pr_v.append(float((vv < vv[0]).sum()) / C)   # percentile rank of true plan
        pr_m.append(float((mse < mse[0]).sum()) / C)
        best_v.append(float(int(np.argmin(vv)) == 0))
    print(f"   AUDIT2 vs raw-latent-MSE ranking ({n2} states x {C} cands): "
          f"Spearman={np.mean(rhos):+.3f} (sd {np.std(rhos):.3f})  "
          f"top1-agree={np.mean(top1):.3f}")
    print(f"   AUDIT3 true-plan rank (0=best): value={np.mean(pr_v):.3f}  "
          f"latent-MSE={np.mean(pr_m):.3f}  value-picks-true-plan={np.mean(best_v):.3f}")
    print(f"RESULT\t{label}\t{res['a_enc->enc'][0]:.3f}\t{res['b_pred->enc'][0]:.3f}\t"
          f"{sep:+.2f}\t{np.mean(rhos):.3f}\t{np.mean(top1):.3f}\t"
          f"{np.mean(pr_v):.3f}\t{np.mean(best_v):.3f}")


if __name__ == "__main__":
    main()
