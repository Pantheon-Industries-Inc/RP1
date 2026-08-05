"""Is LIP's low HELD-at-end the VALUE or the PLANNER?

Held requires arriving AND staying; ever-in-ball is ~90 for every arm while held
is 12-54, so the entire deficit is settling. This asks whether the costs we plan
with can even SEE settling, on held-out data, with no env and no planner in the
loop -- so a failure here is unambiguously the value's.

For each cost, over episodes >= 8000:
  T1  stopped-vs-moving AT the goal. Pairs sharing a goal configuration (worst-
      joint qpos within TOL of the goal row) split by |qvel|: does the cost rank
      the STOPPED one cheaper? AUC 0.5 = blind. This is the at-rest test, now for
      window costs (the 1-frame quasimetric scored 0.496 = chance).
  T2  held-vs-passed discrimination on REAL futures: for a state at the goal now,
      does the cost predict whether the arm is STILL at the goal 25 steps later?
      That is precisely what held-at-end scores.
  T3  dynamic range over true separation (a flat cost cannot rank anything).

Costs compared: window3 (learned, what LIP distils), l2window3 (unlearned), and
the 1-frame quasimetric if present.
"""

import argparse
import json

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from stable_worldmodel.trm import LatentCache, load_metric

p = argparse.ArgumentParser()
p.add_argument("--cache", default="/workspace/caches/canon_lejepa_fs1.pt")
p.add_argument("--h5", default="/workspace/datasets_canon/lewm-reacher/reacher.h5")
p.add_argument("--split", type=int, default=8000)
p.add_argument("--lag", type=int, default=5)
p.add_argument("--tol", type=float, default=0.05)
p.add_argument("--out", default="/workspace/results/diag_value_vs_planner.json")
p.add_argument("--device", default="cuda")
a = p.parse_args()
dev = a.device

c = LatentCache.load(a.cache)
Z, ep, st = c.z, c.episode_idx.numpy(), c.step_idx.numpy()
Q = c.state.float().numpy()
with h5py.File(a.h5, "r") as f:
    QV = f["qvel"][:].astype(np.float32)
assert len(QV) == len(Z), f"qvel {len(QV)} vs cache {len(Z)}"
speed = np.linalg.norm(QV, axis=1)
val = ep >= a.split
print(f"rows {len(Z)}  held-out {val.sum()}  |qvel| median {np.median(speed[val]):.3f}")

D = Z.shape[1]


def window(idx):
    """3-frame window [t-2L, t-L, t] with episode-start clamping (train_window)."""
    outs = []
    for k in (2, 1, 0):
        off = k * a.lag
        j = idx - off
        clamp = st[idx] < off
        j = np.where(clamp, idx - st[idx], j)
        outs.append(Z[torch.from_numpy(j).long()])
    return torch.cat(outs, dim=1)


COSTS = {}
for name, path, frames in [
    ("window3 (learned, LIP distils this)", "/workspace/metrics/window3_lejepa.pt", 3),
    ("l2window3 (unlearned)", "/workspace/metrics/l2window3.pt", 3),
    ("1-frame quasimetric", "/workspace/metrics/td_canon_lejepa_s0.pt", 1),
]:
    try:
        COSTS[name] = (load_metric(path, device=dev).eval(), frames)
        print(f"loaded {name}")
    except Exception as e:  # noqa: BLE001
        print(f"skip {name}: {type(e).__name__}")


def cost_of(m, frames, cur_idx, goal_idx):
    with torch.no_grad():
        if frames == 3:
            x = window(cur_idx).to(dev)
            g = Z[torch.from_numpy(goal_idx).long()].repeat(1, 3).to(dev)
        else:
            x = Z[torch.from_numpy(cur_idx).long()].to(dev)
            g = Z[torch.from_numpy(goal_idx).long()].to(dev)
        return m.cost(x, g).float().cpu().numpy()


rng = np.random.default_rng(0)
vidx = np.nonzero(val)[0]
report = {}

# ---------------- T1: stopped vs moving, both AT the goal
STOP_Q, MOVE_Q = np.quantile(speed[val], 0.15), np.quantile(speed[val], 0.85)
anchors = rng.choice(vidx, size=min(300000, len(vidx)), replace=False)
pairs = []
for g in rng.choice(vidx, size=4000, replace=False):
    near = anchors[np.abs(Q[anchors] - Q[g]).max(1) < a.tol]
    if len(near) < 2:
        continue
    s_ = near[speed[near] <= STOP_Q]
    m_ = near[speed[near] >= MOVE_Q]
    if len(s_) and len(m_):
        pairs.append((s_[0], m_[0], g))
    if len(pairs) >= 1500:
        break
print(f"\nT1 pairs: {len(pairs)} (stopped |qvel|<={STOP_Q:.2f}, moving >={MOVE_Q:.2f})")
if pairs:
    si = np.array([x[0] for x in pairs]); mi = np.array([x[1] for x in pairs])
    gi = np.array([x[2] for x in pairs])
    for name, (m, fr) in COSTS.items():
        cs, cm = cost_of(m, fr, si, gi), cost_of(m, fr, mi, gi)
        auc = float((cs < cm).mean() + 0.5 * (cs == cm).mean())
        sd = float(np.std(np.concatenate([cs, cm])) + 1e-9)
        report.setdefault("T1_stopped_vs_moving", {})[name] = {
            "auc_prefers_stopped": auc, "gap": float(cm.mean() - cs.mean()),
            "gap_over_sd": float((cm.mean() - cs.mean()) / sd)}
        print(f"  {name:<38} AUC {auc:.3f}  gap/sd {(cm.mean()-cs.mean())/sd:+.2f}")

# ---------------- T2: does the cost predict STILL-at-goal 25 steps later?
H = 25
ok = vidx[(st[vidx] + H < 200)]
at_goal = ok[np.abs(Q[ok] - Q[ok]).max(1) < a.tol]  # trivially true; use goal=self
cur = rng.choice(ok, size=min(6000, len(ok)), replace=False)
goal = cur.copy()                     # goal = the state itself -> "stay here"
future = cur + H
stays = (np.abs(Q[future] - Q[goal]).max(1) < a.tol) & (ep[future] == ep[goal])
print(f"\nT2: n={len(cur)}  actually-still-there rate {stays.mean():.3f}")
for name, (m, fr) in COSTS.items():
    cc = cost_of(m, fr, cur, goal)
    a_, b_ = cc[stays], cc[~stays]
    if len(a_) and len(b_):
        auc = float((a_[:, None] < b_[None, :]).mean())
        report.setdefault("T2_predicts_stay", {})[name] = {
            "auc_stay_cheaper": auc, "mean_stay": float(a_.mean()), "mean_leave": float(b_.mean())}
        print(f"  {name:<38} AUC {auc:.3f}  (stay {a_.mean():.3f} vs leave {b_.mean():.3f})")

# ---------------- T3: dynamic range vs true separation
print("\nT3 mean cost by true separation (blocks):")
for name, (m, fr) in COSTS.items():
    row = {}
    for d in (1, 5, 10, 25, 50):
        i0 = rng.choice(vidx[st[vidx] + d * a.lag < 200], size=2000, replace=False)
        row[d] = float(cost_of(m, fr, i0, i0 + d * a.lag).mean())
    report.setdefault("T3_range", {})[name] = row
    print(f"  {name:<38} " + " ".join(f"d{k}={v:.2f}" for k, v in row.items()))

json.dump(report, open(a.out, "w"), indent=1)
print(f"\nreport -> {a.out}")
print("DIAG_VALUE_VS_PLANNER_DONE")
