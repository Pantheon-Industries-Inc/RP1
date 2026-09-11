"""Did the plan pass through the success set and drift out, or never reach it?

The E18 atlas decoded poses from latents, but the probe's agent error (21 px
median) exceeds the success tolerance (20 px), so it cannot answer that. This
script uses the env's OWN pose fields, now recorded alongside the pixels, and
evaluates the environment's exact test at every step:

    success  <=>  ||goal[:4] - state[:4]|| < 20  AND  |goal_angle - angle| < pi/9

(no angle wrapping -- `PushT.eval_state` compares raw angles, so an unwrapped
2*pi offset counts as a miss).

Per task it reports the trajectory of that distance and angle error, the best
point and when it occurred, whether the success set was ever entered, and
where the replan boundaries fall (every horizon*block primitive steps), so a
"reached then left" can be told apart from "never reached", and a drift inside
one plan from a drift across a replan.

Env: D, H5, OUT (default $D/drift), DRAW, LABELS, RH (executed blocks per
replan, default 5), BLOCK (5).
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

with __import__("contextlib").suppress(ImportError):
    import hdf5plugin  # noqa: F401

D = Path(os.environ["D"])
H5 = os.environ["H5"]
OUT = Path(os.environ.get("OUT", str(D / "drift")))
DRAW = int(os.environ.get("DRAW", "43"))
LABELS = os.environ.get("LABELS", "rlp cem_latent").split()
RH = int(os.environ.get("RH", "5"))
BLOCK = int(os.environ.get("BLOCK", "5"))
OFF, TOL_POS, TOL_ANG = 25, 20.0, np.pi / 9
OUT.mkdir(parents=True, exist_ok=True)


def log(m: str) -> None:
    print(f"[drift] {m}", flush=True)


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
    start_frames = [to_frame(h["pixels"][int(r)]) for r in rows]

import lance  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

starts = np.stack([np.asarray(f, dtype=np.float32) for f in start_frames])


def load(label):
    p = D / f"rec_{label}_s{DRAW}.lance"
    if not p.exists():
        log(f"no recording for {label}")
        return {}
    cols = [c for c in ("episode_idx", "step_idx", "pixels", "pos_agent", "block_pose") ]
    t = lance.dataset(str(p)).to_table(columns=cols)
    have = set(t.column_names)
    if not {"pos_agent", "block_pose"} <= have:
        log(f"{label}: recording lacks exact pose columns ({sorted(have)}) -- rerun the eval with the patched recorder")
        return {}
    ep = t.column("episode_idx").to_numpy().reshape(-1)
    st = t.column("step_idx").to_numpy().reshape(-1)
    pix = np.asarray(t.column("pixels").to_pylist(), dtype=object)
    ag = np.stack(t.column("pos_agent").to_numpy(zero_copy_only=False)).astype(np.float64)
    bp = np.stack(t.column("block_pose").to_numpy(zero_copy_only=False)).astype(np.float64)
    eps = {}
    for e in np.unique(ep):
        m = ep == e
        o = np.argsort(st[m])
        eps[int(e)] = (to_frame(pix[m][o][0]), np.column_stack([ag[m][o], bp[m][o]]))  # (T, 5): ax ay bx by ang
    ok = successes(label)
    ks = sorted(eps)
    cost = np.zeros((len(ks), len(starts)))
    for a, k in enumerate(ks):
        f0 = np.asarray(eps[k][0], dtype=np.float32)
        cost[a] = ((starts - f0) ** 2).mean(axis=(1, 2, 3))
        if ok is not None and len(eps[k][1]) < 50:
            for e in range(len(starts)):
                if not ok[e]:
                    cost[a, e] += 1e6
    r_, c_ = linear_sum_assignment(cost)
    return {int(e): eps[ks[a]][1] for a, e in zip(r_, c_) if cost[a, e] < 1e5}


report: dict[str, object] = {"draw": DRAW, "replan_every": RH * BLOCK, "tasks": {}}
for lab in LABELS:
    roll = load(lab)
    ok = successes(lab)
    if not roll:
        continue
    log(f"{lab}: {len(roll)} episodes with exact poses; success {None if ok is None else round(100 * ok.mean(), 1)}")
    reached_but_failed, never_reached = [], []
    for e, traj in sorted(roll.items()):
        g = s_goal[e]
        dist = np.linalg.norm(traj[:, :4] - g[:4], axis=1)
        ang = np.abs(traj[:, 4] - g[4])
        inside = (dist < TOL_POS) & (ang < TOL_ANG)
        best = int(np.argmin(dist))
        entry = {
            "ok": bool(ok[e]) if ok is not None else None,
            "steps": int(len(traj)),
            "min_success_distance_px": round(float(dist.min()), 1),
            "step_of_min": best,
            "angle_err_at_min_deg": round(float(np.degrees(ang[best])), 1),
            "final_distance_px": round(float(dist[-1]), 1),
            "final_angle_err_deg": round(float(np.degrees(ang[-1])), 1),
            "ever_inside_success_set": bool(inside.any()),
            "first_inside_step": int(np.argmax(inside)) if inside.any() else None,
            "distance_every_block": [round(float(v), 1) for v in dist[:: BLOCK]],
        }
        report["tasks"].setdefault(str(e), {})[lab] = entry
        if ok is not None and not ok[e]:
            (reached_but_failed if inside.any() else never_reached).append(e)
    log(f"{lab}: failures that DID enter the success set at some step: {reached_but_failed}")
    log(f"{lab}: failures that never entered it: {never_reached}")

# focus: rlp failures, and how their best point compares with the samplers'
rlp_ok = successes("rlp")
if rlp_ok is not None:
    fails = [e for e in range(len(rows)) if not rlp_ok[e] and str(e) in report["tasks"] and "rlp" in report["tasks"][str(e)]]
    log("task | rlp min-dist (step) | rlp final | angle at min | other planners' min")
    for e in fails:
        r = report["tasks"][str(e)]["rlp"]
        others = " ".join(
            f"{lab}:{report['tasks'][str(e)][lab]['min_success_distance_px']}({report['tasks'][str(e)][lab]['ok']})"
            for lab in LABELS
            if lab != "rlp" and lab in report["tasks"][str(e)]
        )
        log(
            f"  {e:2d} | {r['min_success_distance_px']:6.1f} @ step {r['step_of_min']:2d} | "
            f"{r['final_distance_px']:6.1f} | {r['angle_err_at_min_deg']:5.1f} deg | {others}"
        )
    drifted = [e for e in fails if report["tasks"][str(e)]["rlp"]["final_distance_px"] > report["tasks"][str(e)]["rlp"]["min_success_distance_px"] + 10]
    within = [e for e in drifted if report["tasks"][str(e)]["rlp"]["step_of_min"] % (RH * BLOCK) != 0]
    log(f"failures whose distance grew >10 px after its minimum: {drifted} (minimum inside a plan, not at a replan boundary: {within})")
    log(f"failures that entered the success set yet were scored a miss: {[e for e in fails if report['tasks'][str(e)]['rlp']['ever_inside_success_set']]}")
    report["summary"] = {
        "rlp_failures": fails,
        "drifted_after_minimum": drifted,
        "entered_success_set_but_failed": [e for e in fails if report["tasks"][str(e)]["rlp"]["ever_inside_success_set"]],
        "median_min_distance_of_failures": round(float(np.median([report["tasks"][str(e)]["rlp"]["min_success_distance_px"] for e in fails])), 1) if fails else None,
    }

(OUT / "drift_analysis.json").write_text(json.dumps(report, indent=1))
b = base64.b64encode((OUT / "drift_analysis.json").read_bytes()).decode()
for k in range(0, len(b), 3000):
    print(f"[drift-b64] drift_analysis.json {k // 3000} {(len(b) + 2999) // 3000} {b[k : k + 3000]}", flush=True)
log("DONE")
