"""Recover the per-step outcome label for already-collected on-policy lances.

The arm-2 recordings (onpolicy_full_a{0,1,2}.lance) predate the recorder's
`success` column, but the label is reconstructible offline with no replay and
no env (HANDOFF_20260728.md §2.1):

  1. the task draw is deterministic -- eval_wm.py samples
     default_rng(seed).choice(len(valid_indices) - 1, 50, replace=False) over
     the post-ep_range valid rows, and every collection seed is known
     (1000 + a*100 + call);
  2. terminate_at_goal=False means every rollout ran the full 50-step budget,
     so nothing was dropped (verified: all 36 calls log kept=50 dropped=0) and
     recorded episode k of call i is drawn task k of seed 1000+a*100+i;
  3. the recordings carry per-step `qpos`, the block position is a fixed
     3-slice of it, and the slice is discovered from the EXPERT data itself
     (it has both `qpos` and `privileged_block_0_pos` columns) -- no MuJoCo;
  4. the goal is the expert block position at (episode, start + 25), the same
     value cube.yaml's set_target_pos callable applied at collection time;
  5. success at step t = ||block_t - goal|| <= 0.04 (cube_env's threshold),
     written per step; consumers latch with any() (matching how
     episode_successes is scored).

GATES -- all evaluated BEFORE anything is written; on any failure no output
lance is produced:
  A. slice discovery must match privileged_block_0_pos on every probe row;
  B. per-call: arm 1 collected the SAME rollouts with terminate_at_goal=True,
     so its per-call `dropped` count (episodes that ended < 25 steps) is a
     floor on this call's success count -- early termination IS goal-reaching;
  C. totals: arm 1 dropped 1393 of 1800 and its kept set held ~114 late
     successes, so expect ~1507 successes; hard band [1400,1600].

Run on the pod:
  python3 relabel_onpolicy.py \
      --expert /workspace/datasets/ogb_cube_single/ogb_cube_single.lance \
      --onpolicy /workspace/dyna_split/onpolicy_full_a0.lance \
                 /workspace/dyna_split/onpolicy_full_a1.lance \
                 /workspace/dyna_split/onpolicy_full_a2.lance \
      --arm1-logs '/workspace/logs/dsp_col_a{a}_c{c}.log' \
      --ep-range 0:8000 --goal-offset 25 --num-eval 50 --seed-base 1000
Outputs <name>_lab.lance next to each input; originals untouched.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import lance
import numpy as np
import pyarrow as pa

THRESH = 0.04  # cube_env._compute_successes success radius (m)


def fsl_to_np(chunked, n_rows):
    """FixedSizeList/List ChunkedArray -> (n_rows, width) float64."""
    flat = chunked.combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float64).reshape(n_rows, -1)


def reconstruct_draw(ep_col, step_col, ep_lens, seed, lo, hi, goal_offset, num_eval):
    """Replicate eval_wm.py's task sampling exactly.

    ep_col/step_col: per-row episode_idx / step_idx of the EXPERT dataset.
    ep_lens: dict episode_id -> length.
    Returns (episodes, starts) in the sorted-row order the envs were assigned,
    i.e. env j ran task j of this draw.
    """
    keep = np.array(sorted(e for e in ep_lens if lo <= e < hi))
    max_start = {int(e): ep_lens[int(e)] - goal_offset - 1 for e in keep}
    in_range = np.isin(ep_col, keep)
    msr = np.full(len(ep_col), -1, dtype=np.int64)
    msr[in_range] = [max_start[int(e)] for e in ep_col[in_range]]
    valid = np.nonzero(in_range & (step_col <= msr))[0]
    g = np.random.default_rng(seed)
    # the -1 is eval_wm.py's own off-by-one; replicate, don't fix
    rows = np.sort(valid[g.choice(len(valid) - 1, size=num_eval, replace=False)])
    return ep_col[rows].astype(np.int64), step_col[rows].astype(np.int64)


def discover_block_slice(exp, probe=256, seed=0):
    """Find s with expert qpos[:, s:s+3] == privileged_block_0_pos, from data."""
    n = exp.count_rows()
    rows = np.random.default_rng(seed).choice(n, size=probe, replace=False)
    t = exp.take(rows.tolist(), columns=["qpos", "privileged_block_0_pos"])
    q = fsl_to_np(t.column(0), probe)
    p = fsl_to_np(t.column(1), probe)[:, :3]
    for s in range(q.shape[1] - 2):
        if np.allclose(q[:, s : s + 3], p, atol=1e-5):
            print(f"[slice] block position = qpos[{s}:{s + 3}], "
                  f"verified on {probe} expert rows", flush=True)
            return s
    raise SystemExit("GATE A FAILED: no contiguous qpos 3-slice matches "
                     "privileged_block_0_pos; cannot label without it.")


def episode_bounds(ep_col):
    """Episode row-ranges by VALUE CHANGE, in file order.

    Appearance order is what maps to (call, env) -- the writer appends call
    after call, env 0..49 within each. Boundary-by-value-change is correct
    under both writer id conventions (ids continuing across appends, or
    restarting per append), unlike unique(): with restarting ids, episode
    "0" would appear 12 times and first-occurrence logic silently collapses
    them into one 600-step pseudo-episode.
    """
    change = np.nonzero(np.diff(ep_col))[0] + 1
    starts = np.r_[0, change]
    return np.r_[starts, len(ep_col)]


def label_one(path, a, exp, ep_col, step_col, ep_lens, bs, args, lo, hi):
    """Compute per-row success for one actor's lance. Returns
    (success_rows, succ_ep, n_calls) without writing anything."""
    ds = lance.dataset(path)
    t = ds.to_table(columns=["episode_idx", "qpos"])
    rec_ep = t.column(0).to_numpy().reshape(-1).astype(np.int64)
    qpos = fsl_to_np(t.column(1), len(rec_ep))
    bounds = episode_bounds(rec_ep)
    n_ep = len(bounds) - 1
    assert n_ep % args.num_eval == 0, (
        f"{path}: {n_ep} episodes not a multiple of num_eval -- a call must "
        "have dropped episodes; the k<->task mapping is broken")
    n_calls = n_ep // args.num_eval
    per_ep_rows = np.diff(bounds)
    assert (per_ep_rows == per_ep_rows[0]).all(), (
        f"{path}: unequal episode lengths {np.unique(per_ep_rows)} -- "
        "full-budget premise violated")
    assert per_ep_rows[0] >= args.min_len, (
        f"{path}: episodes are {per_ep_rows[0]} steps < --min-len "
        f"{args.min_len} -- not full-budget rollouts")

    # goals for every (call, env) of this actor, in appearance order
    goals = np.empty((n_ep, 3))
    for c in range(n_calls):
        d_eps, d_starts = reconstruct_draw(
            ep_col, step_col, ep_lens, args.seed_base + a * 100 + c,
            lo, hi, args.goal_offset, args.num_eval)
        L = ep_lens[int(d_eps[0])]  # regular layout asserted in main
        grow = d_eps * L + d_starts + args.goal_offset
        gp = fsl_to_np(
            exp.take(grow.tolist(), columns=["privileged_block_0_pos"]).column(0),
            args.num_eval)[:, :3]
        goals[c * args.num_eval:(c + 1) * args.num_eval] = gp

    success_rows = np.zeros(len(rec_ep), dtype=np.uint8)
    succ_ep = np.zeros(n_ep, dtype=bool)
    for k in range(n_ep):
        sl = slice(bounds[k], bounds[k + 1])
        hit = np.linalg.norm(qpos[sl, bs:bs + 3] - goals[k], axis=1) <= THRESH
        success_rows[sl] = hit.astype(np.uint8)
        succ_ep[k] = bool(hit.any())
    return success_rows, succ_ep, n_calls


def write_labeled(path, success_rows, out_suffix):
    ds = lance.dataset(path)
    base = path.rstrip("/")
    base = base[:-len(".lance")] if base.endswith(".lance") else base
    out = base + out_suffix + ".lance"
    names = [f.name for f in ds.schema] + ["success"]
    schema = pa.schema(list(ds.schema) + [pa.field("success", pa.uint8())])

    def gen():
        off = 0
        for b in ds.to_batches(batch_size=4096):
            arrs = [b.column(j) for j in range(b.num_columns)]
            arrs.append(pa.array(success_rows[off:off + b.num_rows],
                                 type=pa.uint8()))
            off += b.num_rows
            yield pa.RecordBatch.from_arrays(arrs, names=names)

    if os.path.exists(out):
        shutil.rmtree(out)
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(schema, gen()), out, schema=schema)
    chk = lance.dataset(out)
    assert chk.count_rows() == len(success_rows), "row count changed on rewrite"
    got = chk.to_table(columns=["success"]).column(0).to_numpy().reshape(-1)
    assert (got == success_rows).all(), "success column corrupted on write"
    print(f"[write] {out}: rows={chk.count_rows()}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--onpolicy", nargs="+", required=True,
                    help="in actor order a0 a1 a2 (seed = base + a*100 + call)")
    ap.add_argument("--arm1-logs", default=None,
                    help="template with {a} and {c}, e.g. "
                         "'/workspace/logs/dsp_col_a{a}_c{c}.log'; enables gate B")
    ap.add_argument("--ep-range", default="0:8000")
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out-suffix", default="_lab")
    ap.add_argument("--min-len", type=int, default=25,
                    help="minimum episode length; full-budget cube rollouts "
                         "are 50, anything under 25 means the k<->task "
                         "mapping premise is broken")
    ap.add_argument("--total-band", default="1400:1600",
                    help="hard accept band for total successes (gate C)")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.ep_range.split(":"))
    band_lo, band_hi = (int(x) for x in args.total_band.split(":"))

    exp = lance.dataset(args.expert)
    idx = exp.to_table(columns=["episode_idx", "step_idx"])
    ep_col = idx.column(0).to_numpy().reshape(-1).astype(np.int64)
    step_col = idx.column(1).to_numpy().reshape(-1).astype(np.int64)
    eps, cnt = np.unique(ep_col, return_counts=True)
    ep_lens = dict(zip(eps.tolist(), cnt.tolist()))
    # goal lookup uses row = ep*L + step; verify the layout really is regular
    L = int(cnt[0])
    regular = (cnt == L).all() and (ep_col == np.repeat(eps, L)).all() and (
        step_col == np.tile(np.arange(L), len(eps))).all()
    assert regular, "expert lance is not (episode-major, step-contiguous)"

    bs = discover_block_slice(exp)  # gate A inside

    # ---------- pass 1: label everything, evaluate gates, write NOTHING yet
    labelled = []
    gate_b_viol, gate_b_checked = [], 0
    tot_succ = tot_eps = 0
    for a, path in enumerate(args.onpolicy):
        success_rows, succ_ep, n_calls = label_one(
            path, a, exp, ep_col, step_col, ep_lens, bs, args, lo, hi)
        if args.arm1_logs:
            for c in range(n_calls):
                lg = args.arm1_logs.format(a=a, c=c)
                m = []
                if os.path.exists(lg):
                    m = re.findall(r"kept=(\d+) dropped=(\d+)",
                                   open(lg, errors="ignore").read())
                if not m:
                    print(f"[gateB] {lg}: no record line, skipping", flush=True)
                    continue
                gate_b_checked += 1
                dropped = int(m[-1][1])
                got = int(succ_ep[c * args.num_eval:(c + 1) * args.num_eval].sum())
                if got < dropped:
                    gate_b_viol.append((a, c, got, dropped))
        ns = int(succ_ep.sum())
        tot_succ += ns
        tot_eps += len(succ_ep)
        print(f"[label] a{a}: {ns}/{len(succ_ep)} succeeded "
              f"({100 * ns / len(succ_ep):.1f}%), {len(succ_ep) - ns} failed",
              flush=True)
        labelled.append((path, success_rows))

    # ---------- gates B & C
    ok = True
    for a, c, got, dropped in gate_b_viol:
        print(f"GATE B FAILED a{a}/c{c}: relabelled {got} successes but arm 1 "
              f"dropped {dropped} early-terminated (= succeeded) episodes on "
              "the same rollouts", flush=True)
        ok = False
    if args.arm1_logs:
        print(f"[gateB] {gate_b_checked} calls checked, "
              f"{len(gate_b_viol)} violations", flush=True)
    rate = 100 * tot_succ / tot_eps
    print(f"[total] {tot_succ}/{tot_eps} succeeded ({rate:.1f}%)", flush=True)
    if not (band_lo <= tot_succ <= band_hi):
        print(f"GATE C FAILED: total successes {tot_succ} outside "
              f"[{band_lo},{band_hi}] (expected ~1507 = arm 1's 1393 early "
              "terminations + ~114 late successes). Labels are wrong; "
              "nothing was written.", flush=True)
        ok = False
    if not ok:
        sys.exit(2)

    # ---------- pass 2: gates passed, write the labeled lances
    for path, success_rows in labelled:
        write_labeled(path, success_rows, args.out_suffix)
    print("RELABEL_DONE", flush=True)


if __name__ == "__main__":
    main()
