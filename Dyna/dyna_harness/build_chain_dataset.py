"""Build a failure-state chaining dataset: restart tasks from mid-failure states.

For every FAILED on-policy episode (label from the *_lab.lance `success`
column), emit chained tasks that resume the ORIGINAL task from states reached
mid-failure: start = the failure rollout's state at t in --mids (default
15/25/35), goal = the original task's goal, unchanged. Rolling the actors from
these starts collects exactly the compounding-error states the WM never sees
from expert data -- the region where the graspmiss probe showed it imagines
lift on 100 % of real misses.

The output is a lance the UNMODIFIED eval path can consume as a task dataset
(`eval.dataset_name=...`): each chain is a 26-frame episode where
`_extract_init_goal` reads exactly two frames --

  frame 0   init: recorded qpos/qvel (set_state teleports the env into the
            mid-failure state), recorded pixels
  frame 25  goal: the expert goal frame -- pixels for the actor's goal
            conditioning, privileged_block_0_pos/quat for set_target_pos

frames 1-24 are copies of frame 0: `load_chunk` needs them to exist; nothing
reads them (dataset_videos only). With 26-frame episodes and goal_offset 25,
max_start = 0, so every episode has exactly ONE valid start row and an
`ep_range` slice of 51 episodes makes eval_wm's draw
(`choice(len(valid)-1, 50)`) a PERMUTATION of the slice's first 50 episodes --
deterministic full coverage, no cross-call duplicates, no code changes.
289 failures x 3 mids = 867 = 17 x 51 exactly.

Goals are recovered with the same machinery the relabel pass validated
(reconstruct_draw; gates were green: 1511/1800 vs ~1507 predicted).

z-scoring caveat for the COLLECTOR (not this script): cube.yaml defaults
`dataset.stats: ${eval.dataset_name}` -- collection against this dataset MUST
pin `dataset.stats=<expert lance>` or the policy normalizes with chain-set
statistics.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import lance
import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relabel_onpolicy import (  # noqa: E402
    THRESH, episode_bounds, fsl_to_np, reconstruct_draw,
)

# chain episode length is goal_offset+1 (26 in production): max_start = 0, so
# each episode has exactly one valid start row; set per-run in main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--labeled", nargs="+", required=True,
                    help="onpolicy *_lab.lance in actor order a0 a1 a2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mids", type=int, nargs="+", default=[15, 25, 35])
    ap.add_argument("--ep-range", default="0:8000")
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--slice", type=int, default=51,
                    help="ep_range slice size for the collector; the draw "
                         "covers the first slice-1 episodes of each slice "
                         "deterministically")
    ap.add_argument("--shuffle-seed", type=int, default=7,
                    help="chains are shuffled so each 51-slice mixes actors "
                         "and mids; fixed seed = reproducible slice plan")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.ep_range.split(":"))
    CHAIN_LEN = args.goal_offset + 1
    SLICE = args.slice

    exp = lance.dataset(args.expert)
    idx = exp.to_table(columns=["episode_idx", "step_idx"])
    ep_col = idx.column(0).to_numpy().reshape(-1).astype(np.int64)
    step_col = idx.column(1).to_numpy().reshape(-1).astype(np.int64)
    eps, cnt = np.unique(ep_col, return_counts=True)
    ep_lens = dict(zip(eps.tolist(), cnt.tolist()))
    L = int(cnt[0])
    assert (cnt == L).all() and (ep_col == np.repeat(eps, L)).all(), \
        "expert lance is not (episode-major, step-contiguous)"

    exp_cols = set(exp.schema.names)
    for c in ("pixels", "qpos", "qvel",
              "privileged_block_0_pos", "privileged_block_0_quat"):
        assert c in exp_cols, f"expert lance lacks {c!r}"

    # ------------------------------------------------ collect chain specs
    # spec: (actor, rec-lance row range of the mid frame, goal expert row)
    chains = []          # (a, mid_rows_slice_start, src_len, goal_row, src_ep_k)
    n_fail_total = 0
    per_lance = []
    for a, path in enumerate(args.labeled):
        ds = lance.dataset(path)
        assert "success" in set(ds.schema.names), f"{path}: not a labeled lance"
        t = ds.to_table(columns=["episode_idx", "success"])
        rec_ep = t.column(0).to_numpy().reshape(-1).astype(np.int64)
        succ = t.column(1).to_numpy().reshape(-1).astype(bool)
        bounds = episode_bounds(rec_ep)
        n_ep = len(bounds) - 1
        assert n_ep % args.num_eval == 0, f"{path}: episode/task grid broken"
        n_calls = n_ep // args.num_eval

        # original goals per (call, env), same derivation the gates validated
        goal_rows = np.empty(n_ep, dtype=np.int64)
        for c in range(n_calls):
            d_eps, d_starts = reconstruct_draw(
                ep_col, step_col, ep_lens, args.seed_base + a * 100 + c,
                lo, hi, args.goal_offset, args.num_eval)
            goal_rows[c * args.num_eval:(c + 1) * args.num_eval] = (
                d_eps * L + d_starts + args.goal_offset)

        n_fail = 0
        for k in range(n_ep):
            s, e = int(bounds[k]), int(bounds[k + 1])
            if succ[s:e].any():
                continue  # chained starts come from FAILURES only
            n_fail += 1
            for m in args.mids:
                assert m < e - s, f"mid {m} >= episode length {e - s}"
                chains.append((a, s + m, int(goal_rows[k]), k))
        per_lance.append((path, n_fail))
        n_fail_total += n_fail
        print(f"[chain] a{a}: {n_fail} failed episodes -> "
              f"{n_fail * len(args.mids)} chains", flush=True)

    rng = np.random.default_rng(args.shuffle_seed)
    order = rng.permutation(len(chains))
    chains = [chains[i] for i in order]
    n = len(chains)
    n_slices, rem = divmod(n, SLICE)
    print(f"[chain] total {n} chains from {n_fail_total} failures x "
          f"{len(args.mids)} mids -> {n_slices} slices of {SLICE}"
          + (f" + {rem} remainder (never drawn, harmless)" if rem else
         " (exact)"), flush=True)
    assert n_slices >= 1, "not enough chains for one slice"

    # ------------------------------------------------ stream the episodes
    # per-frame source rows: frame0..24 <- mid row (recorded), frame25 <- goal
    # row (expert). Fetch in bulk per source, then interleave.
    rec_dss = [lance.dataset(p) for p, _ in per_lance]
    REC_COLS = ["pixels", "action", "qpos", "qvel"]
    GOAL_COLS = ["pixels", "qpos", "qvel",
                 "privileged_block_0_pos", "privileged_block_0_quat"]

    mid_tabs = []
    for a, ds in enumerate(rec_dss):
        rows = [c[1] for c in chains if c[0] == a]
        tab = ds.take(rows, columns=REC_COLS) if rows else None
        # map recorded-row -> position in tab
        pos = {r: i for i, r in enumerate(rows)}
        mid_tabs.append((tab, pos))
    goal_rows_all = [c[2] for c in chains]
    goal_tab = exp.take(goal_rows_all, columns=GOAL_COLS)

    # goal block pos/quat as python lists once (small)
    gp = goal_tab.column("privileged_block_0_pos")
    gq = goal_tab.column("privileged_block_0_quat")

    schema = pa.schema([
        pa.field("episode_idx", pa.int32()),
        pa.field("step_idx", pa.int32()),
        pa.field("pixels", pa.binary()),
        pa.field("action", exp.schema.field("action").type
                 if "action" in exp_cols else rec_dss[0].schema.field("action").type),
        pa.field("qpos", rec_dss[0].schema.field("qpos").type),
        pa.field("qvel", rec_dss[0].schema.field("qvel").type),
        pa.field("privileged_block_0_pos", exp.schema.field("privileged_block_0_pos").type),
        pa.field("privileged_block_0_quat", exp.schema.field("privileged_block_0_quat").type),
    ])

    def episode_batch(eid, ci):
        a, mid_row, _, _ = chains[ci]
        tab, pos = mid_tabs[a]
        i = pos[mid_row]
        px0 = tab.column("pixels")[i]
        ac0 = tab.column("action")[i]
        qp0 = tab.column("qpos")[i]
        qv0 = tab.column("qvel")[i]
        pxg = goal_tab.column("pixels")[ci]
        qpg = goal_tab.column("qpos")[ci]
        qvg = goal_tab.column("qvel")[ci]
        # frames 0..24 = mid state, frame 25 = goal frame
        arrs = [
            pa.array([eid] * CHAIN_LEN, pa.int32()),
            pa.array(list(range(CHAIN_LEN)), pa.int32()),
            pa.array([px0.as_py()] * (CHAIN_LEN - 1) + [pxg.as_py()], pa.binary()),
            pa.array([ac0.as_py()] * CHAIN_LEN, type=schema.field("action").type),
            pa.array([qp0.as_py()] * (CHAIN_LEN - 1) + [qpg.as_py()],
                     type=schema.field("qpos").type),
            pa.array([qv0.as_py()] * (CHAIN_LEN - 1) + [qvg.as_py()],
                     type=schema.field("qvel").type),
            pa.array([gp[ci].as_py()] * CHAIN_LEN,
                     type=schema.field("privileged_block_0_pos").type),
            pa.array([gq[ci].as_py()] * CHAIN_LEN,
                     type=schema.field("privileged_block_0_quat").type),
        ]
        return pa.RecordBatch.from_arrays(arrs, schema=schema)

    def gen():
        for eid in range(n):
            yield episode_batch(eid, eid)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    lance.write_dataset(pa.RecordBatchReader.from_batches(schema, gen()),
                        args.out, schema=schema)

    chk = lance.dataset(args.out)
    assert chk.count_rows() == n * CHAIN_LEN, "row count mismatch"
    # verify a random chain end-to-end: init qpos == recorded mid qpos,
    # goal pos == expert goal pos
    probe = int(rng.integers(n))
    a, mid_row, goal_row, _ = chains[probe]
    got = chk.take([probe * CHAIN_LEN, probe * CHAIN_LEN + CHAIN_LEN - 1],
                   columns=["qpos", "privileged_block_0_pos"])
    want_q = fsl_to_np(rec_dss[a].take([mid_row], columns=["qpos"]).column(0), 1)
    want_g = fsl_to_np(exp.take([goal_row],
                                columns=["privileged_block_0_pos"]).column(0), 1)
    assert np.allclose(fsl_to_np(got.column(0), 2)[0], want_q[0]), "init qpos mismatch"
    assert np.allclose(fsl_to_np(got.column(1), 2)[1][:3], want_g[0][:3]), "goal pos mismatch"
    print(f"wrote {args.out}: rows={chk.count_rows()} episodes={n} "
          f"(probe chain {probe} verified)", flush=True)
    print(f"SLICE_PLAN n={n} slice={SLICE} n_slices={n_slices}", flush=True)
    print("BUILD_CHAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
