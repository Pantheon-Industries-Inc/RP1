"""Build a Dyna fine-tune lance: expert ⊕ duplicated on-policy rows.

Raw pyarrow/lance streaming — jpeg bytes pass through untouched (no
decode/re-encode). Only the 4 columns the fine-tune loader needs are kept:
episode_idx, step_idx, pixels, action. Episode ids stay monotone
non-decreasing (loader asserts episode-contiguity): expert keeps its ids,
each on-policy duplication copy gets fresh ascending ids.

The arm's on-policy fraction is hit by ROW count (what the window sampler
actually sees), not episode count — on-policy eval-path episodes are ~51
steps vs expert 201, so the recipe's "8x" (episode-count parity) would give
only ~20% by rows. The duplication factor is computed from the target
fraction and logged.

FAILURE FRACTION (--failure-frac). With one uniform duplication factor the
failure share of the mix is not a knob — it is whatever the collector happened
to produce, because duplication scales successes and failures together. That is
how the thin/full arms ended up bracketing an unknown optimum by accident:

    arm            failure rows / total mix     duplication
    thin (K=87)    ~0.40                        87 uniform
    full (K=18)    ~0.08                        18 uniform
    ftmisses r1    ~0.10  (90/10 expert:misses) — the run that broke the
                                                 88-plateau, pre-split

--failure-frac makes it explicit by duplicating the two outcome pools at
separate rates (K_f, K_s), so failure share and on-policy share move
independently. Both fractions are shares of TOTAL mix rows, so they are
directly comparable to the table above, and 0 <= failure_frac <= onpolicy_frac.
Omit it (or pass 'auto') for the historical uniform-K behaviour.

Requires a per-episode outcome label, which comes from the `success` column
written by the SWM_RECORD_PATH recorder (world.py). Lances collected before that
column existed do NOT need re-collecting -- the label is recoverable offline,
because the task draw is a deterministic function of the collection seed and
full-length collection drops no episodes, so recorded episode k is drawn task k
and the target pose can be read from the expert data at (episode, start+25).
See ../HANDOFF_20260728.md §2.1. Relabelling also beats re-collecting on the
merits: every failure-fraction cell is then built from identical on-policy rows,
so the sweep isolates the ratio instead of confounding it with a reshuffle.
"""
# ############################################################################
# # DATA SPLIT -- see ../DATA_SPLIT_POLICY.md
# # The expert slice mixed in here, and the on-policy lances appended to it,
# # MUST exclude every episode the evaluation draws tasks from. Enforced:
# # pass --expert-ep-hi 8000 (fine-tune on episodes 0-7999, eval on 8000-9999;
# # episode_split.COLLECT / .EVAL, assert_disjoint()). The on-policy side is
# # enforced upstream by the collector's `+eval.ep_range=0:8000`, which its
# # driver asserts was applied. Omitting --expert-ep-hi warns but proceeds.
# ############################################################################
import argparse
import os
import shutil

import lance
import numpy as np
import pyarrow as pa

COLS = ["episode_idx", "step_idx", "pixels", "action"]
OUTCOME = "success"
# Duplication above this is the memorisation regime that made arm 1's null
# uninterpretable (K=87: ~10k distinct windows seen ~174x over 2 epochs).
# Runs that were fine sat at K=18-25.
MAX_DUP_DEFAULT = 30


def stream_source(ds, id_offset, next_id, remap, filt=None, keep=None):
    """Yield record batches with episode_idx remapped.

    remap=False: add id_offset to source ids (expert passthrough, offset 0).
    remap=True: assign fresh contiguous ids starting at next_id[0], bumping
    next_id in place; source episode boundaries detected by value change.
    filt: optional lance filter expression, used to hold eval episodes out of
    the expert slice (see ../DATA_SPLIT_POLICY.md).
    keep: optional set of SOURCE episode ids to emit; rows from other episodes
    are dropped. Masked in Python rather than pushed into `filt` because the
    outcome label is per-episode (latched over the rollout) while a filter is
    per-row -- `success == 1` would keep only the post-goal tail of each
    successful episode, not the episode. Costs a full pass per copy over rows
    that are then discarded; the mix build is minutes, so this is not worth
    optimising into a chunked `episode_idx IN (...)` expression.
    """
    prev_src = None
    for batch in ds.to_batches(columns=COLS, batch_size=8192, filter=filt):
        if keep is not None:
            mask = np.isin(batch.column(0).to_numpy(), keep)
            if not mask.any():
                continue
            batch = batch.filter(pa.array(mask))
        ep = batch.column(0).to_numpy()
        if remap:
            new = ep.copy()
            for i in range(len(ep)):
                if prev_src is None or ep[i] != prev_src:
                    prev_src = ep[i]
                    cur = next_id[0]
                    next_id[0] += 1
                new[i] = cur
            arrays = [pa.array(new, type=pa.int32())] + [
                batch.column(j) for j in range(1, batch.num_columns)
            ]
        else:
            arrays = [pa.array(ep + id_offset, type=pa.int32())] + [
                batch.column(j) for j in range(1, batch.num_columns)
            ]
        yield pa.RecordBatch.from_arrays(arrays, names=COLS)


def outcome_split(ds, path):
    """Return (fail_eps, succ_eps, fail_rows, succ_rows) for one on-policy lance.

    Episode label is latched -- succeeded if the goal was ever reached -- which
    is how episode_successes is scored for these tasks (world.py).
    """
    if OUTCOME not in set(ds.schema.names):
        raise SystemExit(
            f"{path}: no {OUTCOME!r} column, so it cannot be divided by outcome.\n"
            f"  columns: {sorted(ds.schema.names)}\n"
            "  This lance predates the recorder's outcome label. It does NOT need\n"
            "  re-collecting -- relabel it offline (../HANDOFF_20260728.md §2.1):\n"
            "  the task draw is deterministic in the collection seed and\n"
            "  full-length collection drops no episodes, so recorded episode k is\n"
            "  drawn task k. New collections should set SWM_RECORD_OUTCOME=1.\n"
            "  Or drop --failure-frac to build with uniform duplication."
        )
    t = ds.to_table(columns=["episode_idx", OUTCOME])
    ep = t.column(0).to_numpy().reshape(-1)
    ok = t.column(1).to_numpy().reshape(-1).astype(bool)
    eps, inv = np.unique(ep, return_inverse=True)
    ever = np.zeros(len(eps), dtype=bool)
    np.logical_or.at(ever, inv, ok)
    rows = np.bincount(inv, minlength=len(eps))
    return eps[~ever], eps[ever], int(rows[~ever].sum()), int(rows[ever].sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True)
    p.add_argument("--onpolicy", nargs="+", required=True)
    p.add_argument("--out", required=True, help="path ending in <name>.lance")
    p.add_argument("--onpolicy-frac", type=float, required=True,
                   help="target on-policy fraction of total ROWS (0.5 / 0.2)")
    p.add_argument("--failure-frac", default="auto",
                   help="target FAILED-rollout fraction of total ROWS, or 'auto' "
                        "(default) for one uniform duplication factor, which "
                        "leaves the failure share at whatever the collector "
                        "produced. Must be <= --onpolicy-frac. Reference points: "
                        "thin arm ~0.40, full arm ~0.08, ftmisses r1 ~0.10.")
    p.add_argument("--max-dup", type=int, default=MAX_DUP_DEFAULT,
                   help=f"refuse a duplication factor above this "
                        f"(default {MAX_DUP_DEFAULT}); K=87 is the regime that "
                        f"made arm 1's null uninterpretable")
    p.add_argument("--allow-dup", action="store_true",
                   help="override --max-dup, having decided the memorisation "
                        "risk is acceptable")
    p.add_argument("--expert-ep-hi", type=int, default=None,
                   help="keep only expert episodes with episode_idx < this. REQUIRED for "
                        "the episode-disjoint Dyna control: the eval episodes must not "
                        "enter the fine-tune. See ../DATA_SPLIT_POLICY.md")
    args = p.parse_args()

    exp = lance.dataset(args.expert)
    exp_filt = None
    if args.expert_ep_hi is not None:
        exp_filt = f"episode_idx < {args.expert_ep_hi}"
        n_exp = exp.scanner(columns=["episode_idx"], filter=exp_filt).count_rows()
        print(f"[split] expert restricted to {exp_filt}: {n_exp} of "
              f"{exp.count_rows()} rows", flush=True)
    else:
        n_exp = exp.count_rows()
        print("[split] WARNING: no --expert-ep-hi given, using ALL expert episodes. "
              "If eval draws from this file the fine-tune is leaking. "
              "See ../DATA_SPLIT_POLICY.md", flush=True)
    ons = [lance.dataset(q) for q in args.onpolicy]
    n_on = sum(d.count_rows() for d in ons)

    f = args.onpolicy_frac
    # total rows T satisfies n_exp = (1-f)*T
    T = n_exp / (1.0 - f)

    if str(args.failure_frac).lower() == "auto":
        # f = K*n_on / (n_exp + K*n_on)  =>  K = f*n_exp / (n_on*(1-f))
        K = max(1, round(f * n_exp / (n_on * (1.0 - f))))
        eff = K * n_on / (n_exp + K * n_on)
        print(f"expert rows {n_exp}, on-policy rows {n_on}, dup K={K} "
              f"-> effective on-policy frac {eff:.3f} (target {f})", flush=True)
        pools = [(None, K)]  # (keep-set, dup) with keep=None meaning "all rows"
        n_fail = n_succ = None
    else:
        phi = float(args.failure_frac)
        if not 0.0 <= phi <= f:
            raise SystemExit(
                f"--failure-frac {phi} must be within [0, --onpolicy-frac {f}]: "
                "both are shares of TOTAL mix rows, so failures cannot exceed "
                "the on-policy half."
            )
        fail_eps, succ_eps, n_fail, n_succ = [], [], 0, 0
        for d, q in zip(ons, args.onpolicy):
            fe, se, fr, sr = outcome_split(d, q)
            print(f"[outcome] {q.split('/')[-1]}: {len(fe)} failed eps "
                  f"({fr} rows), {len(se)} succeeded eps ({sr} rows)", flush=True)
            fail_eps.append(fe); succ_eps.append(se); n_fail += fr; n_succ += sr
        if (n_fail == 0 and phi > 0) or (n_succ == 0 and phi < f):
            raise SystemExit(
                f"an outcome pool needed by --failure-frac {phi} is empty "
                f"(failed rows {n_fail}, succeeded rows {n_succ}). If this is "
                "full-length collection, check the recorder logged "
                "kept_success>0 -- terminate_at_goal=False makes "
                "world.terminateds useless as a label."
            )
        # K_f*n_fail = phi*T ; K_s*n_succ = (f-phi)*T. A pool whose target is
        # zero rows (phi=0: success-only; phi=f: failure-only) gets K=0 and is
        # excluded entirely -- flooring at 1 would silently leak it in.
        K_f = 0 if phi == 0 else max(1, round(phi * T / n_fail))
        K_s = 0 if phi == f else max(1, round((f - phi) * T / n_succ))
        tot = n_exp + K_f * n_fail + K_s * n_succ
        print(f"expert rows {n_exp}, on-policy rows {n_on} "
              f"(failed {n_fail}, succeeded {n_succ})", flush=True)
        print(f"dup K_fail={K_f} K_succ={K_s} -> effective failure frac "
              f"{K_f * n_fail / tot:.3f} (target {phi}), on-policy frac "
              f"{(K_f * n_fail + K_s * n_succ) / tot:.3f} (target {f})", flush=True)
        worst = max(K_f, K_s)
        if worst > args.max_dup and not args.allow_dup:
            scarce = "failed" if K_f >= K_s else "succeeded"
            n_scarce = len(np.concatenate(fail_eps if K_f >= K_s else succ_eps))
            raise SystemExit(
                f"duplication K={worst} exceeds --max-dup {args.max_dup}: only "
                f"{n_scarce} distinct {scarce} episodes back that pool, so this "
                f"mix would repeat them {worst}x. Collect more {scarce} episodes, "
                f"lower --failure-frac, or pass --allow-dup to accept the "
                f"memorisation risk (K=87 is what made arm 1's null "
                f"uninterpretable)."
            )
        # failures first, then successes: ids stay monotone across both;
        # zero-K pools are dropped, not floored to 1
        pools = [(e, k) for e, k in
                 [(np.concatenate(fail_eps), K_f), (np.concatenate(succ_eps), K_s)]
                 if k > 0]

    # expert schema for the 4 columns is the reference
    ref_schema = pa.schema([exp.schema.field(c) for c in COLS])

    def gen():
        yield from stream_source(exp, 0, None, remap=False, filt=exp_filt)
        max_exp_id = 100000  # expert ids are 0..9999; start fresh ids high
        nid = [max_exp_id]
        for keep, dup in pools:
            for k in range(dup):
                for d in ons:
                    yield from stream_source(d, 0, nid, remap=True, keep=keep)
        print(f"final on-policy episode id: {nid[0] - 1}", flush=True)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    reader = pa.RecordBatchReader.from_batches(ref_schema, gen())
    lance.write_dataset(reader, args.out, schema=ref_schema)

    chk = lance.dataset(args.out)
    ep = chk.to_table(columns=["episode_idx"]).column(0).to_numpy()
    assert (np.diff(ep) >= 0).all(), "episode_idx not monotone!"
    n_out = chk.count_rows()
    print(f"wrote {args.out}: rows={n_out} episodes={len(np.unique(ep))}", flush=True)
    if n_fail is not None:
        # the achieved split is what the fine-tune actually sees; report it
        # rather than the target, since both K are rounded to integers
        got_on = n_out - n_exp
        print(f"achieved: on-policy {got_on / n_out:.3f}, "
              f"failure {K_f * n_fail / n_out:.3f}", flush=True)
    print("BUILD_MIX_DONE", flush=True)


if __name__ == "__main__":
    main()
