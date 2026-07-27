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
"""
# ############################################################################
# # DATA SPLIT WARNING -- see ../DATA_SPLIT_POLICY.md
# # The expert slice mixed in here, and the on-policy lances appended to it,
# # MUST exclude every episode the evaluation draws tasks from. They currently
# # do NOT: expert episodes are sampled from all 10k and the on-policy rollouts
# # were collected from all 10k, while eval draws from the same pool. Required
# # split: fine-tune on episodes 0-7999 only, eval on 8000-9999
# # (episode_split.COLLECT / .EVAL; assert_disjoint() to enforce).
# ############################################################################
import argparse
import os
import shutil

import lance
import pyarrow as pa

COLS = ["episode_idx", "step_idx", "pixels", "action"]


def stream_source(ds, id_offset, next_id, remap):
    """Yield record batches with episode_idx remapped.

    remap=False: add id_offset to source ids (expert passthrough, offset 0).
    remap=True: assign fresh contiguous ids starting at next_id[0], bumping
    next_id in place; source episode boundaries detected by value change.
    """
    prev_src = None
    for batch in ds.to_batches(columns=COLS, batch_size=8192):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True)
    p.add_argument("--onpolicy", nargs="+", required=True)
    p.add_argument("--out", required=True, help="path ending in <name>.lance")
    p.add_argument("--onpolicy-frac", type=float, required=True,
                   help="target on-policy fraction of total ROWS (0.5 / 0.2)")
    args = p.parse_args()

    exp = lance.dataset(args.expert)
    n_exp = exp.count_rows()
    ons = [lance.dataset(q) for q in args.onpolicy]
    n_on = sum(d.count_rows() for d in ons)
    # f = K*n_on / (n_exp + K*n_on)  =>  K = f*n_exp / (n_on*(1-f))
    f = args.onpolicy_frac
    K = max(1, round(f * n_exp / (n_on * (1.0 - f))))
    eff = K * n_on / (n_exp + K * n_on)
    print(f"expert rows {n_exp}, on-policy rows {n_on}, dup K={K} "
          f"-> effective on-policy frac {eff:.3f} (target {f})", flush=True)

    # expert schema for the 4 columns is the reference
    ref_schema = pa.schema([exp.schema.field(c) for c in COLS])

    def gen():
        yield from stream_source(exp, 0, None, remap=False)
        max_exp_id = 100000  # expert ids are 0..9999; start fresh ids high
        nid = [max_exp_id]
        for k in range(K):
            for d in ons:
                yield from stream_source(d, 0, nid, remap=True)
        print(f"final on-policy episode id: {nid[0] - 1}", flush=True)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    reader = pa.RecordBatchReader.from_batches(ref_schema, gen())
    lance.write_dataset(reader, args.out, schema=ref_schema)

    chk = lance.dataset(args.out)
    ep = chk.to_table(columns=["episode_idx"]).column(0).to_numpy()
    import numpy as np
    assert (np.diff(ep) >= 0).all(), "episode_idx not monotone!"
    print(f"wrote {args.out}: rows={chk.count_rows()} "
          f"episodes={len(np.unique(ep))}", flush=True)
    print("BUILD_MIX_DONE", flush=True)


if __name__ == "__main__":
    main()
