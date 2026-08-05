"""Build a Dyna fine-tune lance: expert ⊕ duplicated on-policy rows.

Repo-tracked successor to Dyna/dyna_harness/build_dyna_mix.py, which lived
only on the pod. Two changes, both for the DINO-WM proprio variant:

  * the carried column set is no longer hardcoded to 4 columns. The old script
    kept exactly [episode_idx, step_idx, pixels, action], which silently
    dropped `proprio` -- and PreJEPA/DINO-WM builds keys_to_load from
    wm.encoding, so a proprio-variant fine-tune on such a mix either dies in
    the loader or (worse) trains on a re-derived column. Extra columns are now
    auto-detected (any of proprio/state present in every input) or named
    explicitly with --extra-cols.
  * a missing requested column is a hard error, not a skip.

Raw pyarrow/lance streaming -- jpeg bytes pass through untouched (no
decode/re-encode). Episode ids stay monotone non-decreasing (the loader
asserts episode-contiguity): expert keeps its ids, each on-policy duplication
copy gets fresh ascending ids.

The arm's on-policy fraction is hit by ROW count (what the window sampler
actually sees), not episode count -- on-policy eval-path episodes are ~51
steps vs expert ~201, so the recipe's "8x" (episode-count parity) would give
only ~20% by rows. The duplication factor is computed from the target
fraction and logged.
"""
import argparse
import os
import shutil

import lance
import pyarrow as pa

BASE_COLS = ["episode_idx", "step_idx", "pixels", "action"]
# Carried when present in every input. TwoRoom emits both (env.py: proprio ==
# state == agent position); MuJoCo envs emit qpos/qvel instead.
OPTIONAL_COLS = ["proprio", "state", "qpos", "qvel"]


def stream_source(ds, cols, id_offset, next_id, remap):
    """Yield record batches with episode_idx remapped.

    remap=False: add id_offset to source ids (expert passthrough, offset 0).
    remap=True: assign fresh contiguous ids starting at next_id[0], bumping
    next_id in place; source episode boundaries detected by value change.
    """
    prev_src = None
    for batch in ds.to_batches(columns=cols, batch_size=8192):
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
        yield pa.RecordBatch.from_arrays(arrays, names=cols)


def resolve_cols(exp, ons, requested):
    """Pick the carried columns; hard-error on a requested-but-absent one."""
    have = [set(d.schema.names) for d in [exp, *ons]]
    missing = [c for c in BASE_COLS if not all(c in h for h in have)]
    if missing:
        raise SystemExit(f"inputs lack required column(s) {missing}")
    if requested:
        extra = [c.strip() for c in requested.split(",") if c.strip()]
        for c in extra:
            absent = [i for i, h in enumerate(have) if c not in h]
            if absent:
                where = ", ".join(
                    "expert" if i == 0 else f"onpolicy[{i - 1}]" for i in absent
                )
                raise SystemExit(
                    f"--extra-cols asked for {c!r} but it is absent from: {where}. "
                    "The DINO-WM proprio variant needs proprio in BOTH the "
                    "expert data and the collected rollouts -- check that the "
                    "collector ran with proprio in SWM_RECORD_COLS."
                )
    else:
        extra = [c for c in OPTIONAL_COLS if all(c in h for h in have)]
    return BASE_COLS + extra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True)
    p.add_argument("--onpolicy", nargs="+", required=True)
    p.add_argument("--out", required=True, help="path ending in <name>.lance")
    p.add_argument("--onpolicy-frac", type=float, required=True,
                   help="target on-policy fraction of total ROWS (0.5 / 0.2)")
    p.add_argument("--extra-cols", default=None,
                   help="comma list carried beyond episode_idx/step_idx/pixels/"
                        "action; absent columns are a hard error. Default: "
                        "auto-detect from " + ",".join(OPTIONAL_COLS))
    args = p.parse_args()

    exp = lance.dataset(args.expert)
    ons = [lance.dataset(q) for q in args.onpolicy]
    cols = resolve_cols(exp, ons, args.extra_cols)
    print(f"carrying columns: {cols}", flush=True)

    n_exp = exp.count_rows()
    n_on = sum(d.count_rows() for d in ons)
    # f = K*n_on / (n_exp + K*n_on)  =>  K = f*n_exp / (n_on*(1-f))
    f = args.onpolicy_frac
    K = max(1, round(f * n_exp / (n_on * (1.0 - f))))
    eff = K * n_on / (n_exp + K * n_on)
    print(f"expert rows {n_exp}, on-policy rows {n_on}, dup K={K} "
          f"-> effective on-policy frac {eff:.3f} (target {f})", flush=True)

    # expert schema for the carried columns is the reference
    ref_schema = pa.schema([exp.schema.field(c) for c in cols])

    def gen():
        yield from stream_source(exp, cols, 0, None, remap=False)
        max_exp_id = 100000  # expert ids are 0..9999; start fresh ids high
        nid = [max_exp_id]
        for _k in range(K):
            for d in ons:
                yield from stream_source(d, cols, 0, nid, remap=True)
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
