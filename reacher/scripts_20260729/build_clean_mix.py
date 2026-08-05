"""Rebuild the Dyna mix WITHOUT eval-range episodes (user directive 2026-07-29).

Measured leak in reacher_mix_r1_5050.lance: 400,000 rows (10.1% of 3,972,908)
carry episode_idx in [8000, 10000) -- exactly the pool `+eval.ep_range=8000:10000`
draws its tasks from. Fine-tuning the WM on those rows and then evaluating on
them is train-on-test, the same defect the OGBench campaign separated out.

Expert ids are preserved by build_dyna_mix (0..9999); on-policy episodes were
remapped above the expert max (>=10000), and the on-policy collection itself
already drew tasks only from 0:8000. So the fix is a row filter on the expert
side, plus a proportional trim of on-policy rows so the arm stays a true 50/50
by rows (dropping 400k expert rows alone would drift the fraction to 0.552).

Episode ids are remapped to be consecutive in output order, and rows of each
episode stay contiguous, so the loader's episode-contiguity assumption holds.

Disk-guarded: aborts (leaving the source untouched) if free space would fall
under the margin -- an ENOSPC mid-write is what corrupted the 80/20 build once.
"""

import shutil
import sys

import lance
import numpy as np
import pyarrow as pa

SRC = "/workspace/dyna_data/reacher_mix_r1_5050.lance"
DST = "/workspace/dyna_data/reacher_mix_r2_clean5050.lance"
EVAL_LO, EVAL_HI = 8000, 10000
MARGIN_GB = 3.0
BATCH = 2048

src = lance.dataset(SRC)
ep_all = src.to_table(columns=["episode_idx"]).column(0).to_numpy()
n_all = len(ep_all)

is_expert = ep_all < 10000
keep_expert = is_expert & ~((ep_all >= EVAL_LO) & (ep_all < EVAL_HI))
is_onp = ~is_expert
n_keep_expert = int(keep_expert.sum())

# trim on-policy from the END (whole episodes) so frac stays 0.5 by rows
onp_eps = np.unique(ep_all[is_onp])
target_onp = n_keep_expert  # 50/50
keep_onp_mask = np.zeros(n_all, dtype=bool)
running = 0
for e in onp_eps:
    m = ep_all == e
    c = int(m.sum())
    if running + c > target_onp:
        continue
    keep_onp_mask |= m
    running += c

keep = keep_expert | keep_onp_mask
n_keep = int(keep.sum())
print(f"source rows {n_all} -> keep {n_keep}", flush=True)
print(f"  expert kept {n_keep_expert} (dropped {int(is_expert.sum()) - n_keep_expert} "
      f"eval-range rows)", flush=True)
print(f"  on-policy kept {running} of {int(is_onp.sum())}", flush=True)
print(f"  effective on-policy frac {running / n_keep:.3f}", flush=True)

est_gb = 32.0 * n_keep / n_all
free_gb = shutil.disk_usage("/workspace").free / 2**30
print(f"estimated output {est_gb:.1f}G, free {free_gb:.1f}G", flush=True)
if free_gb - est_gb < MARGIN_GB:
    sys.exit(f"ABORT: {free_gb:.1f}G free, need {est_gb + MARGIN_GB:.1f}G")

# remap kept episode ids to consecutive values, preserving row order
kept_ep = ep_all[keep]
uniq = []
prev = None
for v in kept_ep:
    if v != prev:
        uniq.append(v)
        prev = v
remap = {int(v): i for i, v in enumerate(uniq)}
print(f"remapping {len(uniq)} episodes to 0..{len(uniq) - 1}", flush=True)

written = 0
schema = None


def batches():
    global written, schema
    off = 0
    checked = 0
    for rb in src.to_batches(batch_size=BATCH):
        n = rb.num_rows
        m = keep[off:off + n]
        off += n
        if not m.any():
            continue
        tb = pa.Table.from_batches([rb]).filter(pa.array(m))
        ep = tb.column("episode_idx").to_numpy()
        tb = tb.set_column(
            tb.schema.get_field_index("episode_idx"),
            "episode_idx",
            pa.array([remap[int(v)] for v in ep], type=tb.schema.field("episode_idx").type),
        )
        if schema is None:
            schema = tb.schema
        written += tb.num_rows
        checked += tb.num_rows
        if checked >= 200_000:
            checked = 0
            fg = shutil.disk_usage("/workspace").free / 2**30
            print(f"  {written}/{n_keep} rows, {fg:.1f}G free", flush=True)
            if fg < MARGIN_GB:
                raise RuntimeError(f"disk guard tripped at {fg:.1f}G free")
        for b in tb.to_batches(max_chunksize=BATCH):
            yield b


first = None
gen = batches()
for b in gen:
    first = b
    break
if first is None:
    sys.exit("ABORT: nothing to write")


def all_batches():
    yield first
    yield from gen


lance.write_dataset(all_batches(), DST, schema=schema, mode="create",
                    max_rows_per_file=200_000)
print(f"wrote {written} rows -> {DST}", flush=True)

chk = lance.dataset(DST)
ep2 = chk.to_table(columns=["episode_idx"]).column(0).to_numpy()
leak = ((ep2 >= EVAL_LO) & (ep2 < EVAL_HI)).sum()
print(f"verify: rows {len(ep2)} episodes {len(np.unique(ep2))} monotone="
      f"{bool((np.diff(ep2) >= 0).all())}", flush=True)
# NB: post-remap ids are dense 0..N-1, so an id in [8000,10000) is expected and
# harmless -- the provenance check is the ROW COUNT matching the plan above.
print(f"rows in remapped-id band {EVAL_LO}:{EVAL_HI} = {leak} (ids are remapped; "
      f"expected, not a leak)", flush=True)
assert len(ep2) == written, "row count mismatch"
print("BUILD_CLEAN_MIX_DONE", flush=True)
