"""Build a Dyna fine-tune arm dataset (v2 — pure-arrow append, no re-encode):
  slim expert lance (6 cols)  +  on-policy episodes x DUP.

Schemas of the slim expert and the on-policy lances are identical (same
writer machinery), so appending is plain `lance.write_dataset(mode='append')`
with episode_idx remapped to stay globally unique (expert 0..9999, on-policy
copies 10000+). Pixels (jpeg bytes) pass through untouched — no
decode/re-encode generation loss.

Usage: build_arm_dataset.py OUT_LANCE DUP_FACTOR
"""
import shutil
import sys
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

EXPERT = "/workspace/datasets/ogb_cube_single/ogb_cube_single.lance"
SLIM = Path("/workspace/dyna_data/expert_slim.lance")
ONPOLICY = [f"/workspace/dyna_data/onpolicy_r1_a{i}.lance" for i in range(3)]
COLS = ["episode_idx", "step_idx", "pixels", "action", "qpos", "qvel"]

out = Path(sys.argv[1])
dup = int(sys.argv[2])

# ---------------------------------------------------------- 1. slim expert
if not SLIM.exists():
    print("building slim expert (6 cols, streamed)...", flush=True)
    src = lance.dataset(EXPERT)
    reader = src.scanner(columns=COLS, batch_size=4096).to_reader()
    lance.write_dataset(reader, str(SLIM))
    n = lance.dataset(str(SLIM)).count_rows()
    assert n == src.count_rows(), (n, src.count_rows())
    print(f"slim expert done: {n} rows", flush=True)
else:
    print("slim expert already present", flush=True)

# ------------------------------------------------- 2. copy slim -> arm base
if out.exists():
    print(f"{out} exists — removing stale copy", flush=True)
    shutil.rmtree(out)
print(f"copying slim -> {out} ...", flush=True)
shutil.copytree(SLIM, out)

# ------------------------------------- 3. append on-policy episodes x DUP
next_base = 10000
total_eps = 0
for rep in range(dup):
    for p in ONPOLICY:
        ds = lance.dataset(p)
        n_eps = int(
            pc.max(ds.to_table(columns=["episode_idx"])["episode_idx"]).as_py()
        ) + 1
        batches = []
        for b in ds.scanner(columns=COLS, batch_size=8192).to_batches():
            epi = pc.add(b.column(0), pa.scalar(next_base, pa.int32()))
            epi = pc.cast(epi, pa.int32())
            batches.append(
                pa.RecordBatch.from_arrays(
                    [epi] + [b.column(i) for i in range(1, len(COLS))],
                    names=COLS,
                )
            )
        tbl = pa.Table.from_batches(batches)
        lance.write_dataset(tbl, str(out), mode="append")
        next_base += n_eps
        total_eps += n_eps
    print(f"rep {rep + 1}/{dup} appended (total on-policy eps {total_eps})", flush=True)

ds = lance.dataset(str(out))
n = ds.count_rows()
last = ds.take([n - 1], columns=["episode_idx"]).to_pydict()["episode_idx"][0]
exp = 2010000 + dup * (38144 + 23177 + 19425)
print(f"ARM DONE {out}: rows={n} (expected {exp}) last_episode_idx={last} dup={dup}",
      flush=True)
assert n == exp, (n, exp)
