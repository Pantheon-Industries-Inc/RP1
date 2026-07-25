"""Round-2 fine-tune dataset — 60/30/10 split, ZERO duplication (user spec).

Composition (single copies, exposure ratios set by construction):
  60%  expert  — random-EPISODE subsample sized to 2x the r1 slice
  30%  r1 on-policy — ALL of onpolicy_r1_a{0,1,2}
  10%  r2 on-policy — ALL of onpolicy_r2_a{0,1,2} (volume permitting; the
       actual share is reported — r2 pins nothing, expert is sized off r1)

Sizing rule: expert_steps = 2 * r1_steps; r2 rides along at its natural size
(~10% if collection landed near expectation). Episode-level sampling keeps
training windows contiguous. Schema = slim 6-col; episode_idx remapped unique.

Usage: build_r2_dataset.py OUT_LANCE [SEED]
"""
import sys
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

SLIM = "/workspace/dyna_data/expert_slim.lance"
R1 = [f"/workspace/dyna_data/onpolicy_r1_a{i}.lance" for i in range(3)]
R2 = [f"/workspace/dyna_data/onpolicy_r2_a{i}.lance" for i in range(3)]
COLS = ["episode_idx", "step_idx", "pixels", "action", "qpos", "qvel"]

out = Path(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(seed)
assert not out.exists(), f"{out} exists"

def rows_of(paths):
    return sum(lance.dataset(p).count_rows() for p in paths)

r1_steps = rows_of(R1)
r2_steps = rows_of(R2)
expert_target = 2 * r1_steps
print(f"r1={r1_steps} r2={r2_steps} expert_target={expert_target}", flush=True)

# ---- expert subsample: whole episodes (201 steps each) until target
slim = lance.dataset(SLIM)
n_expert_eps = expert_target // 201 + 1
ep_ids = rng.choice(10000, size=n_expert_eps, replace=False)
ep_ids.sort()
print(f"sampling {n_expert_eps} expert episodes", flush=True)

writer_schema = None
next_epi = 0
total = 0

def write_tbl(tbl, mode):
    lance.write_dataset(tbl, str(out), mode=mode)

first = True
batch_rows = []
# expert episodes: rows are ep*201 .. ep*201+200 (slim preserves order)
CHUNK = 40  # episodes per write
for ci in range(0, len(ep_ids), CHUNK):
    chunk = ep_ids[ci:ci + CHUNK]
    idx = np.concatenate([np.arange(e * 201, (e + 1) * 201) for e in chunk])
    tbl = slim.take(idx.tolist(), columns=COLS)
    # remap episode_idx to 0..n (dense) to keep uniqueness
    old = tbl["episode_idx"].to_numpy()
    remap = {int(e): next_epi + j for j, e in enumerate(chunk)}
    new = pa.array(np.vectorize(remap.get)(old).astype(np.int32))
    tbl = tbl.set_column(0, "episode_idx", new)
    write_tbl(tbl, "create" if first else "append")
    first = False
    next_epi += len(chunk)
    total += tbl.num_rows

print(f"expert slice written: {total} rows, {next_epi} episodes", flush=True)

# ---- on-policy slices (all rows, single copy)
for group, paths in (("r1", R1), ("r2", R2)):
    for p in paths:
        ds = lance.dataset(p)
        n_eps = int(pc.max(ds.to_table(columns=["episode_idx"])["episode_idx"]).as_py()) + 1
        batches = []
        for b in ds.scanner(columns=COLS, batch_size=8192).to_batches():
            epi = pc.cast(pc.add(b.column(0), pa.scalar(next_epi, pa.int32())), pa.int32())
            batches.append(pa.RecordBatch.from_arrays(
                [epi] + [b.column(i) for i in range(1, len(COLS))], names=COLS))
        tbl = pa.Table.from_batches(batches)
        write_tbl(tbl, "append")
        next_epi += n_eps
        total += tbl.num_rows
    print(f"{group} appended (running total {total} rows)", flush=True)

ds = lance.dataset(str(out))
n = ds.count_rows()
exp_rows = total - r1_steps - r2_steps
print(f"R2SET DONE {out}: rows={n} episodes~{next_epi} | "
      f"shares: expert {exp_rows/n:.1%} r1 {r1_steps/n:.1%} r2 {r2_steps/n:.1%}",
      flush=True)
