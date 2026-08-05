"""Redo Dyna fine-tune dataset — parametrized no-dup expert/on-policy mix.

expert slice = EXPERT_MULT * (total on-policy steps), sampled as whole episodes;
on-policy = ALL rows of the given lances, single copy. episode_idx remapped
unique. EXPERT_MULT=1.5 -> 60/40 expert/on-policy.

Usage: build_redo_dataset.py OUT_LANCE EXPERT_MULT ONPOLICY.lance [ONPOLICY2 ...] [--seed N]
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
import shutil
import sys
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

SLIM = "/workspace/dyna_data/expert_slim.lance"
COLS = ["episode_idx", "step_idx", "pixels", "action", "qpos", "qvel"]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
seed = 0
for a in sys.argv[1:]:
    if a.startswith("--seed"):
        seed = int(a.split("=")[1]) if "=" in a else 0
out = Path(args[0])
mult = float(args[1])
onpolicy = args[2:]
rng = np.random.default_rng(seed)
assert not out.exists(), f"{out} exists"
assert Path(SLIM).exists(), "expert_slim.lance missing — build it first"

onp_steps = sum(lance.dataset(p).count_rows() for p in onpolicy)
expert_target = int(mult * onp_steps)
n_expert_eps = expert_target // 201 + 1
print(f"on-policy={onp_steps} steps; expert_target={expert_target} "
      f"({n_expert_eps} eps); mult={mult}", flush=True)

slim = lance.dataset(SLIM)
ep_ids = np.sort(rng.choice(10000, size=n_expert_eps, replace=False))

next_epi = 0
total = 0
first = True
CHUNK = 40
for ci in range(0, len(ep_ids), CHUNK):
    chunk = ep_ids[ci:ci + CHUNK]
    idx = np.concatenate([np.arange(e * 201, (e + 1) * 201) for e in chunk])
    tbl = slim.take(idx.tolist(), columns=COLS)
    remap = {int(e): next_epi + j for j, e in enumerate(chunk)}
    new = pa.array(np.vectorize(remap.get)(tbl["episode_idx"].to_numpy()).astype(np.int32))
    tbl = tbl.set_column(0, "episode_idx", new)
    lance.write_dataset(tbl, str(out), mode="create" if first else "append")
    first = False
    next_epi += len(chunk)
    total += tbl.num_rows
exp_rows = total
print(f"expert slice: {exp_rows} rows", flush=True)

for p in onpolicy:
    ds = lance.dataset(p)
    n_eps = int(pc.max(ds.to_table(columns=["episode_idx"])["episode_idx"]).as_py()) + 1
    batches = []
    for b in ds.scanner(columns=COLS, batch_size=8192).to_batches():
        epi = pc.cast(pc.add(b.column(0), pa.scalar(next_epi, pa.int32())), pa.int32())
        batches.append(pa.RecordBatch.from_arrays(
            [epi] + [b.column(i) for i in range(1, len(COLS))], names=COLS))
    lance.write_dataset(pa.Table.from_batches(batches), str(out), mode="append")
    next_epi += n_eps
    total += ds.count_rows()

n = lance.dataset(str(out)).count_rows()
print(f"REDOSET DONE {out}: rows={n} | shares expert {exp_rows/n:.1%} "
      f"on-policy {(n-exp_rows)/n:.1%}", flush=True)
