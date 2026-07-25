"""Test: can we append 6-column on-policy episodes into a copy of the
33-column expert lance (schema evolution / null-fill), with pixels passed as
already-encoded JPEG bytes? Uses a tiny 50-episode copy of the expert lance.
"""
import shutil
from pathlib import Path

import lance
import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.data.format import get_format

TEST = Path("/workspace/dyna_data/_schema_test.lance")
if TEST.exists():
    shutil.rmtree(TEST)

# tiny base: first 2 fragments of the expert lance via lance write of a slice
src = lance.dataset("/workspace/datasets/ogb_cube_single/ogb_cube_single.lance")
tbl = src.take(list(range(402)), columns=None)  # 2 episodes worth, all columns
lance.write_dataset(tbl, str(TEST))
print("base rows:", lance.dataset(str(TEST)).count_rows(),
      "cols:", len(lance.dataset(str(TEST)).schema))

# one on-policy episode (6 cols, pixels as jpeg bytes)
onp = lance.dataset("/workspace/dyna_data/onpolicy_r1_a0.lance")
cols = ["pixels", "action", "qpos", "qvel"]
ep_rows = onp.take(list(range(50)), columns=cols + ["episode_idx"]).to_pydict()

episode = {c: [ep_rows[c][i] for i in range(50)] for c in cols}
episode["action"] = [np.asarray(a, dtype=np.float32) for a in episode["action"]]
episode["qpos"] = [np.asarray(a, dtype=np.float32) for a in episode["qpos"]]
episode["qvel"] = [np.asarray(a, dtype=np.float32) for a in episode["qvel"]]
# pixels stay bytes

with get_format("lance").open_writer(str(TEST)) as w:
    w.write_episodes(iter([episode]))

ds = lance.dataset(str(TEST))
print("after append rows:", ds.count_rows())
t = ds.take([ds.count_rows() - 1], columns=["pixels", "action", "episode_idx", "step_idx"]).to_pydict()
print("appended row: episode_idx", t["episode_idx"], "step_idx", t["step_idx"],
      "pixels bytes", len(t["pixels"][0]), "action", np.round(np.asarray(t["action"][0]), 3))
import io
from PIL import Image
img = Image.open(io.BytesIO(t["pixels"][0]))
print("pixel decode:", img.size, img.mode)
print("SCHEMA_APPEND_OK")
