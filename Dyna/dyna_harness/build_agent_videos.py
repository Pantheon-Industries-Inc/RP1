"""Build clean agent-rollout mp4s from an SWM_RECORD_PATH lance.
Usage: build_agent_videos.py REC_LANCE OUT_DIR EP1,EP2,...  [fps]
"""
import io
import sys
from pathlib import Path

import imageio
import lance
import numpy as np
from PIL import Image

rec, outdir = sys.argv[1], sys.argv[2]
eps = [int(x) for x in sys.argv[3].split(",")]
fps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
Path(outdir).mkdir(parents=True, exist_ok=True)

ds = lance.dataset(rec)
n = ds.count_rows()
epi = np.asarray(ds.take(list(range(n)), columns=["episode_idx"]).to_pydict()["episode_idx"])
for e in eps:
    rows = np.where(epi == e)[0]
    if len(rows) == 0:
        print(f"ep{e}: MISSING")
        continue
    px = ds.take(rows.tolist(), columns=["pixels"]).to_pydict()["pixels"]
    frames = [np.array(Image.open(io.BytesIO(b)).convert("RGB")) for b in px]
    out = f"{outdir}/agent_ep{e}.mp4"
    imageio.mimwrite(out, frames, fps=fps, codec="libx264")
    print(f"ep{e}: {len(frames)} frames -> {out}")
