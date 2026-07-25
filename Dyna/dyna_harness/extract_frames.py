"""Extract N evenly-spaced frames from a video as PNGs (for visual analysis).
Usage: extract_frames.py VIDEO OUT_DIR [N]
"""
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

vid, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
frames = iio.imread(vid, index=None)
T = len(frames)
Path(out).mkdir(parents=True, exist_ok=True)
tag = Path(vid).stem
for i, ix in enumerate(np.linspace(0, T - 1, n).astype(int)):
    iio.imwrite(f"{out}/{tag}_f{i}_t{int(ix)}.png", frames[ix])
print(f"{vid}: T={T} -> {n} frames")
