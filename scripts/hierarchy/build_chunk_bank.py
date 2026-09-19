"""Build a bank of REAL primitive-action chunks for constrained high-level search.

Why a bank instead of free macro-actions: FIG6 (RESULTS_hwm.md) measured the
trained high-level F2 as WORSE than simply composing the frozen low-level WM at
every horizon (l1 0.42 vs 0.37 at 200 env steps, z-scale 0.80), and a 3x larger
F2 got worse still (0.56) -- the floor is the 125->8-D macro information
bottleneck, not capacity. So the high level here searches over chunks that
actually occur in the data and rolls them through the low-level WM, which has no
bottleneck and is the better predictor. It is also the constraint Hi-LeWM found
load-bearing: "unconstrained search can select latent macro-actions that appear
favorable under the learned model but produce poor control targets".

Output: {"chunks": (N, K, act_dim) float32 z-scored, "mean", "std", "stride"}.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="action h5 (tools/build_action_h5)")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=25, help="primitive steps per chunk")
    p.add_argument("--num-chunks", type=int, default=4096)
    p.add_argument("--holdout-ep", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(a.h5, "r") as h:
        act = h["action"][:]
        ep_off = h["ep_offset"][:]
        ep_len = h["ep_len"][:]
    # cube pads episode-terminal steps with NaN: nan-aware stats, and never take
    # a chunk that would run into the padding
    mean, std = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-6
    act_n = np.nan_to_num((act - mean) / std).astype(np.float32)

    rng = np.random.default_rng(a.seed)
    n_eps = len(ep_off)
    pool = [e for e in range(n_eps) if (a.holdout_ep <= 0 or e < a.holdout_ep) and ep_len[e] > a.stride + 1]
    chunks = []
    while len(chunks) < a.num_chunks:
        e = int(rng.choice(pool))
        start = int(rng.integers(0, int(ep_len[e]) - a.stride - 1))
        row = int(ep_off[e]) + start
        c = act_n[row : row + a.stride]
        if c.shape[0] == a.stride and np.isfinite(c).all():
            chunks.append(c)
    arr = np.stack(chunks)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "chunks": torch.from_numpy(arr),
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "stride": a.stride,
        },
        out,
    )
    print(f"saved {out}: {arr.shape} chunks from {len(pool)} episodes", flush=True)


if __name__ == "__main__":
    main()
