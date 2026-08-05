"""Build the two reacher h5 files from the lance dataset:

1. train h5 (--train-out): ALL episodes, scalar columns only (action, qpos,
   qvel, ...), no pixels. Feeds train_lip_ac.py, which indexes ep_offset BY
   EPISODE ID (ids come from the latent cache built over the same lance), so
   ep_len/ep_offset are id-indexed arrays. ~50 MB.
2. eval h5 (--eval-out): first --eval-episodes episode blocks in FILE order,
   all columns including decoded pixels. Feeds eval_wm.py (task replay + goal
   images + z-stats); its HDF5Dataset consumes ep_len/ep_offset positionally.
   Sized to fit the volume (full 10k x 224px would be ~300 GB).

Both get explicit episode_idx/step_idx datasets (eval_wm.episode_col needs
them; the generic hdf5 writer only stores ep_len/ep_offset).
"""
import argparse
import os
from io import BytesIO

import h5py
import numpy as np

import stable_worldmodel as swm

SCALAR_COLS = ["action", "qpos", "qvel", "observation", "success", "score",
               "target_pos", "finger_pos", "reward", "terminated", "truncated"]


def episode_blocks(ds):
    """Contiguous episode blocks in file order: list of (id, start, length).
    Episodes are written whole by the collector but not necessarily in id
    order; each id must appear in exactly one contiguous block."""
    ep = np.asarray(ds.get_col_data("episode_idx")).reshape(-1).astype(np.int64)
    st = np.asarray(ds.get_col_data("step_idx")).reshape(-1).astype(np.int64)
    bounds = np.flatnonzero(np.diff(ep) != 0) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(ep)]])
    blocks = [(int(ep[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]
    ids = [b[0] for b in blocks]
    assert len(ids) == len(set(ids)), "an episode id appears in >1 block"
    for bid, s, ln in blocks:
        assert (st[s:s + ln] == np.arange(ln)).all(), f"episode {bid}: step_idx not 0..{ln-1}"
    return ep, st, blocks


def write_scalars(f, ds, n_rows, ep, st):
    cols = [c for c in SCALAR_COLS if c in ds.column_names]
    for c in cols:
        f.create_dataset(c, data=np.asarray(ds.get_col_data(c))[:n_rows])
    f.create_dataset("episode_idx", data=ep[:n_rows])
    f.create_dataset("step_idx", data=st[:n_rows])
    return cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dmc/reacher_random.lance")
    p.add_argument("--train-out", required=True)
    p.add_argument("--eval-out", required=True)
    p.add_argument("--eval-episodes", type=int, default=1024)
    args = p.parse_args()

    ds = swm.data.load_dataset(args.dataset)
    ep, st, blocks = episode_blocks(ds)
    n_total = len(blocks)
    print(f"dataset: {n_total} episodes, {len(ep)} rows", flush=True)

    # ---- train h5: all rows, id-indexed ep_len/ep_offset, no pixels
    if not os.path.exists(args.train_out):
        ids = np.array([b[0] for b in blocks])
        assert ids.min() == 0 and ids.max() == n_total - 1, "ids not 0..N-1"
        off_by_id = np.zeros(n_total, dtype=np.int64)
        len_by_id = np.zeros(n_total, dtype=np.int64)
        for bid, s, ln in blocks:
            off_by_id[bid] = s
            len_by_id[bid] = ln
        tmp = args.train_out + ".tmp"
        with h5py.File(tmp, "w") as f:
            cols = write_scalars(f, ds, len(ep), ep, st)
            f.create_dataset("ep_len", data=len_by_id)
            f.create_dataset("ep_offset", data=off_by_id)
        os.replace(tmp, args.train_out)
        print(f"wrote {args.train_out}: {n_total} eps (id-indexed), cols={cols}", flush=True)
    else:
        print(f"train h5 exists: {args.train_out}", flush=True)

    # ---- eval h5: first n_eval blocks in file order, with pixels
    if not os.path.exists(args.eval_out):
        n_eval = min(args.eval_episodes, n_total)
        sub = blocks[:n_eval]
        n_rows = sub[-1][1] + sub[-1][2]
        assert all(s + ln <= n_rows for _, s, ln in sub)
        tmp = args.eval_out + ".tmp"
        with h5py.File(tmp, "w") as f:
            cols = write_scalars(f, ds, n_rows, ep, st)
            f.create_dataset("ep_len", data=np.array([b[2] for b in sub], dtype=np.int64))
            f.create_dataset("ep_offset", data=np.array([b[1] for b in sub], dtype=np.int64))
            from PIL import Image
            first = np.array(Image.open(BytesIO(bytes(ds.get_row_data([0])["pixels"][0]))))
            dset = f.create_dataset(
                "pixels", shape=(n_rows, *first.shape), dtype=np.uint8,
                chunks=(64, *first.shape),
            )
            B = 512
            for i in range(0, n_rows, B):
                rows = ds.get_row_data(list(range(i, min(i + B, n_rows))))
                imgs = [np.array(Image.open(BytesIO(bytes(q)))) for q in rows["pixels"]]
                dset[i:i + len(imgs)] = np.stack(imgs)
                if (i // B) % 20 == 0:
                    print(f"  pixels {i}/{n_rows}", flush=True)
        os.replace(tmp, args.eval_out)
        print(f"wrote {args.eval_out}: {n_eval} eps, {n_rows} rows, cols={cols}+pixels", flush=True)
    else:
        print(f"eval h5 exists: {args.eval_out}", flush=True)
    print("BUILD_H5_DONE", flush=True)


if __name__ == "__main__":
    main()
