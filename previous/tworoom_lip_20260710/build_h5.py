"""Flatten a lance play dataset into the h5 layout train_lip.py expects:
'action' (T_total, a_dim) primitive-step actions and 'ep_offset' (N,) row
offset of each episode, indexed by episode id.
"""

import argparse

import h5py
import numpy as np

import stable_worldmodel as swm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ds = swm.data.load_dataset(args.dataset)
    act = np.asarray(ds.get_col_data("action"), dtype=np.float32)
    ep = np.asarray(ds.get_col_data("episode_idx"))

    n_ep = int(ep.max()) + 1
    ep_offset = np.zeros(n_ep, dtype=np.int64)
    # first row of each episode id (rows are episode-contiguous in lance)
    change = np.flatnonzero(np.diff(ep)) + 1
    starts = np.concatenate([[0], change])
    for row, e in zip(starts, ep[starts]):
        ep_offset[int(e)] = row

    with h5py.File(args.out, "w") as h:
        h.create_dataset("action", data=act)
        h.create_dataset("ep_offset", data=ep_offset)
    print(f"h5: {act.shape[0]} action rows (dim {act.shape[1]}), "
          f"{n_ep} episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
