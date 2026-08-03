"""Rebuild expert_actions.h5 for train_lip_ac.py from the lance dataset.

The original lived only on a dead pod and was not preserved. train_lip_ac reads
exactly two datasets:

    action     (N_rows, action_dim)  every dataset row, in row order
    ep_offset  (max_ep_id + 1,)      first row index of each episode id,
                                     indexed as ep_off[e] + frameskip * t

so ep_offset must be addressable by the real episode id (0..9999), not by a
compacted rank. Actions are written unmodified -- NaN padding on episode-terminal
rows is preserved because train_lip_ac relies on nan-aware stats
(np.nanmean/np.nanstd) and would otherwise be handed fabricated values.
"""

import argparse
import sys

import h5py
import numpy as np

sys.path.insert(0, '/workspace/code/stable-worldmodel')

import stable_worldmodel as swm  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',
                   default='/root/datasets/ogb_cube_single/ogb_cube_single.lance')
    p.add_argument('--out', default='/workspace/datasets/expert_actions.h5')
    args = p.parse_args()

    ds = swm.data.load_dataset(args.dataset)
    epi = np.asarray(ds.get_col_data('episode_idx')).reshape(-1).astype(np.int64)
    stp = np.asarray(ds.get_col_data('step_idx')).reshape(-1).astype(np.int64)
    act = np.asarray(ds.get_col_data('action'), dtype=np.float32)
    if act.ndim == 1:
        act = act.reshape(len(epi), -1)
    print(f'[h5] rows {len(epi)} | action {act.shape} | '
          f'NaN rows {int(np.isnan(act).any(1).sum())}', flush=True)

    # rows must already be in (episode, step) order for ep_offset + fs*t to work
    assert np.all(np.diff(np.lexsort((stp, epi))) == 1) or \
        np.all(epi[:-1] <= epi[1:]), 'dataset rows are not in episode order'

    n_ep = int(epi.max()) + 1
    ep_offset = np.full(n_ep, -1, dtype=np.int64)
    first = np.nonzero(np.concatenate(([True], epi[1:] != epi[:-1])))[0]
    ep_offset[epi[first]] = first
    assert (ep_offset >= 0).all(), 'some episode id has no rows'

    # verify the addressing contract on a few episodes
    for e in (0, 1, n_ep // 2, n_ep - 1):
        o = ep_offset[e]
        assert epi[o] == e and stp[o] == 0, \
            f'ep {e}: offset {o} points at ep {epi[o]} step {stp[o]}'
    print(f'[h5] ep_offset for {n_ep} episodes, '
          f'stride check {np.unique(np.diff(ep_offset))[:3]}', flush=True)

    with h5py.File(args.out, 'w') as h:
        h.create_dataset('action', data=act, compression=None)
        h.create_dataset('ep_offset', data=ep_offset, compression=None)
    print(f'[h5] wrote {args.out}', flush=True)

    with h5py.File(args.out, 'r') as h:
        a2, o2 = h['action'][:], h['ep_offset'][:]
    amu, astd = np.nanmean(a2, 0), np.nanstd(a2, 0)
    print(f'[h5] readback action {a2.shape} ep_offset {o2.shape}', flush=True)
    print(f'[h5] action mean {np.round(amu, 6).tolist()}', flush=True)
    print(f'[h5] action std  {np.round(astd, 6).tolist()}', flush=True)
    # train_lip_ac --action-stats-pin expert hardcodes these; they should match
    print('[h5] expected expert pin mean '
          '[0.010831, -0.003126, 0.002633, 0.000422, 0.15846]', flush=True)
    print('[h5] expected expert pin std  '
          '[0.2887, 0.392736, 0.641535, 0.391823, 0.249935]', flush=True)


if __name__ == '__main__':
    main()
