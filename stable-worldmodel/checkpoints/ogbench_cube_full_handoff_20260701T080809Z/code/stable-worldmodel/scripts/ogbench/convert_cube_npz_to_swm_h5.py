#!/usr/bin/env python
"""Convert official OGBench cube `.npz` files into SWM HDF5 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def episode_index(terminals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    terminal_idx = np.flatnonzero(terminals.astype(bool))
    if terminal_idx.size == 0:
        raise ValueError('No terminal rows found in OGBench dataset.')

    starts = np.r_[0, terminal_idx[:-1] + 1]
    lengths = terminal_idx - starts
    if np.any(lengths <= 0):
        raise ValueError('Invalid episode lengths computed from terminals.')

    offsets = np.r_[0, np.cumsum(lengths[:-1])].astype(np.int64)
    return lengths.astype(np.int32), offsets


def create_dataset(f: h5py.File, name: str, shape, dtype, chunks):
    return f.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
    )


def convert(src: Path, dst: Path, chunk_rows: int, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f'{dst} exists; pass --overwrite to replace it.')

    dst.parent.mkdir(parents=True, exist_ok=True)

    with np.load(src) as raw:
        required = {'observations', 'actions', 'terminals', 'qpos', 'qvel'}
        missing = required.difference(raw.files)
        if missing:
            raise KeyError(f'{src} is missing keys: {sorted(missing)}')

        observations = raw['observations']
        actions = raw['actions']
        qpos = raw['qpos']
        qvel = raw['qvel']
        ep_len, ep_offset = episode_index(raw['terminals'])
        total_rows = int(ep_len.sum())
        obs_shape = observations.shape
        action_shape = actions.shape
        qpos_shape = qpos.shape
        qvel_shape = qvel.shape
        proprio_dim = qpos_shape[1] + qvel_shape[1]

        with h5py.File(dst, 'w', libver='latest') as f:
            f.create_dataset('ep_len', data=ep_len, dtype=np.int32)
            f.create_dataset('ep_offset', data=ep_offset, dtype=np.int64)
            pixels = create_dataset(
                f,
                'pixels',
                (total_rows, *obs_shape[1:]),
                np.uint8,
                (min(chunk_rows, total_rows), *obs_shape[1:]),
            )
            action = create_dataset(
                f,
                'action',
                (total_rows, action_shape[1]),
                np.float32,
                (min(chunk_rows, total_rows), action_shape[1]),
            )
            observation = create_dataset(
                f,
                'observation',
                (total_rows, proprio_dim),
                np.float32,
                (min(chunk_rows, total_rows), proprio_dim),
            )
            proprio = create_dataset(
                f,
                'proprio',
                (total_rows, proprio_dim),
                np.float32,
                (min(chunk_rows, total_rows), proprio_dim),
            )

            out = 0
            for start, length in tqdm(
                zip(ep_offset, ep_len),
                total=len(ep_len),
                desc=f'Converting {src.name}',
            ):
                raw_start = int(start + np.searchsorted(ep_offset, start))
                raw_end = raw_start + int(length)

                for lo in range(raw_start, raw_end, chunk_rows):
                    hi = min(lo + chunk_rows, raw_end)
                    n = hi - lo
                    dst_slice = slice(out, out + n)
                    pixels[dst_slice] = observations[lo:hi]
                    action[dst_slice] = actions[lo:hi].astype(
                        np.float32, copy=False
                    )
                    obs = np.concatenate(
                        [qpos[lo:hi], qvel[lo:hi]], axis=-1
                    ).astype(np.float32, copy=False)
                    observation[dst_slice] = obs
                    proprio[dst_slice] = obs
                    out += n

            f.attrs['source'] = str(src)
            f.attrs['format'] = 'stable_worldmodel_hdf5'
            f.attrs['notes'] = (
                'Converted from official OGBench visual cube npz. Terminal '
                'dummy rows are dropped; observation/proprio are qpos+qvel.'
            )
            f.swmr_mode = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=Path)
    parser.add_argument('dst', type=Path)
    parser.add_argument('--chunk-rows', type=int, default=4096)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    convert(args.src, args.dst, args.chunk_rows, args.overwrite)


if __name__ == '__main__':
    main()
