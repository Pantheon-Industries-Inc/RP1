#!/usr/bin/env python
"""Write short MP4 examples from official OGBench visual cube datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np


def first_npz_episode_frames(
    src: Path, max_frames: int, stride: int
) -> np.ndarray:
    with np.load(src) as raw:
        terminals = raw['terminals'].astype(bool)
        terminal_idx = np.flatnonzero(terminals)
        if terminal_idx.size == 0:
            raise ValueError(f'No terminal rows found in {src}')
        end = int(terminal_idx[0])
        frames = raw['observations'][:end:stride]
        return frames[:max_frames]


def first_h5_episode_frames(
    src: Path, max_frames: int, stride: int
) -> np.ndarray:
    with h5py.File(src, 'r') as f:
        length = int(f['ep_len'][0])
        stop = min(length, max_frames * stride)
        return f['pixels'][:stop:stride][:max_frames]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-dir', type=Path)
    parser.add_argument('--h5-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=160)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()
    if args.npz_dir is None and args.h5_dir is None:
        parser.error('Provide --h5-dir or --npz-dir.')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        '1cube': ('visual_cube_single_play_v0.h5', 'visual-cube-single-play-v0.npz'),
        '2cube': ('visual_cube_double_play_v0.h5', 'visual-cube-double-play-v0.npz'),
        '3cube': ('visual_cube_triple_play_v0.h5', 'visual-cube-triple-play-v0.npz'),
    }
    for label, (h5_name, npz_name) in variants.items():
        if args.h5_dir is not None and (args.h5_dir / h5_name).exists():
            frames = first_h5_episode_frames(
                args.h5_dir / h5_name, args.max_frames, args.stride
            )
        elif args.npz_dir is not None:
            frames = first_npz_episode_frames(
                args.npz_dir / npz_name, args.max_frames, args.stride
            )
        else:
            raise FileNotFoundError(h5_name)
        out = args.out_dir / f'ogbench_{label}_example.mp4'
        iio.imwrite(out, frames, fps=args.fps, codec='libx264')
        print(out)


if __name__ == '__main__':
    main()
