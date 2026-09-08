"""Agent-displaced latent cache for PushT (counterfactual critic augmentation).

Every dataset frame is re-rendered from its logged 7-d state with the agent
moved by a per-episode constant offset (the agent keeps its real motion, at a
random place in the arena) and encoded with the frozen world model. Rows line
up with the plain fs1 cache row-for-row, so the two can be indexed together;
``state`` carries ``[logged state (7) | displaced agent xy (2) | transit (1)]``
where ``transit`` is the number of primitive steps a free agent needs to get
from the displaced position back to its real one (``|delta| / v``, ``v`` = the
dataset's p90 per-step agent speed).

Why: the PushT temporal-distance critic can satisfy itself by agent placement
alone (E9 in docs/campaigns/2026-09-03/PUSHT_DIAG.md). Expert data never shows
"agent where the expert's agent ended, block not there", so nothing pins V
high in that region and the refiner finds it. Training the critic on
displaced-agent frames labelled ``transit + d`` (the cost of walking back and
then following the expert) covers the counterfactual; the label is an upper
bound on the true distance, which the low expectile treats as usual.

Example::

    pixi run tool tool=cache_agent_aug wm=$WM dataset=$H5 \
        out=caches/counterstrike_fs1_agentaug.pt max_episodes=16000
"""

from __future__ import annotations

import multiprocessing as mp
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from rlp.config import dispatch, run_hydra
from rlp.core.world_model.runtime import build_featurizer, load_wm, pick_device
from rlp.data import LatentCache
from rlp.logging import logger

_ENV: Any = None


def _init_worker(resolution: int) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/PushT-v1)

    global _ENV
    _ENV = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=int(resolution)).unwrapped
    _ENV.reset(seed=0)


def _render_chunk(states5: np.ndarray, jpeg_quality: int = 0) -> np.ndarray:
    """Render ``(B, 5)`` PushT states (agent xy, block xy, block angle) to uint8 frames.

    ``jpeg_quality > 0`` round-trips every frame through JPEG so the renders
    carry the stored dataset frames' compression artifacts.
    """
    if _ENV is None:
        raise RuntimeError("render worker not initialised")
    size = int(_ENV.render_size)
    out = np.empty((len(states5), size, size, 3), dtype=np.uint8)
    for i, s in enumerate(states5):
        _ENV._set_state(np.asarray(s[:5], dtype=np.float64))
        out[i] = _ENV.render()
    return _jpeg_roundtrip(out, jpeg_quality) if jpeg_quality else out


def _render_chunk_args(args: tuple[np.ndarray, int]) -> np.ndarray:
    return _render_chunk(*args)


def _decode_jpeg(p: object) -> np.ndarray:
    from PIL import Image

    if isinstance(p, (bytes, bytearray, np.bytes_)):
        return np.asarray(Image.open(BytesIO(bytes(p))).convert("RGB"))
    x = np.asarray(p)
    return x.transpose(1, 2, 0) if (x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3) else x


def _jpeg_roundtrip(frames: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode raw renders as JPEG at ``quality`` so they carry the stored
    frames' compression artifacts (otherwise the encoder can tell augmented
    frames from real ones by their lack of artifacts)."""
    from PIL import Image

    out = np.empty_like(frames)
    for i, frame in enumerate(frames):
        buffer = BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=int(quality))
        out[i] = np.asarray(Image.open(BytesIO(buffer.getvalue())).convert("RGB"))
    return out


def _match_jpeg_quality(renders: np.ndarray, stored: np.ndarray, candidates: tuple[int, ...]) -> tuple[int, float]:
    """Pick the JPEG quality whose round-trip best reproduces the stored frames (pixel MAE)."""
    best_q, best_mae = 0, float(np.abs(renders.astype(np.int16) - stored.astype(np.int16)).mean())
    for q in candidates:
        mae = float(np.abs(_jpeg_roundtrip(renders, q).astype(np.int16) - stored.astype(np.int16)).mean())
        if mae < best_mae:
            best_q, best_mae = q, mae
    return best_q, best_mae


def _episode_rows(episode_idx: np.ndarray, step_idx: np.ndarray) -> list[np.ndarray]:
    order = np.lexsort((step_idx, episode_idx))
    ep_sorted = episode_idx[order]
    bounds = np.flatnonzero(np.diff(ep_sorted)) + 1
    return [order[a:b] for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(order)], strict=True)]


def _run(cfg: DictConfig) -> None:
    args = cfg
    rng = np.random.default_rng(int(args.seed))

    with h5py.File(str(args.dataset), "r") as h:
        episode_idx = np.asarray(h["episode_idx"][:]).reshape(-1).astype(np.int64)
        step_idx = np.asarray(h["step_idx"][:]).reshape(-1).astype(np.int64)
        state = np.asarray(h[str(args.state_key)][:], dtype=np.float64)
        n = len(episode_idx)
        if args.max_episodes is not None:
            in_split = int(np.count_nonzero(episode_idx < int(args.max_episodes)))
            if not np.array_equal(np.flatnonzero(episode_idx < int(args.max_episodes)), np.arange(in_split)):
                raise ValueError("episodes are not stored contiguously; cannot apply max_episodes as a prefix")
            n = in_split
        episode_idx, step_idx, state = episode_idx[:n], step_idx[:n], state[:n]
        if state.ndim != 2 or state.shape[1] < 5:
            raise ValueError(
                f"{args.state_key} must be (N, >=5) [agent xy, block xy, block angle, ...], got {state.shape}"
            )
        # fidelity probe rows: re-render at the TRUE state and compare with the stored frame
        n_fid = int(min(int(args.fidelity_rows), n - 1))
        fid_rows = np.sort(rng.choice(n - 1, size=n_fid, replace=False)) if n_fid > 0 else np.empty(0, np.int64)
        fid_rows = fid_rows[episode_idx[fid_rows] == episode_idx[fid_rows + 1]]
        fid_frames = np.stack([_decode_jpeg(h["pixels"][int(r)]) for r in fid_rows]) if len(fid_rows) else None
        fid_next = np.stack([_decode_jpeg(h["pixels"][int(r) + 1]) for r in fid_rows]) if len(fid_rows) else None
    logger.info(f"Agent-aug cache: {n} rows, {len(np.unique(episode_idx))} episodes, state dim {state.shape[1]}")

    # ---- per-episode constant displacement of the agent; free-transit speed from the data
    rows_by_episode = _episode_rows(episode_idx, step_idx)
    speeds: list[np.ndarray] = []
    disp = np.empty((n, 2), dtype=np.float64)
    lo, hi, margin = float(args.arena_low), float(args.arena_high), float(args.margin)
    for rows in rows_by_episode:
        agent = state[rows, :2]
        if len(rows) > 1:
            speeds.append(np.linalg.norm(np.diff(agent, axis=0), axis=1))
        target = rng.uniform(lo, hi, size=2)
        delta = target - agent.mean(axis=0)
        disp[rows] = np.clip(agent + delta, margin, 512.0 - margin)
    speed_all = np.concatenate(speeds) if speeds else np.ones(1)
    speed_p50, speed_p90 = float(np.percentile(speed_all, 50)), float(np.percentile(speed_all, 90))
    transit_speed = max(speed_p90, 1e-3)
    transit = np.linalg.norm(disp - state[:, :2], axis=1) / transit_speed
    logger.info(
        f"agent speed px/step p50={speed_p50:.2f} p90={speed_p90:.2f}; displacement px "
        f"mean={np.linalg.norm(disp - state[:, :2], axis=1).mean():.1f}; transit steps mean={transit.mean():.1f} "
        f"p90={np.percentile(transit, 90):.1f}"
    )
    aug5 = state[:, :5].copy()
    aug5[:, :2] = disp

    # ---- render (CPU pool, created before the WM touches CUDA) + encode
    workers = max(1, int(args.workers))
    # spawn by default: forked children inherit the parent's torch/SDL state
    # and hang on macOS (and are fragile after CUDA init on Linux)
    ctx = mp.get_context(str(args.start_method))
    pool = ctx.Pool(workers, initializer=_init_worker, initargs=(int(args.resolution),))
    try:
        device = pick_device(args.device)
        wm = load_wm(str(args.wm), device=device)
        featurizer = build_featurizer(wm, device=device, img_size=int(args.resolution), train_res=args.train_res)

        # ---- JPEG matching: the stored frames are JPEG-decoded, raw renders are
        # not; pick the quality that best reproduces the stored pixels so the
        # encoder cannot tell augmented frames from real ones by artifacts
        jpeg_q = 0
        raw_true_render = None
        if fid_frames is not None and len(fid_rows):
            raw_true_render = pool.apply(_render_chunk, (state[fid_rows, :5], 0))
            jq = args.jpeg_quality
            if jq is not None and str(jq) == "auto":
                qualities = (50, 60, 70, 75, 80, 85, 90, 95)
                jpeg_q, matched_mae = _match_jpeg_quality(raw_true_render, fid_frames, qualities)
                logger.info(
                    f"JPEG quality matched to the stored frames: q={jpeg_q or 'none'} (pixel MAE {matched_mae:.3f})"
                )
            elif jq is not None and int(jq) > 0:
                jpeg_q = int(jq)
        elif args.jpeg_quality is not None and str(args.jpeg_quality) != "auto":
            jpeg_q = int(args.jpeg_quality)

        batch = int(args.batch_size)
        chunks = [(aug5[i : i + batch], jpeg_q) for i in range(0, n, batch)]
        z_chunks: list[torch.Tensor] = []
        log_every = max(1, len(chunks) // 20)
        for k, frames in enumerate(pool.imap(_render_chunk_args, chunks, chunksize=1), start=1):
            with torch.no_grad():
                z_chunks.append(featurizer({"pixels": frames}).float().cpu())
            if k == 1 or k % log_every == 0 or k == len(chunks):
                logger.info(f"agent-aug encode chunk {k}/{len(chunks)} rows={min(k * batch, n)}/{n}")
        z = torch.cat(z_chunks, dim=0)

        # ---- fidelity: re-render at the TRUE state vs. the stored frame
        meta: dict[str, object] = {
            "wm": str(args.wm),
            "dataset": str(args.dataset),
            "device": device,
            "train_res": args.train_res,
            "augmentation": "agent_offset_per_episode",
            "seed": int(args.seed),
            "arena": [lo, hi],
            "margin": margin,
            "agent_speed_p50": speed_p50,
            "agent_speed_p90": speed_p90,
            "transit_speed": transit_speed,
            "jpeg_quality": jpeg_q,
            "state_layout": "state(7)|displaced_agent_xy(2)|transit_steps(1)",
        }
        if fid_frames is not None and len(fid_rows) and raw_true_render is not None:
            true_render = _jpeg_roundtrip(raw_true_render, jpeg_q) if jpeg_q else raw_true_render
            aug_render = pool.apply(_render_chunk, (aug5[fid_rows], jpeg_q))
            pix_mae = float(np.abs(true_render.astype(np.int16) - fid_frames.astype(np.int16)).mean())
            pix_frac = float((np.abs(true_render.astype(np.int16) - fid_frames.astype(np.int16)).max(-1) > 40).mean())
            with torch.no_grad():
                z_jpeg = featurizer({"pixels": fid_frames}).float().cpu()
                z_next = featurizer({"pixels": fid_next}).float().cpu()
                z_true = featurizer({"pixels": true_render}).float().cpu()
                z_aug = featurizer({"pixels": aug_render}).float().cpu()
            d_rerender = float((z_jpeg - z_true).norm(dim=-1).mean())
            d_consec = float((z_jpeg - z_next).norm(dim=-1).mean())
            d_aug = float((z_jpeg - z_aug).norm(dim=-1).mean())
            meta.update(
                fidelity_rows=int(len(fid_rows)),
                fidelity_pixel_mae=pix_mae,
                fidelity_pixel_frac_gt40=pix_frac,
                fidelity_latent_rerender_l2=d_rerender,
                fidelity_latent_consecutive_l2=d_consec,
                fidelity_latent_aug_l2=d_aug,
            )
            ratio = d_rerender / max(d_consec, 1e-6)
            logger.info(
                f"fidelity over {len(fid_rows)} rows: pixel MAE {pix_mae:.2f} (frac>40: {pix_frac:.4f}); "
                f"latent L2 re-render {d_rerender:.3f} vs consecutive-frame {d_consec:.3f} (ratio {ratio:.2f}); "
                f"displaced-agent {d_aug:.3f}"
            )
            if d_rerender > d_consec:
                logger.warning(
                    "re-rendered frames differ from stored frames by more than one step of motion; check the renderer"
                )
    finally:
        pool.close()
        pool.join()

    aug_state = np.concatenate([state, disp, transit[:, None]], axis=1).astype(np.float32)
    cache = LatentCache(
        z=z,
        episode_idx=torch.from_numpy(episode_idx),
        step_idx=torch.from_numpy(step_idx),
        state=torch.from_numpy(aug_state),
        meta=meta,
    )
    Path(str(args.out)).parent.mkdir(parents=True, exist_ok=True)
    cache.save(str(args.out))
    logger.success(f"Cached {len(cache.z)} agent-displaced latents (dim={cache.latent_dim}) at {args.out}")


def main() -> object:
    return run_hydra(dispatch, config_name="tools/cache_agent_aug")


if __name__ == "__main__":
    main()
