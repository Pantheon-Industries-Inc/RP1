"""Split eval_wm panel videos into individual unlabelled streams.

`save_panel_videos` writes one `env_{i}.mp4` per task laid out as
`agent | dataset | goal`. PRE and POST runs of the same draw seed put the same
task in the same env slot with the same start state and goal, so panel i of
each arm is directly comparable. This crops the layout back apart and writes
one file per stream, with no burned-in text — everything identifying the clip
lives in the filename:

    task{i}_pre-dyna_{OUTCOME}.mp4   task{i}_post-dyna_{OUTCOME}.mp4
    task{i}_expert-demo.mp4          task{i}_goal.png

Usage: export_individual.py PRE_DIR POST_DIR OUT_DIR TASK,TASK,... [LOG_PREFIX]
"""

import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

FPS = 10
HOLD_S = 1.0  # freeze the last frame so the end state is readable


def fits(w, cw, ch, n=3):
    pad, gap, lh = max(12, w // 14), max(10, w // 16), max(22, w // 9)
    return ((2 * pad + n * w + (n - 1) * gap + 15) // 16 * 16 == cw
            and (2 * pad + w + lh + 15) // 16 * 16 == ch)


def panel_geometry(cw, ch, n=3, known=224):
    """Recover the panel box from save_panel_videos' canvas arithmetic.

    The arithmetic rounds the canvas up to a multiple of 16, so it does NOT
    invert uniquely: a 736x288 canvas is produced by panel widths 222 through
    225 alike. Searching and taking the first (or largest) match silently
    crops 1-3 px off every frame. So trust the render size we actually asked
    for (`eval.img_size`) and only fall back to a search if it is inconsistent.
    """
    if fits(known, cw, ch, n):
        pad, gap = max(12, known // 14), max(10, known // 16)
        return known, known, pad, gap
    for w in range(16, 1025):
        if fits(w, cw, ch, n):
            print(f'warning: panel size {known} does not fit canvas '
                  f'{cw}x{ch}; falling back to {w}')
            return w, w, max(12, w // 14), max(10, w // 16)
    raise SystemExit(f'cannot infer panel geometry from canvas {cw}x{ch}')


def read_panels(path, n=3):
    frames = np.asarray(imageio.mimread(path, memtest=False))
    if frames.ndim == 4 and frames.shape[-1] == 4:
        frames = frames[..., :3]
    ch, cw = frames.shape[1:3]
    w, h, pad, gap = panel_geometry(cw, ch, n)
    panels = [
        frames[:, pad:pad + h, pad + j * (w + gap):pad + j * (w + gap) + w]
        for j in range(n)
    ]
    return panels, w, h


def successes(log_path):
    p = Path(log_path)
    if not p.exists():
        return None
    m = re.search(r"episode_successes.: array\(\[(.*?)\]\)", p.read_text(), re.S)
    if not m:
        return None
    return [x.strip() == 'True' for x in m.group(1).replace('\n', ' ').split(',')]


def write(path, frames):
    """Write the raw render, unmodified, with the final frame held."""
    frames = np.concatenate(
        [frames, np.repeat(frames[-1:], int(FPS * HOLD_S), axis=0)]
    )
    imageio.mimwrite(path, frames, fps=FPS, codec='libx264',
                     macro_block_size=None, quality=8)
    return len(frames)


def main():
    pre_dir, post_dir, out_dir = map(Path, sys.argv[1:4])
    tasks = [int(x) for x in sys.argv[4].split(',')]
    prefix = sys.argv[5] if len(sys.argv) > 5 else 'vidcmp'
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = Path('/workspace/logs')
    pre_ok = successes(logs / f'{prefix}_{pre_dir.name}.log')
    post_ok = successes(logs / f'{prefix}_{post_dir.name}.log')

    rows = []
    for i in tasks:
        p_path, q_path = pre_dir / f'env_{i}.mp4', post_dir / f'env_{i}.mp4'
        if not (p_path.exists() and q_path.exists()):
            print(f'task {i}: missing mp4, skipped')
            continue
        (p_agent, p_demo, p_goal), w, h = read_panels(p_path)
        (q_agent, _, _), _, _ = read_panels(q_path)

        pv = 'SUCCESS' if (pre_ok and pre_ok[i]) else 'FAIL'
        qv = 'SUCCESS' if (post_ok and post_ok[i]) else 'FAIL'

        write(out_dir / f'task{i:02d}_pre-dyna_{pv}.mp4', p_agent)
        write(out_dir / f'task{i:02d}_post-dyna_{qv}.mp4', q_agent)
        write(out_dir / f'task{i:02d}_expert-demo.mp4', p_demo)

        goal = p_goal[0] if p_goal.ndim == 4 else p_goal
        Image.fromarray(np.asarray(goal, dtype=np.uint8)).save(
            out_dir / f'task{i:02d}_goal.png'
        )
        rows.append((i, pv, qv, len(p_agent), len(q_agent)))
        print(f'task {i:2d}: pre={pv:7s} post={qv:7s} ({len(p_agent)} steps)')

    (out_dir / 'INDEX.md').write_text(
        f'# {out_dir.name}\n\n'
        '| task | PRE-Dyna | POST-Dyna | steps |\n|---|---|---|---|\n'
        + '\n'.join(f'| {i} | {a} | {b} | {c} |' for i, a, b, c, _ in rows)
        + '\n\nFiles per task: `pre-dyna_<outcome>.mp4`, '
          '`post-dyna_<outcome>.mp4`, `expert-demo.mp4`, `goal.png`. '
          'No text is burned into the frames.\n'
    )
    print(f'{len(rows)} tasks -> {out_dir}')


if __name__ == '__main__':
    main()
