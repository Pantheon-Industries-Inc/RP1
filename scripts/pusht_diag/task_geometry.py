"""Task geometry of the PushT eval draws: how far does the BLOCK actually move
between the start frame and the goal frame (offset 25), versus the agent?

Reads the same draw rule as the evaluator (episodes 16000-18685, rows with
step <= max_step - 25, 50 tasks per draw, seeded by the draw). Prints one line
per task and a per-draw summary, and flags tasks whose block start pose is
already within the success tolerance (20 px, 20 deg) of the goal block pose --
for those, a critic that returns "nearly done" once only the agent is in
place is CORRECT, so they cannot count as evidence of an agent shortcut (E9).
Env: H5 (dataset), OUT (json dir), DRAWS (default "42 43 44"), OFF (25).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

H5 = os.environ["H5"]
OUT = Path(os.environ.get("OUT", "."))
DRAWS = [int(d) for d in os.environ.get("DRAWS", "42 43 44").split()]
OFF = int(os.environ.get("OFF", "25"))
OUT.mkdir(parents=True, exist_ok=True)

with h5py.File(H5, "r") as h:
    epi = np.asarray(h["episode_idx"][:]).reshape(-1)
    stp = np.asarray(h["step_idx"][:]).reshape(-1)
    states = np.asarray(h["state"][:], dtype=np.float64)
lo, hi, n = 16000, 18685, 50
ep_ids = np.unique(epi)
ep_ids = ep_ids[(ep_ids >= lo) & (ep_ids < hi)]
maxlen = {int(e): int(stp[epi == e].max()) for e in ep_ids}
keep = np.isin(epi, ep_ids)
mx = np.zeros(len(epi), dtype=np.int64)
mx[keep] = np.array([maxlen[int(e)] for e in epi[keep]])
valid = np.nonzero(keep & (stp <= mx - OFF))[0]


def ang_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return np.degrees(d)


result: dict[str, object] = {"offset": OFF, "draws": {}}
for draw in DRAWS:
    rows = np.sort(np.random.default_rng(draw).choice(valid, size=n, replace=False))
    s0, g = states[rows], states[rows + OFF]
    agent_move = np.linalg.norm(g[:, :2] - s0[:, :2], axis=1)
    block_move = np.linalg.norm(g[:, 2:4] - s0[:, 2:4], axis=1)
    block_rot = ang_diff(g[:, 4], s0[:, 4])
    block_done_at_start = (block_move < 20) & (block_rot < 20)
    joint_pos_err = np.linalg.norm(g[:, :4] - s0[:, :4], axis=1)
    per_task = []
    print(f"== draw {draw}: task | agent px | block px | block deg | block already within tolerance at start")
    for i in range(n):
        per_task.append(
            {
                "task": i,
                "row": int(rows[i]),
                "agent_move_px": float(agent_move[i]),
                "block_move_px": float(block_move[i]),
                "block_rot_deg": float(block_rot[i]),
                "block_done_at_start": bool(block_done_at_start[i]),
                "joint_pos_err_px": float(joint_pos_err[i]),
            }
        )
        print(
            f"[geom d{draw}] {i:2d} | {agent_move[i]:6.1f} | {block_move[i]:6.1f} | {block_rot[i]:6.1f} | "
            f"{'YES' if block_done_at_start[i] else 'no'}"
        )
    summary = {
        "n_block_done_at_start": int(block_done_at_start.sum()),
        "n_block_move_lt_40px": int((block_move < 40).sum()),
        "median_agent_move_px": float(np.median(agent_move)),
        "median_block_move_px": float(np.median(block_move)),
        "median_block_rot_deg": float(np.median(block_rot)),
        "first15_block_done_at_start": [int(i) for i in np.nonzero(block_done_at_start[:15])[0]],
        "first15_block_move_lt_40px": [int(i) for i in np.nonzero(block_move[:15] < 40)[0]],
    }
    print(f"[geom d{draw}] SUMMARY {json.dumps(summary)}", flush=True)
    result["draws"][str(draw)] = {"per_task": per_task, "summary": summary}
(OUT / "task_geometry.json").write_text(json.dumps(result, indent=1))
print("[geom] DONE", flush=True)
