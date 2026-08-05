"""Rank eval tasks by how much cube motion they actually demand.

eval_wm prints the sorted dataset row indices it drew; row -> (episode, step),
and `privileged_block_0_pos` over [row, row+offset] is the cube trajectory the
goal was taken from. Two measures:

  disp  ||cube(goal) - cube(start)||      how far the cube must travel
  lift  max z - z(start) over the window  whether the demo picks it UP
        (a lift means a genuine pick-and-place; a flat window is a push/slide)

Usage: task_stats.py PRE_LOG POST_LOG OFFSET
"""

import re
import sys
from pathlib import Path

import lance
import numpy as np

pre_log, post_log, offset = sys.argv[1], sys.argv[2], int(sys.argv[3])


def rows(p):
    t = Path(p).read_text()
    m = re.search(
        r"valid starting points found for evaluation\.\s*\n\[(.*?)\]", t, re.S
    )
    return [int(x) for x in m.group(1).split()]


def succ(p):
    m = re.search(
        r"episode_successes.: array\(\[(.*?)\]\)", Path(p).read_text(), re.S
    )
    return [x.strip() == "True" for x in m.group(1).replace("\n", " ").split(",")]


r = rows(pre_log)
assert r == rows(post_log), "arms drew different tasks -- not a paired comparison"
pre, post = succ(pre_log), succ(post_log)

ds = lance.dataset("/workspace/datasets/ogb_cube_single/ogb_cube_single.lance")
want = sorted({x + k for x in r for k in range(offset + 1)})
tb = ds.take(
    want, columns=["episode_idx", "step_idx", "privileged_block_0_pos"]
).to_pydict()
pos = {
    i: np.asarray(p, dtype=float).reshape(-1)
    for i, p in zip(want, tb["privileged_block_0_pos"])
}
epi = {i: int(np.asarray(e).reshape(-1)[0]) for i, e in zip(want, tb["episode_idx"])}
stp = {i: int(np.asarray(s).reshape(-1)[0]) for i, s in zip(want, tb["step_idx"])}

out = []
for i, row in enumerate(r):
    traj = np.stack([pos[row + k] for k in range(offset + 1)])
    disp = float(np.linalg.norm(traj[-1] - traj[0]))
    lift = float(traj[:, 2].max() - traj[0, 2])
    out.append((i, epi[row], stp[row], disp, lift, pre[i], post[i]))

hdr = ("task", "ep", "step", "disp_m", "lift_m")
print("%4s %5s %4s %7s %7s  PRE      POST" % hdr)
for i, e, s, d, l, a, b in sorted(out, key=lambda x: -x[3]):
    flag = "  <-- CURED" if (not a and b) else ("  <-- regress" if (a and not b) else "")
    print("%4d %5d %4d %7.3f %7.3f  %-7s %-7s%s" % (i, e, s, d, l, a, b, flag))

cured = [x for x in out if not x[5] and x[6]]
both = [x for x in out if x[5] and x[6]]
print("\nCURED_BY_DISP=" + ",".join(str(x[0]) for x in sorted(cured, key=lambda x: -x[3])))
print("LIFTED_BOTH_OK=" + ",".join(str(x[0]) for x in sorted(both, key=lambda x: -x[4])[:6]))
print("LIFTED_CURED=" + ",".join(str(x[0]) for x in sorted(cured, key=lambda x: -x[4])))
