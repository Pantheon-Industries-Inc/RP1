import glob, re
import numpy as np
import imageio.v2 as imageio
def redmask(img):
    r, g, b = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)
    return (r > 110) & (g < 95) & (b < 95) & (r - np.maximum(g, b) > 40)
def centroid(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 8: return None
    return float(xs.mean()), float(ys.mean()), len(xs)
rows = []
for path in sorted(glob.glob("/workspace/failure_gallery/*.mp4")):
    name = path.split("/")[-1][:-4]
    frames = imageio.mimread(path, memtest=False)
    H, W = frames[0].shape[:2]
    col_is_gutter = [np.std(frames[0][:, x].astype(int)) < 18 and frames[0][:, x].mean() > 180 for x in range(W)]
    xs = [x for x in range(W) if not col_is_gutter[x]]
    runs = []
    s = xs[0]
    for a, b in zip(xs, xs[1:]):
        if b != a + 1: runs.append((s, a)); s = b
    runs.append((s, xs[-1]))
    runs = [r for r in runs if r[1] - r[0] > 100]
    (ax0, ax1), (gx0, gx1) = runs[0], runs[-1]
    goal = None
    for f in frames[:3]:
        goal = centroid(redmask(np.asarray(f)[:, gx0:gx1+1]))
        if goal: break
    tr = []
    for f in frames:
        tr.append(centroid(redmask(np.asarray(f)[:, ax0:ax1+1])))
    tr = [t for t in tr if t]
    if not tr or goal is None:
        rows.append((name, "TRACK-FAIL", 0, 0, 0, 0)); continue
    x0, y0 = tr[0][0], tr[0][1]
    xe = np.mean([t[0] for t in tr[-4:]]); ye = np.mean([t[1] for t in tr[-4:]])
    ymin = min(t[1] for t in tr)
    rise_max = y0 - ymin; rise_end = y0 - ye
    gdx = xe - goal[0]; gdy = ye - goal[1]
    miss = (gdx**2 + gdy**2) ** 0.5
    goal_h = y0 - goal[1]
    if rise_max < 5: verdict = "never-lifted (scrape/no-attach)"
    elif rise_end < 4 and rise_max >= 5: verdict = "lifted-then-dropped"
    elif miss <= 12: verdict = "near-goal-miss (<~4cm)"
    elif abs(gdx) < 12 and gdy > 0: verdict = "height-undershoot (right xy, %.0f%% of goal height)" % (100*rise_end/max(goal_h,1))
    else: verdict = "carried-off-target"
    rows.append((name, verdict, rise_max, rise_end, gdx, gdy))
print("%-14s %-46s %7s %7s %8s %8s" % ("clip","verdict","riseMax","riseEnd","dx_goal","dy_goal"))
for r in rows:
    print("%-14s %-46s %7.0f %7.0f %8.0f %8.0f" % r)
