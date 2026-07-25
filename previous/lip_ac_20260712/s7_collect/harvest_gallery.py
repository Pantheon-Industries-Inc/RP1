import re, sys, shutil, os
import numpy as np
import imageio.v2 as imageio
DRAW = sys.argv[1]
TXT = "/workspace/ckpts/gallery_s%s.txt" % DRAW
GAL = "/workspace/failure_gallery"
os.makedirs(GAL, exist_ok=True)
s = open(TXT).read()
m = re.findall(r"episode_successes.:\s*array\((.*?)\)", s, re.S)[-1]
bools = [b == "True" for b in re.findall(r"True|False", m)]
fails = [i for i, b in enumerate(bools) if not b]
print("draw s%s: %d/%d success, fails: %s" % (DRAW, sum(bools), len(bools), fails))
def cube_track(path):
    frames = imageio.mimread(path, memtest=False)
    pts = []
    for f in frames:
        f = np.asarray(f)
        mx = f.max(2).astype(int); mn = f.min(2).astype(int)
        sat = mx - mn
        mask = (sat > 70) & (mx > 80)
        ys, xs = np.nonzero(mask)
        if len(xs) < 12: pts.append((np.nan, np.nan, 0)); continue
        pts.append((float(xs.mean()), float(ys.mean()), int(len(xs))))
    return frames, np.array(pts)
for i in fails:
    src = "/workspace/ckpts/env_%d.mp4" % i
    if not os.path.exists(src): print("  MISSING", src); continue
    dst = "%s/s%s_task%02d.mp4" % (GAL, DRAW, i)
    shutil.copy(src, dst)
    frames, tr = cube_track(dst)
    x, y, area = tr[:,0], tr[:,1], tr[:,2]
    v = ~np.isnan(x)
    if v.sum() < 5:
        verdict = "cube-not-tracked"
        print("  task %02d: %s" % (i, verdict)); continue
    y0 = np.nanmedian(y[:5]); x0 = np.nanmedian(x[:5])
    rise = y0 - np.nanmin(y)          # pixels up
    end_rise = y0 - np.nanmedian(y[-5:])
    drift = abs(np.nanmedian(x[-5:]) - x0)
    moved = (abs(y - y0) + abs(x - x0))[v].max()
    if moved < 4: verdict = "untouched (block never disturbed)"
    elif rise < 6: verdict = "contact-no-attach (nudged, never lifted)"
    elif rise >= 6 and end_rise < 4: verdict = "lifted-then-fell/released (rise %.0fpx, ends low)" % rise
    else: verdict = "lifted-but-wrong-place (rise %.0fpx, ends %.0fpx up, drift %.0fpx)" % (rise, end_rise, drift)
    ks = [0, int(np.nanargmin(y)) if v.any() else len(frames)//2, len(frames)-1]
    for j, t in enumerate(ks):
        imageio.imwrite("%s/s%s_task%02d_kf%d.png" % (GAL, DRAW, i, j), frames[min(t, len(frames)-1)])
    print("  task %02d: %s  [frames=%d]" % (i, verdict, len(frames)))
