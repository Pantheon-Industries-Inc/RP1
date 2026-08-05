import h5py, numpy as np
print("=== CANONICAL reacher.h5")
with h5py.File("/workspace/datasets_canon/lewm-reacher/reacher.h5","r") as f:
    print("keys:", {k: (f[k].shape, str(f[k].dtype)) for k in f.keys()})
    if "ep_len" in f:
        L = f["ep_len"][:]; print("episodes:", len(L), "len min/med/max:", L.min(), int(np.median(L)), L.max())
    a = f["action"][:200000]
    print("action shape", a.shape, "mean", np.nanmean(a,0), "std", np.nanstd(a,0),
          "min", np.nanmin(a), "max", np.nanmax(a))
    if "pixels" in f:
        px = f["pixels"][:64]
        print("pixels", px.shape, px.dtype, "mean", float(px.mean()), "std", float(px.std()))
    for c in ("qpos","qvel","observation"):
        if c in f:
            v = f[c][:100000]; print(c, v.shape, "mean", np.nanmean(v,0)[:4], "std", np.nanstd(v,0)[:4])
