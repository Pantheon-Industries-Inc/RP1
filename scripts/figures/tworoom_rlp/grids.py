"""Compute cost-to-go grids on a lattice; cache latent grid per base."""
import sys, os, time; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, torch
import twcore as C

def lattice(u, step=5, pad=2):
    lo = u.BORDER_SIZE + pad; hi = u.IMG_SIZE - u.BORDER_SIZE - pad
    xs = np.arange(lo, hi+1, step); ys = np.arange(lo, hi+1, step)
    XX, YY = np.meshgrid(xs, ys)  # (ny,nx)
    return xs, ys, XX, YY

def compute_latent_grid(base="lejepa", step=5, device="cpu"):
    u = C.make_env()
    wm, actor, value, meta = C.load_stack(base, 0, device=device)
    xs, ys, XX, YY = lattice(u, step)
    pos = np.stack([XX.ravel(), YY.ravel()],1).astype(np.float32)
    occ = C.occupancy(u)
    blocked = occ[np.clip(pos[:,1].astype(int),0,223), np.clip(pos[:,0].astype(int),0,223)]
    t0=time.time()
    Z = C.encode_pos(wm, u, pos, device=device, bs=96).numpy()  # (N,192)
    print(f"[{base}] encoded {len(pos)} cells in {time.time()-t0:.1f}s  grid {XX.shape}")
    np.savez(f"scripts/figures/tworoom_rlp/_latgrid_{base}.npz",
             xs=xs, ys=ys, XX=XX, YY=YY, pos=pos, Z=Z, blocked=blocked, step=step)
    return f"scripts/figures/tworoom_rlp/_latgrid_{base}.npz"

if __name__=="__main__":
    import sys
    base = sys.argv[1] if len(sys.argv)>1 else "lejepa"
    compute_latent_grid(base, step=int(sys.argv[2]) if len(sys.argv)>2 else 5)
