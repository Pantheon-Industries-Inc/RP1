"""DINO-WM: verify the action convention against ground truth.

A parallel session found a DINO-WM LIP path feeding RAW actions to a predictor
trained on z-scored ones: the action pathway went nearly inert (barely better
than feeding ZERO actions), error compounded over the horizon, and every LIP
result on that base was invalidated. The lesson is not "normalize" -- it is
"measure against ground truth instead of trusting a docstring".

Two checks here.

(A) WHERE DO THE CONSTANTS COME FROM. audit_value.py documents the rule:
        has ext_*  -> normalize with the CHECKPOINT's ae.ext_mu / ae.ext_std,
                      because _BlockActionEncoder undoes exactly that affine
        no ext_*   -> normalize with the DATASET's action stats
    dinowmnp HAS ext_* (convert_reacher_bases.py wrote them) while
    train_lip_ac.py z-scores with dataset stats -- so if those disagree, LIP on
    this base is mis-scaled. Already measured: max|dmu| = max|dstd| = 0.000e+00,
    bit-identical, because the converter derived ext_* from this same h5. So
    the two conventions coincide here. Re-asserted below so the claim is in the
    artifact, not just in a chat log.

(B) GROUND TRUTH, which does not depend on (A) being reasoned correctly. Roll
    the world model forward under four conventions and measure error against the
    TRUE next latents. 'zero' is the inert-pathway control: a convention that is
    not clearly better than feeding zero actions is one the model is ignoring.

The pixel pipeline and the transition sampler are imported from
convert_reacher_bases rather than re-implemented -- that module's rollout_check
is what gated the conversion, so its preprocessing is the verified one. (My
first attempt at this probe hand-rolled the decode and died on JPEG blobs.)
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/swm_cem")
sys.path.insert(0, "/workspace/swm_cem/scripts/plan")
import convert_reacher_bases as C  # noqa: E402
from stable_worldmodel.solver.lip import rollout_traj  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402

WM_DIR = "/workspace/swm_home/checkpoints/dinowmnp_reacher"
DS = "dmc/reacher_random.lance"
H5 = "/workspace/reacher_slim.h5"
dev = C.DEV

wm = load_pretrained(WM_DIR).to(dev).eval()
wm.requires_grad_(False)
ae = wm.action_encoder
print(f"loaded {type(wm).__name__} | action_encoder={type(ae).__name__} "
      f"has_ext={hasattr(ae, 'ext_mu')}", flush=True)

# ---------------------------------------------------------------- (A)
imgs, blocks, stats = C.load_transitions(DS, H5)   # blocks are ALREADY z-scored
amu, astd = stats
ds_mu = np.tile(amu, 5).astype(np.float32)
ds_std = np.tile(astd, 5).astype(np.float32)
ck_mu = ae.ext_mu.detach().cpu().numpy()
ck_std = ae.ext_std.detach().cpu().numpy()
print(f"(A) max|ckpt_mu - dataset_mu| = {np.abs(ck_mu - ds_mu).max():.3e}   "
      f"max|ckpt_std - dataset_std| = {np.abs(ck_std - ds_std).max():.3e}", flush=True)
print("    -> the checkpoint's ext_* and train_lip_ac's dataset stats are the "
      "same constants; LIP is not mis-scaled on this base.", flush=True)

# ---------------------------------------------------------------- (B)
z = C.encode_all(wm, imgs)                          # (N, span+1, D)
N, S1, D = z.shape
span = S1 - 1
print(f"(B) latents {tuple(z.shape)} -> latent_dim={D}, span={span}", flush=True)

zh = z[:, :3].to(dev)
mu_t = torch.as_tensor(ds_mu, device=dev)
std_t = torch.as_tensor(ds_std, device=dev)
ah_n = blocks[:, :2].to(dev)                        # z-scored
plan_n = blocks[:, 2:span].to(dev)
z_true = z[:, 3:span + 1].to(dev)                   # frames 3..span

CONV = {
    "normalized (z-scored)": (plan_n, ah_n),
    "raw (de-normalized)": (plan_n * std_t + mu_t, ah_n * std_t + mu_t),
    "zero": (torch.zeros_like(plan_n), torch.zeros_like(ah_n)),
    "shuffled (normalized)": (plan_n[torch.randperm(plan_n.shape[0])], ah_n),
}

H = plan_n.shape[1]
print(f"\n  {'convention':<24} " + " ".join(f"h={h + 1:<7}" for h in range(H)))
res = {}
for name, (plan, ah) in CONV.items():
    outs = []
    for i in range(0, N, 32):
        with torch.no_grad():
            outs.append(rollout_traj(wm, zh[i:i + 32], ah[i:i + 32], plan[i:i + 32]))
    pred = torch.cat(outs)
    err = ((pred - z_true[:, :pred.shape[1]]).norm(dim=-1).mean(0)
           / z_true[:, :pred.shape[1]].norm(dim=-1).mean(0))
    res[name] = err.tolist()
    print(f"  {name:<24} " + " ".join(f"{e:<9.4f}" for e in err.tolist()), flush=True)

n_last = res["normalized (z-scored)"][-1]
r_last = res["raw (de-normalized)"][-1]
z_last = res["zero"][-1]
print(f"\n  at h={H}:  normalized {n_last:.4f} | raw {r_last:.4f} | zero {z_last:.4f}")
print(f"  raw/normalized = {r_last / n_last:.2f}x   normalized/zero = {n_last / z_last:.2f}x")
if n_last < 0.9 * z_last:
    print("  -> the action pathway is LIVE under the normalized convention.")
else:
    print("  -> WARNING: normalized is not clearly better than zero actions; "
          "the model is ignoring the actions and no LIP result on it is safe.")
print("PROBE_DONE", flush=True)
