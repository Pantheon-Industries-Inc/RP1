"""Teacher-forced (training-convention) vs autoregressive (eval-convention)
1-step latent prediction for the dinowmnp_reacher base.

dinowm code path ONLY: stable_worldmodel.wm.dinowm.DinoWMTokens via load_pretrained.
No PreJEPA wrapper is imported or instantiated (the checkpoint config's
prejepa.module.{create_backbone,CausalPredictor,Embedder} are DinoWMTokens'
own building blocks and are constructed by hydra from the shipped config.json).

Forward passes only. Writes nothing into the repo.
"""
import io
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained
from stable_worldmodel.data.normalization import ZScoreScaler

DEV = "cuda:0"
FS = 5
NFRAMES = 7          # frames 0..6 -> 6 action blocks; TF uses frames 0..3 / blocks 0..2
N = 256              # windows
EP_LO, EP_HI = 5000, 10000   # held out from the converter's draw (it used episodes[:400])
SEED = 7
torch.manual_seed(0)

# ------------------------------------------------------------------ data
ds = swm.data.load_dataset("dmc/reacher_random.lance")
ep = np.asarray(ds.get_col_data("episode_idx")).reshape(-1)
st = np.asarray(ds.get_col_data("step_idx")).reshape(-1)
act_raw = np.asarray(ds.get_col_data("action"))            # (rows, 2)

# EXACTLY what training used: get_column_normalizer(dataset, 'action', 'action')
scaler = ZScoreScaler().fit(act_raw)
amu = scaler.mean.ravel().astype(np.float64)
astd = scaler.std.ravel().astype(np.float64)
act_n = ((act_raw - amu) / np.maximum(astd, scaler.eps)).astype(np.float32)
print(f"[actnorm] train-convention (ZScoreScaler on lance 'action') mu {amu.tolist()} std {astd.tolist()}")
with open("/workspace/reacher_action_stats.json") as f:
    js = json.load(f)
print(f"[actnorm] converter/h5 stats               mu {js['mean']} std {js['std']}")
print(f"[actnorm] max rel diff vs converter: mean {np.max(np.abs(np.array(js['mean'])-amu)):.3e} "
      f"std {np.max(np.abs(np.array(js['std'])-astd)/astd):.3e}")

rng = np.random.default_rng(SEED)
need = FS * NFRAMES                       # rows spanned
eps = np.unique(ep)
eps = eps[(eps >= EP_LO) & (eps < EP_HI)]
rng.shuffle(eps)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dec(p):
    if isinstance(p, (bytes, bytearray, np.bytes_)):
        return np.array(Image.open(io.BytesIO(bytes(p))))
    return np.asarray(p)


px, blocks = [], []
for e in eps:
    rows = np.nonzero(ep == e)[0]
    rows = rows[np.argsort(st[rows])]
    if len(rows) <= need + 1:
        continue
    t0 = int(rng.integers(0, len(rows) - need - 1))
    frame_rows = rows[t0: t0 + need: FS][:NFRAMES]
    assert len(frame_rows) == NFRAMES
    rd = ds.get_row_data(list(frame_rows))
    px.append(np.stack([dec(p) for p in rd["pixels"]]))
    g0 = int(rows[0]) + int(st[rows[t0]])          # global row of frame 0
    assert g0 == int(frame_rows[0])
    blocks.append(np.stack([act_n[g0 + FS * j: g0 + FS * j + FS].reshape(-1)
                            for j in range(NFRAMES - 1)]))
    if len(px) >= N:
        break

imgs_u8 = torch.from_numpy(np.stack(px))                       # (N,7,224,224,3)
blocks = torch.from_numpy(np.stack(blocks))                    # (N,6,10)
imgs = imgs_u8.permute(0, 1, 4, 2, 3).float() / 255.0
imgs = (imgs - MEAN.unsqueeze(0)) / STD.unsqueeze(0)           # ImageNet norm, 224px
print(f"[data] windows {tuple(imgs.shape)} blocks {tuple(blocks.shape)} "
      f"episodes {EP_LO}..{EP_HI} (disjoint from converter's episodes[:400])")

# torchvision-Resize variant (training: ToImage(ImageNet) -> Resize(196))
from torchvision.transforms import v2 as tvv2
res196 = tvv2.Resize(196, antialias=True)
imgs196 = res196(imgs.reshape(-1, 3, 224, 224)).reshape(imgs.shape[0], NFRAMES, 3, 196, 196)

# ------------------------------------------------------------------ model
wm = load_pretrained("/workspace/swm_home/checkpoints/dinowmnp_reacher").to(DEV).eval()
print(f"[model] {type(wm).__module__}.{type(wm).__name__} latent_dim {wm.latent_dim} "
      f"img_size {wm.img_size} history {wm.history_size} tok_dim {wm.tok_dim}")
print(f"[model] pred_bias: norm {wm.pred_bias.norm().item():.6f} "
      f"(must be 0 for the uncalibrated base)")

ae = wm.action_encoder
print(f"[actenc] ext_mu   {ae.ext_mu.tolist()}")
print(f"[actenc] train_mu {ae.train_mu.tolist()}")
print(f"[actenc] ext_std  {ae.ext_std.tolist()}")
print(f"[actenc] train_std{ae.train_std.tolist()}")
ident = (torch.allclose(ae.ext_mu, ae.train_mu) and torch.allclose(ae.ext_std, ae.train_std))
print(f"[actenc] re-map identity: {ident}")


@torch.no_grad()
def encode(x, bs=8):
    out = []
    for i in range(0, x.shape[0], bs):
        out.append(wm.encode({"pixels": x[i:i + bs].to(DEV)})["emb"].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def tf_pred(emb, aemb, bs=16):
    out = []
    for i in range(0, emb.shape[0], bs):
        out.append(wm.predict(emb[i:i + bs].to(DEV), aemb[i:i + bs].to(DEV)).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def ar_pred(emb3, plan, a_hist, bs=16):
    out = []
    for i in range(0, emb3.shape[0], bs):
        ah = None if a_hist is None else a_hist[i:i + bs].to(DEV)
        out.append(wm.rollout_traj(emb3[i:i + bs].to(DEV), plan[i:i + bs].to(DEV),
                                   a_hist=ah).float().cpu())
    return torch.cat(out)


def mse(a, b):
    return (a - b).pow(2).mean().item()


def offset_stats(pred, tgt):
    """pred,tgt: (n, D). c = mean_n(pred-tgt)."""
    d = (pred - tgt).double()
    c = d.mean(0)
    n = d.shape[0] // 2
    cA, cB = d[:n].mean(0), d[n:].mean(0)
    cos = F.cosine_similarity(cA.unsqueeze(0), cB.unsqueeze(0)).item()
    tn = tgt.double().norm(dim=1).pow(2).mean().sqrt().item()   # rms ||target||
    D = d.shape[1]
    return dict(c_norm=c.norm().item(), rel=c.norm().item() / tn, cos=cos,
                mse=mse(pred, tgt), mse_debiased=(d - c).pow(2).mean().item(),
                frac_mse_from_c=(c.pow(2).sum().item() / D) / max(mse(pred, tgt), 1e-12),
                tgt_rms_norm=tn)


emb = encode(imgs)                       # (N,7,D)
emb196 = encode(imgs196)
print(f"[enc] emb {tuple(emb.shape)} | resize-path latent max|d| "
      f"{(emb - emb196).abs().max().item():.4e} rel rms "
      f"{((emb-emb196).pow(2).mean().sqrt()/emb.pow(2).mean().sqrt()).item():.3e}")

with torch.no_grad():
    aemb = ae(blocks.to(DEV)).float().cpu()                     # (N,6,10)
    aemb_bypass = ae.embedder(blocks.to(DEV)).float().cpu()     # skip the ext->train re-map
print(f"[actenc] bypass-vs-remap act_emb max|d| {(aemb - aemb_bypass).abs().max().item():.3e}")

rows = []

# ---------------- (A) TEACHER FORCED, exactly the training loss -------------
pred_tf = tf_pred(emb[:, :3], aemb[:, :3])                      # (N,3,D)
tgt_tf = emb[:, 1:4]
e_pred = mse(pred_tf, tgt_tf)
e_copy = mse(emb[:, :3], tgt_tf)
print("\n================ TEACHER-FORCED (training convention) ================")
print(f"overall  pred {e_pred:.6f} | copy-last {e_copy:.6f} | ratio {e_pred/e_copy:.4f}")
rows.append(("TF overall (t=0,1,2)", e_pred, e_copy, e_pred / e_copy, None))
for t in range(3):
    ep_t = mse(pred_tf[:, t], tgt_tf[:, t])
    ec_t = mse(emb[:, t], tgt_tf[:, t])
    o = offset_stats(pred_tf[:, t], tgt_tf[:, t])
    print(f"  pos {t} -> frame {t+1}: pred {ep_t:.6f} copy {ec_t:.6f} ratio {ep_t/ec_t:.4f} "
          f"| ||c|| {o['c_norm']:.3f} rel {o['rel']:.4f} splithalf-cos {o['cos']:.5f} "
          f"| debiased mse {o['mse_debiased']:.6f} (ratio {o['mse_debiased']/ec_t:.4f}) "
          f"| c explains {100*o['frac_mse_from_c']:.1f}% of mse")
    rows.append((f"TF pos{t}->f{t+1}", ep_t, ec_t, ep_t / ec_t, o))
o_all = offset_stats(pred_tf.reshape(-1, wm.latent_dim), tgt_tf.reshape(-1, wm.latent_dim))
print(f"  pooled offset: ||c|| {o_all['c_norm']:.3f} rel-to-||target|| {o_all['rel']:.4f} "
      f"splithalf-cos {o_all['cos']:.5f} ||target||rms {o_all['tgt_rms_norm']:.2f}")

# TF with the re-map bypassed
pred_tf_by = tf_pred(emb[:, :3], aemb_bypass[:, :3])
print(f"  [remap bypass] overall ratio {mse(pred_tf_by, tgt_tf)/e_copy:.4f} "
      f"(pos2 {mse(pred_tf_by[:,2], tgt_tf[:,2])/mse(emb[:,2], tgt_tf[:,2]):.4f})")

# TF on the torchvision-resize latents (self-consistent target)
pred_tf_r = tf_pred(emb196[:, :3], aemb[:, :3])
tgt_r = emb196[:, 1:4]
print(f"  [tv-Resize(196) pixels] overall ratio "
      f"{mse(pred_tf_r, tgt_r)/mse(emb196[:, :3], tgt_r):.4f}")

# ---------------- (B) AUTOREGRESSIVE ---------------------------------------
print("\n================ AUTOREGRESSIVE (eval / LIP convention) =============")
H = NFRAMES - 3            # plan blocks 2..5 -> frames 3..6
plan = blocks[:, 2:2 + H]
for name, ah in (("real a_hist (blocks 0,1)", blocks[:, :2]), ("zero a_hist (LIP)", None)):
    tr = ar_pred(emb[:, :3], plan, ah)                          # (N,H,D)
    print(f"-- {name}")
    for k in range(H):
        tgt = emb[:, 3 + k]
        ep_k = mse(tr[:, k], tgt)
        ec_k = mse(emb[:, 2], tgt)                              # copy-last = last observed
        o = offset_stats(tr[:, k], tgt)
        print(f"   step {k+1} (frame {3+k}): pred {ep_k:.6f} copy-last {ec_k:.6f} "
              f"ratio {ep_k/ec_k:.4f} | ||c|| {o['c_norm']:.3f} rel {o['rel']:.4f} "
              f"cos {o['cos']:.5f} | debiased ratio {o['mse_debiased']/ec_k:.4f}")
        rows.append((f"AR[{ 'real' if ah is not None else 'zero'}] step{k+1}->f{3+k}",
                     ep_k, ec_k, ep_k / ec_k, o))
    if ah is not None:
        d = (tr[:, 0] - pred_tf[:, 2]).abs().max().item()
        print(f"   identity check: AR step1(real a_hist) vs TF pos2 max|d| {d:.3e}")

print("\nDONE")
