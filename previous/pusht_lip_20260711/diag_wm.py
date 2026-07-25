"""Diagnose the PushT LeWM checkpoint quality.

(a) latent -> state ridge probe (is block pose decodable from the frozen latent?)
(b) open-loop predictor rollout error over 1..5 action blocks vs a shuffle floor
    (does the predictor track dynamics at planning horizons?)
"""

import numpy as np
import torch
import stable_pretraining as spt
from torchvision.transforms import v2 as transforms
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import stable_worldmodel as swm
from stable_worldmodel.trm.latent_cache import LatentCache

DEV = "cuda"
FS, HIST, HOR = 5, 3, 5
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- (a) probe
import sys
cache = LatentCache.load(sys.argv[1])
Z = cache.z.float().numpy()
S = cache.state.float().numpy()
n = len(Z)
idx = rng.permutation(n)
tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n) :]

names = ["agent_x", "agent_y", "block_x", "block_y", "block_angle", "vel_x", "vel_y"]
reg = Ridge(alpha=1.0).fit(Z[tr], S[tr])
pred = reg.predict(Z[te])
print("latent->state ridge R2 per dim:")
for i, nm in enumerate(names):
    print(f"  {nm:12s} {r2_score(S[te, i], pred[:, i]):.3f}", flush=True)

ang = S[:, 4]
tgt = np.stack([np.sin(ang), np.cos(ang)], 1)
reg2 = Ridge(alpha=1.0).fit(Z[tr], tgt[tr])
print(f"  block_angle sin/cos R2: {r2_score(tgt[te], reg2.predict(Z[te])):.3f}", flush=True)

# ---------------------------------------------------------------- (b) rollout
wm = swm.wm.utils.load_pretrained(sys.argv[2]).to(DEV).eval()
wm.requires_grad_(False)

ds = swm.data.load_dataset("pusht_play.lance")
tf = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ]
)

ep_col = np.asarray(ds.get_col_data("episode_idx"))
act_all = np.asarray(ds.get_col_data("action"), dtype=np.float32)
a_mu, a_sd = act_all.mean(0), act_all.std(0) + 1e-8


def _img(r):
    arr = np.asarray(ds[int(r)]["pixels"]).squeeze()
    if arr.shape[0] == 3:  # stored channel-first
        arr = arr.transpose(1, 2, 0)
    return tf(arr)


def enc_rows(rows):
    imgs = torch.stack([_img(r) for r in rows]).to(DEV)
    with torch.no_grad():
        return wm.encode({"pixels": imgs.unsqueeze(0)})["emb"][0]  # (T, D)


def act_block(row):
    a = (act_all[row : row + FS] - a_mu) / a_sd
    return a.reshape(-1)


errs = {k: [] for k in range(1, HOR + 1)}
one_step_true = []
finals_pred, finals_true = [], []
for _ in range(40):
    e = int(rng.choice(1000))
    rows = np.flatnonzero(ep_col == e)
    t0 = int(rng.integers((HIST - 1) * FS, len(rows) - HOR * FS - 1))
    base = rows[0]
    hist_rows = [base + t0 - (HIST - 1 - i) * FS for i in range(HIST)]
    z_hist = list(enc_rows(hist_rows).unbind(0))  # HIST tensors (D,)

    # action blocks aligned per history/rollout frame
    blocks = [act_block(base + t0 - (HIST - 1 - i) * FS) for i in range(HIST + HOR)]
    a_emb_all = wm.action_encoder(
        torch.tensor(np.stack(blocks), device=DEV).unsqueeze(0)
    )  # (1, HIST+HOR, D)

    embs = [z.unsqueeze(0) for z in z_hist]  # list of (1, D)
    for k in range(1, HOR + 1):
        lo = max(0, len(embs) - HIST)
        emb_trunc = torch.stack(embs[lo:], dim=1)  # (1, <=HIST, D)
        act_trunc = a_emb_all[:, lo : len(embs)]
        with torch.no_grad():
            nxt = wm.predict(emb_trunc, act_trunc)[:, -1]  # (1, D)
        embs.append(nxt)
        ztrue = enc_rows([base + t0 + k * FS])[0]
        errs[k].append(torch.norm(nxt[0] - ztrue).item())
        if k == HOR:
            finals_pred.append(nxt[0].cpu())
            finals_true.append(ztrue.cpu())

ft = torch.stack(finals_true)
fp = torch.stack(finals_pred)
sh = torch.tensor(rng.permutation(len(ft)))
floor_d = torch.norm(ft - ft[sh], dim=1).mean().item()
copy_d = []  # "no-dynamics" baseline: predict last history frame stays put
print("\nopen-loop rollout ||pred - true|| (40 windows):")
for k in range(1, HOR + 1):
    print(f"  block {k} ({k * FS:2d} steps): {np.mean(errs[k]):.3f}")
print(f"  shuffle floor (unrelated frames): {floor_d:.3f}")
cos = torch.nn.functional.cosine_similarity(fp, ft, dim=1).mean().item()
print(f"  cosine(pred, true) at block {HOR}: {cos:.3f}")

# persistence baseline: ||z(t) - z(t+25)|| — error if you predicted "nothing moves"
pers = []
for _ in range(40):
    e = int(rng.choice(1000))
    rows = np.flatnonzero(ep_col == e)
    t0 = int(rng.integers(0, len(rows) - HOR * FS - 1))
    z2 = enc_rows([rows[0] + t0, rows[0] + t0 + HOR * FS])
    pers.append(torch.norm(z2[0] - z2[1]).item())
print(f"  persistence baseline ||z_t - z_(t+25)||: {np.mean(pers):.3f}")
