"""Merge grasp-miss latents into the expert caches by episode replication, and
emit a small combined actions H5 for the actor.

Outputs:
  aug_fs1.pt   critic cache = expert fs1  +  REP copies of each miss episode
  aug_fs5.pt   actor  cache = derived from aug_fs1 (step_idx % 5 == 0, // 5)
  aug_actions.h5   {action, ep_offset} for expert + miss replicas so the actor's
                   blocks(e,t)=act_n[ep_off[e]+5t:+5] finds miss actions too.

Miss replicas are numbered CONTIGUOUSLY after the expert episodes (N_exp, N_exp+1,
...) so ep_offset stays a dense array indexable by episode id. Episode-uniform
samplers => miss share = M*REP / (N_exp + M*REP).

Usage:
  python merge_augment.py OUT_DIR REP EXPERT_FS1 MAIN_H5 MISS_FS1 MISS_ALL_H5
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from stable_worldmodel.trm import LatentCache

out_dir, rep = Path(sys.argv[1]), int(sys.argv[2])
expert_fs1, main_h5, miss_fs1, miss_all_h5 = sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
out_dir.mkdir(parents=True, exist_ok=True)

expert = LatentCache.load(expert_fs1)
exp_eps = expert.episodes()
N_exp = len(exp_eps)
assert set(exp_eps) == set(range(N_exp)), "expert episode ids must be 0..N-1 contiguous"
print(f"expert fs1: {len(expert.z)} rows, {N_exp} eps, dim {expert.latent_dim}")

miss = LatentCache.load(miss_fs1)
miss_eps = miss.episodes()
M = len(miss_eps)
assert miss.latent_dim == expert.latent_dim
print(f"miss fs1: {len(miss.z)} rows, {M} eps; replicating x{rep} -> {M*rep} eps "
      f"(share {M*rep/(N_exp+M*rep):.1%})")

# expert actions/offsets from the main H5 (actor reads only action + ep_offset)
with h5py.File(main_h5, "r") as h:
    exp_act = h["action"][:].astype(np.float32)          # (~2.01M, 5)
    exp_off = h["ep_offset"][:].astype(np.int64)         # (N_exp,)
with h5py.File(miss_all_h5, "r") as hm:
    miss_act = hm["action"][:].astype(np.float32)        # (M*90, 5)
    miss_off = hm["ep_offset"][:].astype(np.int64)       # (M,)
    miss_len = hm["ep_len"][:].astype(np.int64)          # (M,)

# ---- build augmented cache + combined actions, miss numbered N_exp.. contiguously
zs, eids, sids = [expert.z], [expert.episode_idx.long()], [expert.step_idx.long()]
comb_act = [exp_act]
new_off = list(exp_off)                                  # index by episode id
cursor = exp_act.shape[0]
gid = N_exp
mkeys = sorted(miss_eps)
for r in range(rep):
    for m in mkeys:
        rows = miss_eps[m]
        zs.append(miss.z[rows])
        sids.append(miss.step_idx[rows].long())
        eids.append(torch.full((len(rows),), gid, dtype=torch.long))
        a0, aL = int(miss_off[m]), int(miss_len[m])
        comb_act.append(miss_act[a0:a0 + aL])
        new_off.append(cursor)
        cursor += aL
        gid += 1

aug_z = torch.cat(zs); aug_eid = torch.cat(eids); aug_sid = torch.cat(sids)
n_aug_eps = N_exp + M * rep
LatentCache(z=aug_z, episode_idx=aug_eid, step_idx=aug_sid,
            meta={**(expert.meta or {}), "miss_eps": M * rep, "rep": rep}
            ).save(out_dir / "aug_fs1.pt")

mask = (aug_sid % 5 == 0)
LatentCache(z=aug_z[mask], episode_idx=aug_eid[mask], step_idx=aug_sid[mask] // 5,
            meta={"stride": 5, "miss_eps": M * rep}).save(out_dir / "aug_fs5.pt")

combined_actions = np.concatenate(comb_act).astype(np.float32)
combined_off = np.asarray(new_off, dtype=np.int64)
assert len(combined_off) == n_aug_eps
with h5py.File(out_dir / "aug_actions.h5", "w") as f:
    f.create_dataset("action", data=combined_actions)
    f.create_dataset("ep_offset", data=combined_off)
print(f"aug_actions.h5: {combined_actions.shape[0]} action rows, {n_aug_eps} eps")

# sanity: miss fs5 episode length must exceed max_delta(10)+4=14
fs5 = LatentCache.load(out_dir / "aug_fs5.pt")
mlens = [len(r) for e, r in fs5.episodes().items() if e >= N_exp]
print(f"OK. aug_fs1 {len(aug_z)} rows / {n_aug_eps} eps; "
      f"miss fs5 len min {min(mlens)} max {max(mlens)} (need >14)")
