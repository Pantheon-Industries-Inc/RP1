"""Let the LIP-AC tandem critic carry the RANKING objective.

Without this, `train_lip_ac.py` keeps TD-training the critic during actor
training, which would erode the ranking structure we initialise from.
Adds --critic-rank-weight (planner-relevant hinge on the critic) and
--critic-td-weight (0 => pure-ranking critic throughout the tandem).
Idempotent.
"""
from pathlib import Path

p = Path("/workspace/code/stable-worldmodel/scripts/plan/train_lip_ac.py")
t = p.read_text()

if "critic_rank_weight" in t or "critic-rank-weight" in t:
    print("already patched")
    raise SystemExit(0)

# 1) CLI
a = '    p.add_argument("--action-stats-pin"'
assert t.count(a) == 1, "cli anchor"
t = t.replace(a, '    p.add_argument("--critic-rank-weight", type=float, default=0.0,\n'
                 '                   help="planner-relevant ranking hinge on the tandem critic")\n'
                 '    p.add_argument("--critic-td-weight", type=float, default=1.0,\n'
                 '                   help="weight on the critic TD term (0 = pure-ranking critic)")\n'
                 '    p.add_argument("--critic-rank-margin", type=float, default=0.5)\n' + a)

# 2) rank triplet sampler over the fs1 critic cache, built next to td_sampler
a = "    c_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=a.critic_wd)"
assert t.count(a) == 1, "c_opt anchor"
t = t.replace(a, '''    if a.critic_rank_weight > 0:
        _ck_eps = c_td.episodes()
        _ck_keys = [k for k in _ck_eps if len(_ck_eps[k]) > 3]
        _ck_rows = {k: np.asarray(_ck_eps[k]) for k in _ck_keys}
        _ck_rng = np.random.default_rng(a.seed + 7)

        def _crank_batch(bs):
            near = np.empty(bs, np.int64); far = np.empty(bs, np.int64)
            goal = np.empty(bs, np.int64); gap = np.empty(bs, np.float32)
            for b_ in range(bs):
                r = _ck_rows[_ck_keys[_ck_rng.integers(len(_ck_keys))]]
                L = len(r)
                d_far = int(_ck_rng.integers(2, L))
                d_near = int(_ck_rng.integers(1, d_far))
                gi = int(_ck_rng.integers(d_far, L))
                near[b_], far[b_], goal[b_] = r[gi - d_near], r[gi - d_far], r[gi]
                gap[b_] = d_far - d_near
            return (c_td.z[near], c_td.z[far], c_td.z[goal], torch.from_numpy(gap))

''' + a)

# 3) critic loss: scale TD term, add the hinge
a = "        loss = _expectile_loss(critic(z_t, z_g) - tgt, tau, a.huber_beta)"
assert t.count(a) == 1, "critic loss anchor"
t = t.replace(a, "        loss = a.critic_td_weight * _expectile_loss(critic(z_t, z_g) - tgt, tau, a.huber_beta)")

a = """            loss = loss + a.expand_weight * _expectile_loss(
                critic(z0e, zge) - tgt_e, tau, a.huber_beta)"""
assert t.count(a) == 1, "expand anchor"
t = t.replace(a, """            loss = loss + a.critic_td_weight * a.expand_weight * _expectile_loss(
                critic(z0e, zge) - tgt_e, tau, a.huber_beta)""")

a = "        c_opt.zero_grad(set_to_none=True)\n        loss.backward()\n        c_opt.step()"
assert t.count(a) == 1, "critic opt anchor"
t = t.replace(a, """        if a.critic_rank_weight > 0:
            zn_, zf_, zg_, gap_ = _crank_batch(a.td_batch)
            zn_, zf_, zg_, gap_ = zn_.to(dev), zf_.to(dev), zg_.to(dev), gap_.to(dev)
            loss = loss + a.critic_rank_weight * torch.relu(
                a.critic_rank_margin * gap_ + critic(zn_, zg_) - critic(zf_, zg_)).mean()
        c_opt.zero_grad(set_to_none=True)
        loss.backward()
        c_opt.step()""")

p.write_text(t)
print("train_lip_ac.py: critic ranking objective added")
print("LIPAC_RANK_PATCH_OK")
