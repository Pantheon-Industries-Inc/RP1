"""Add --replay-prob and --expand-weight to the patched PWM trainer.

WHY. Paper PWM trains ONLINE: the policy's own visited states enter training.
Our port is fully offline from a static expert cache, and we gave it no
substitute -- while giving LIPv4 two (--replay-prob, --expand-weight, together
worth +10.6 on PLDM). Every PWM-vs-LIP gap we have reported is therefore an
upper bound on LIP's real advantage. This closes that, so the head-to-head
becomes an ablation of the ACTOR ARCHITECTURE alone:

    same critic, same teacher, same objective (--terminal), same replay/expand,
    same amax, same caches  ->  differing only in feedforward policy vs
                                iterative plan refiner

WHAT EACH DOES HERE, and how it differs from LIP:

  --replay-prob   LIP substitutes the actor's z0 CONTEXT, because its actor is
                  called once per inner iterate on a fixed z0 and never sees its
                  own downstream states as input. PWM's actor IS already called
                  at self-generated latents for t=1..H-1 inside the rollout, so
                  it self-conditions over 4 imagined steps for free. What it
                  never sees is a start state drawn from its own distribution --
                  z0 is always expert-cache, while at deploy it runs 50-200 of
                  its own steps. So here replay seeds z0 from previously-imagined
                  ENDPOINTS, extending self-conditioning past H and compounding
                  across training steps. Smaller correction than for LIP, but real.

  --expand-weight identical in spirit to LIP: regress the critic at the actor's
                  own imagined (z0, z_H, zg) with target
                  plan_cost + plan_disc * teacher(z_H, zg). If anything it
                  matters MORE for PWM, whose dense objective reads the critic at
                  every one of H imagined states, where LIP reads it at the
                  rollout endpoint per iterate.
                  CAVEAT worth remembering when reading results: PWM's
                  --freeze-at defaults to 1.0, i.e. its critic NEVER freezes.
                  Expand on a permanently-live critic amplifies exactly the
                  co-adaptation that LIP's early freeze exists to prevent, so a
                  freeze fraction should usually be set alongside it.

plan_cost / plan_disc reuse LIP's convention: n_plan = horizon * fs primitive
steps with fs = 5, so 25 steps -- 25.0/1.0 at gamma 1.0, 19.83/0.604 at 0.98.

Patches the CUBE COPY (train_pwm_ac_cube.py), which already carries the a_dim
fix. The parallel session's committed train_pwm_ac.py is never touched.

Gates: every anchor must appear exactly once, the result must compile, --help
must expose both new flags.
"""

import py_compile
import subprocess
import sys

DST = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/code/stable-worldmodel/scripts/plan/train_pwm_ac_cube.py"

EDITS = [
    # ---- 1. the two new flags
    ("    p.add_argument('--max-delta', type=int, default=12)",
     "    p.add_argument('--max-delta', type=int, default=12)\n"
     "    p.add_argument('--replay-prob', type=float, default=0.0,\n"
     "                   help=\"fraction of the actor batch whose z0 is reseeded from the \"\n"
     "                        \"previous step's own imagined endpoint (offline stand-in for \"\n"
     "                        \"PWM's online interaction)\")\n"
     "    p.add_argument('--expand-weight', type=float, default=0.0,\n"
     "                   help='weight of the TD backup along the planner rollout, i.e. fit '\n"
     "                        'the critic where the actor actually queries it (0 = off)')"),

    # ---- 2. plan_cost / plan_disc, defined just before critic_step
    ("    def critic_step(tau: float) -> float:",
     "    fs = 5                                    # primitive steps per action block\n"
     "    n_plan = a.horizon * fs\n"
     "    if a.gamma >= 1.0:\n"
     "        plan_cost, plan_disc = float(n_plan), 1.0\n"
     "    else:\n"
     "        plan_disc = a.gamma ** n_plan\n"
     "        plan_cost = (1.0 - plan_disc) / (1.0 - a.gamma)\n"
     "\n"
     "    def critic_step(tau: float, expand=None) -> float:"),

    # ---- 3. the expand term inside the critic loss
    ("        loss = _expectile_loss(critic(z_t, z_g) - tgt, tau, a.huber_beta)\n"
     "        c_opt.zero_grad(set_to_none=True)",
     "        loss = _expectile_loss(critic(z_t, z_g) - tgt, tau, a.huber_beta)\n"
     "        if expand is not None:              # value expansion on planner rollouts\n"
     "            z0e, zTe, zge = expand\n"
     "            with torch.no_grad():\n"
     "                tgt_e = plan_cost + plan_disc * teacher(zTe, zge)\n"
     "            loss = loss + a.expand_weight * _expectile_loss(\n"
     "                critic(z0e, zge) - tgt_e, tau, a.huber_beta)\n"
     "        c_opt.zero_grad(set_to_none=True)"),

    # ---- 4. buffers, declared before the training loop
    ("    pbar = tqdm(range(a.steps), desc=f'pwm-ac(H={a.horizon},g={a.gamma})')",
     "    replay_buf = {'z0': None, 'zg': None}\n"
     "    expand = None\n"
     "    pbar = tqdm(range(a.steps), desc=f'pwm-ac(H={a.horizon},g={a.gamma})')"),

    # ---- 5. feed expand into the critic, consuming each actor batch once
    ("        if step < freeze_step:\n"
     "            for _ in range(a.critic_ratio):\n"
     "                c_loss = critic_step(tau)",
     "        if step < freeze_step:\n"
     "            for _ in range(a.critic_ratio):\n"
     "                c_loss = critic_step(tau, expand if a.expand_weight > 0 else None)\n"
     "                expand = None               # consume each actor batch once"),

    # ---- 6. replay substitution, before the history window is built
    ("        B = z0.shape[0]\n"
     "        z_hist = z0.unsqueeze(1).expand(-1, hs, -1).contiguous()",
     "        B = z0.shape[0]\n"
     "        if a.replay_prob > 0 and replay_buf['z0'] is not None:\n"
     "            k = int(a.replay_prob * B)\n"
     "            if k > 0:\n"
     "                idx = torch.randperm(B, device=dev)[:k]\n"
     "                take = torch.randint(0, replay_buf['z0'].shape[0], (k,), device=dev)\n"
     "                z0 = z0.clone(); zg = zg.clone()\n"
     "                z0[idx] = replay_buf['z0'][take]\n"
     "                zg[idx] = replay_buf['zg'][take]\n"
     "        z_hist = z0.unsqueeze(1).expand(-1, hs, -1).contiguous()"),

    # ---- 7. capture the rollout for the next step's replay / expand
    ("        a_loss = (-J / a.horizon).mean()",
     "        if a.replay_prob > 0:\n"
     "            replay_buf['z0'] = z.detach()\n"
     "            replay_buf['zg'] = zg.detach()\n"
     "        if a.expand_weight > 0:\n"
     "            expand = (z0.detach(), z.detach(), zg.detach())\n"
     "        a_loss = (-J / a.horizon).mean()"),
]


def main():
    src = open(DST).read()
    if "--replay-prob" in src:
        print("already patched -- nothing to do")
        print("PATCH_PWM_RE_OK")
        return
    for i, (old, new) in enumerate(EDITS, 1):
        n = src.count(old)
        if n != 1:
            sys.exit(f"FATAL: edit {i} anchor found {n} times, expected 1:\n"
                     f"{old[:120]!r}")
        src = src.replace(old, new)
    open(DST, "w").write(src)
    py_compile.compile(DST, doraise=True)
    r = subprocess.run([sys.executable, DST, "--help"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("FATAL: --help failed\n" + r.stderr[-1500:])
    for flag in ("--replay-prob", "--expand-weight", "--terminal"):
        if flag not in r.stdout:
            sys.exit(f"FATAL: {flag} missing from --help")
    print(f"patched {DST}: 7 edits, compiles, --help exposes "
          f"--replay-prob / --expand-weight / --terminal")
    print("PATCH_PWM_RE_OK")


if __name__ == "__main__":
    main()
