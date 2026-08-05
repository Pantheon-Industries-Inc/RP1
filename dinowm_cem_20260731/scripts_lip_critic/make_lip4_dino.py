"""Generate the LIPv4-on-DINO stack: solver additions + train_lip_ac_dino.py.

Port strategy: train_lip_ac.py (the recipe behind LeWM's 87.6) is copied and
ONLY its data/rollout layer is swapped -- the critic block, teacher EMA,
schedules, v4 actor and loss are untouched. Swaps:

  * zh/ah (pooled fs5 latents + h5 action history)  ->  toks: 3-frame pixel
    history fetched from the lance and encoded to (B,3,196,394) tokens with
    ZERO action history (the deploy-time convention).
  * rollout_traj(wm, zh, ah, X)  ->  rollout_traj_dino(wm, toks, X*ast5+amu5)
    (PlannerNet plans in normalized action space; the dino WM consumes raw --
    the convention validated end-to-end by the v1 actor's 70.0).
  * z0 = pooled current-frame pixel tokens (deploy-exact).
  * saved kind = "lip4_dino" + amu5/ast5/img_size for the solver.

Solver: adds rollout_traj_dino (trajectory variant of rollout_terminal_dino),
whitelists kind "lip4_dino", and routes it through _proposal_lip_dino -- whose
existing net call `actor(A, gA, E, z0, zg)` is already correct for a v4 net
(use_z0/use_zg are False, so z0/zg are ignored; E there equals the training
E_feat because A is unchanged since the grad pass).

Not ported (asserted off): --replay-prob (needs token-level buffers; pooled
replay windows cannot be fed back into a token rollout) and arch != v4.
"""

import subprocess
import sys
from pathlib import Path

PLAN = Path('/workspace/code/stable-worldmodel/scripts/plan')
LIP = Path('/workspace/code/stable-worldmodel/stable_worldmodel/solver/lip.py')
SRC, DST = PLAN / 'train_lip_ac.py', PLAN / 'train_lip_ac_dino.py'


def apply(text, edits, tag):
    for old, new, n_expect in edits:
        n = text.count(old)
        if n != n_expect:
            sys.exit(f'[{tag}] anchor x{n} (want {n_expect}):\n{old[:120]}')
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------- solver
lip = LIP.read_text()
if 'def rollout_traj_dino(' not in lip:
    lip = apply(lip, [
        # trajectory rollout, appended after the terminal variant
        ("    return toks[-1][..., :pix_dim].mean(dim=1)\n",
         "    return toks[-1][..., :pix_dim].mean(dim=1)\n"
         "\n"
         "\n"
         "def rollout_traj_dino(wm, toks_hist, plan, pix_dim=384, act_emb_dim=10):\n"
         '    """Trajectory variant of :func:`rollout_terminal_dino`: all T imagined\n'
         "    pooled pixel embeddings (B, T, pix_dim). Same action-injection rollout,\n"
         '    differentiable w.r.t. ``plan`` (raw, denormalized action units)."""\n'
         "    toks = list(toks_hist.unbind(1))\n"
         "    act_enc = wm.extra_encoders[\"action\"]\n"
         "    outs = []\n"
         "    for t in range(plan.shape[1]):\n"
         "        a_emb = act_enc(plan[:, t].unsqueeze(1))[:, 0]\n"
         "        last = toks[-1]\n"
         "        last = torch.cat([last[..., :-act_emb_dim],\n"
         "                          a_emb.unsqueeze(1).expand(-1, last.shape[1], -1)], dim=-1)\n"
         "        toks[-1] = last\n"
         "        win = torch.stack(toks[-3:], dim=1)\n"
         "        nxt = wm.predict(win)[:, -1]\n"
         "        toks.append(nxt)\n"
         "        outs.append(nxt[..., :pix_dim].mean(dim=1))\n"
         "    return torch.stack(outs, dim=1)\n", 1),
        # accept the new checkpoint kind
        ('if self.kind not in ("lip", "lip2", "lip3", "lip4", "lip4r", "lip_dino"):',
         'if self.kind not in ("lip", "lip2", "lip3", "lip4", "lip4r", "lip_dino", "lip4_dino"):', 1),
        # amu/ast load + solve() dispatch (both sites, deliberately)
        ('if self.kind == "lip_dino":',
         'if self.kind in ("lip_dino", "lip4_dino"):', 2),
    ], 'solver')
    LIP.write_text(lip)
    print('solver patched')
else:
    print('solver already patched')

# ---------------------------------------------------------------- trainer
t = SRC.read_text()

t = apply(t, [
    # imports
    ("from stable_worldmodel.solver.lip import (\n"
     "    PlannerNet, PlannerNetV3, PlannerNetRec, rollout_terminal, rollout_traj)",
     "from stable_worldmodel.solver.lip import (\n"
     "    PlannerNet, PlannerNetV3, PlannerNetRec, rollout_terminal, rollout_traj,\n"
     "    rollout_terminal_dino, rollout_traj_dino)", 1),
    # new args
    ('    p.add_argument("--seed", type=int, default=0)',
     '    p.add_argument("--dataset", required=True, help="lance dataset (pixel source)")\n'
     '    p.add_argument("--img-size", type=int, default=196)\n'
     '    p.add_argument("--ckpt-every", type=int, default=0,\n'
     '                   help="save tagged actor+value snapshots every N steps (0=off)")\n'
     '    p.add_argument("--seed", type=int, default=0)', 1),
    # DINO data layer, inserted just before sample()
    ("    def sample(B):\n",
     "    # -------- DINO (PreJEPA) data layer: pixels from lance, tokens on the fly\n"
     '    assert a.arch == "v4", "dino port: --arch v4 only"\n'
     '    assert a.replay_prob == 0, "dino port: replay needs token buffers (not ported)"\n'
     "    from io import BytesIO\n"
     "    from PIL import Image\n"
     "    _ds_px = swm.data.load_dataset(a.dataset)\n"
     "    _IM = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)\n"
     "    _IS = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)\n"
     "    ast5 = torch.from_numpy(np.tile(astd, fs).astype(np.float32)).to(dev)\n"
     "    amu5 = torch.from_numpy(np.tile(amu, fs).astype(np.float32)).to(dev)\n"
     "    _SPE = 201                                  # lance rows per episode\n"
     "\n"
     "    def _decode_px(q):\n"
     "        if isinstance(q, (bytes, bytearray, np.bytes_)):\n"
     "            return np.array(Image.open(BytesIO(bytes(q))))\n"
     "        return np.asarray(q)\n"
     "\n"
     "    def encode_toks(rows_flat, B):\n"
     "        # lance DEDUPLICATES repeated row indices -> fetch unique, scatter back\n"
     "        uniq, inv = np.unique(np.asarray(rows_flat), return_inverse=True)\n"
     "        px = _ds_px.get_row_data(uniq.tolist())[\"pixels\"]\n"
     "        if isinstance(px, np.ndarray) and px.dtype == object:\n"
     "            px = np.stack([_decode_px(q) for q in px])\n"
     "        else:\n"
     "            px = np.asarray(px)\n"
     "            if px.ndim != 4:\n"
     "                px = np.stack([_decode_px(q) for q in px])\n"
     "        assert len(px) == len(uniq), \"lance dedup mismatch\"\n"
     "        px = px[inv].reshape(B, 3, *px.shape[1:])\n"
     "        x = torch.from_numpy(px).to(dev).permute(0, 1, 4, 2, 3).float() / 255.0\n"
     "        x = (x - _IM) / _IS                     # normalise, THEN resize\n"
     "        if x.shape[-1] != a.img_size:\n"
     "            x = torch.nn.functional.interpolate(\n"
     "                x.reshape(B * 3, *x.shape[2:]), size=a.img_size, mode=\"bilinear\",\n"
     "                antialias=True, align_corners=False\n"
     "            ).reshape(B, 3, 3, a.img_size, a.img_size)\n"
     "        with torch.no_grad():\n"
     "            info = wm.encode({\"pixels\": x,\n"
     "                              \"action\": torch.zeros(B, 3, a_dim, device=dev)})\n"
     "        return info[\"emb\"].float()             # (B, 3, P, 394)\n"
     "\n"
     "    def sample(B):\n", 1),
    # sample(): rows instead of zh/ah
    ("        zh, ah, zg, aref = [], [], [], []\n",
     "        rows_flat, zg, aref = [], [], []\n", 1),
    ("            zh.append(torch.stack([z[rows[t - 2]], z[rows[t - 1]], z[rows[t]]]))\n"
     "            ah.append(np.stack([blocks(e, t - 2), blocks(e, t - 1)]))\n",
     "            for k2 in (2, 1, 0):\n"
     "                rows_flat.append(int(e) * _SPE + fs * int(c.step_idx[rows[t - k2]]))\n", 1),
    ("        return (torch.stack(zh), torch.from_numpy(np.stack(ah)).to(dev), torch.stack(zg),\n"
     "                torch.from_numpy(np.stack(aref)).to(dev).float())\n",
     "        toks = encode_toks(rows_flat, B)\n"
     "        return (toks, torch.stack(zg),\n"
     "                torch.from_numpy(np.stack(aref)).to(dev).float())\n", 1),
    # actor_step(): token history + dino rollouts
    ("        zh, ah, zg, aref = sample(a.batch)\n",
     "        toks, zg, aref = sample(a.batch)\n", 1),
    ("        z0 = zh[:, -1]\n",
     "        z0 = toks[:, -1, :, :384].mean(dim=1)   # pooled pixel tokens (deploy-exact)\n", 1),
    ("                traj = rollout_traj(wm, zh, ah, A_in)\n",
     "                traj = rollout_traj_dino(wm, toks, A_in * ast5 + amu5)\n", 1),
    ("            tr = rollout_traj(wm, zh, ah, A)\n",
     "            tr = rollout_traj_dino(wm, toks, A * ast5 + amu5)\n", 1),
    # checkpoint helper + hook
    ("    # ------------------------------------------------------------ schedule\n",
     "    def _ckpt(tag):\n"
     "        # snapshot WITHOUT touching live modules; tagged paths, never a.out.\n"
     "        vpath = a.out_value.replace(\".pt\", f\"_s{tag}.pt\")\n"
     "        save_metric(copy.deepcopy(teacher).cpu(), \"td\", c_td.latent_dim, arch, vpath)\n"
     "        sd = {k2: v2.detach().cpu().clone() for k2, v2 in net.state_dict().items()}\n"
     "        torch.save({\"kind\": \"lip4_dino\", \"feed\": a.feed, \"sd\": sd,\n"
     "                    \"z_dim\": z.shape[-1], \"horizon\": a.horizon, \"iters\": a.iters,\n"
     "                    \"a_dim\": a_dim, \"amax\": a.amax, \"use_gate\": False,\n"
     "                    \"use_zg\": False, \"use_z0\": False, \"use_grad\": not a.drop_grad,\n"
     "                    \"value\": vpath,\n"
     "                    \"amu5\": torch.from_numpy(np.tile(amu, fs).astype(np.float32)),\n"
     "                    \"ast5\": torch.from_numpy(np.tile(astd, fs).astype(np.float32)),\n"
     "                    \"img_size\": a.img_size},\n"
     "                   a.out.replace(\".pt\", f\"_s{tag}.pt\"))\n"
     "        print(f\"[ckpt] s{tag} saved\", flush=True)\n"
     "\n"
     "    # ------------------------------------------------------------ schedule\n", 1),
    ('                  f"td_loss {cl:.4f} tau {tau_s:.3f} clr {clr_s:.2e} alr {alr_s:.2e}",\n'
     "                  flush=True)\n",
     '                  f"td_loss {cl:.4f} tau {tau_s:.3f} clr {clr_s:.2e} alr {alr_s:.2e}",\n'
     "                  flush=True)\n"
     "        if a.ckpt_every and (step + 1) % a.ckpt_every == 0:\n"
     "            _ckpt(step + 1)\n", 1),
    # final save: new kind + deploy fields
    ('        kind = "lip4"\n', '        kind = "lip4_dino"\n', 1),
    # anchored on the final save's unique net.cpu() prefix -- the _ckpt helper
    # inserted above contains the same z_dim line, so that alone matches twice
    ('"sd": net.cpu().state_dict(),\n'
     '                "z_dim": z.shape[-1], "horizon": a.horizon, "iters": a.iters,\n',
     '"sd": net.cpu().state_dict(),\n'
     '                "z_dim": z.shape[-1], "horizon": a.horizon, "iters": a.iters,\n'
     '                "amu5": torch.from_numpy(np.tile(amu, fs).astype(np.float32)),\n'
     '                "ast5": torch.from_numpy(np.tile(astd, fs).astype(np.float32)),\n'
     '                "img_size": a.img_size,\n', 1),
], 'trainer')

DST.write_text(t)
for f in (LIP, DST):
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(f)],
                       capture_output=True, text=True)
    print(f'{f.name}: compile {"OK" if r.returncode == 0 else "FAIL " + r.stderr[-300:]}')
