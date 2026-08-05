"""Re-point Dyna at the WINDOW critic (user: 'stick with the window held
version') and fix the checkpoint-ambiguity crash IN THE DRIVER.

Two changes to /workspace/run_dyna2w3.sh:

1. The post-fine-tune value goes back to the 3-frame window quasimetric
   (train_window.py), and the CEM comparison arm back to window-L2. Held stays
   the primary metric (that part of the terminal patch is kept). References
   updated to the window-held baselines: LIP-w pooled 44.2, Latent+CEM-w 42.7,
   TD+CEM-w 11.7.

2. The trainer writes weights_epoch_{1,2}.pt and no weights.pt; the loader
   refuses dirs with multiple .pt files. Last time this was fixed BY HAND and
   therefore recurred on the clean re-run. Now the driver normalises the dir
   right after fine-tuning: newest epoch file becomes weights.pt, the epoch
   files move to _extra_ckpts/.
"""

P = "/workspace/run_dyna2w3.sh"
s = open(P).read()

# ---- 1a. window value instead of terminal TD
OLD_W3 = "W3=/workspace/metrics/td_dyna2_lejepa.pt   # TERMINAL critic (1-frame quasimetric)"
NEW_W3 = "W3=/workspace/metrics/window3_dyna2_lejepa.pt   # WINDOW critic (3-frame, held setting)"
if OLD_W3 in s:
    s = s.replace(OLD_W3, NEW_W3)
OLD_TRAIN = '''  log "terminal TD quasimetric on FT latents (canonical tau 0.1 / n 50)"
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$PLAN/train_metric.py" \\
    --cache "$C1" --learner td --head quasimetric \\
    --expectile 0.1 --n-step 50 --steps 6000 --seed 0 \\
    --out "$W3" > "$L/td.log" 2>&1 || die "terminal TD failed"'''
NEW_TRAIN = '''  log "window3 value on FT latents (the held-setting critic)"
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 /workspace/train_window.py \\
    --cache "$C1" --lag 5 --frames 3 --expectile 0.1 --n-step 50 --steps 6000 --seed 0 \\
    --out "$W3" > "$L/w3.log" 2>&1 || die "window value failed"'''
if OLD_TRAIN in s:
    s = s.replace(OLD_TRAIN, NEW_TRAIN)

# ---- 1b. window-L2 CEM arm + window-held references
s = s.replace('ev 1 "dyw3l2_s${seed}" $seed solver=cem solver.n_steps=10',
              'ev 1 "dyw3l2_s${seed}" $seed solver=cem solver.n_steps=10 "+metric=/workspace/metrics/l2window3.pt"')
s = s.replace('card6 "dyw3l2_s" "DYNA Latent+CEM terminal [pre-Dyna held 41.7]"',
              'card6 "dyw3l2_s" "DYNA Latent+CEM window [pre-Dyna held 42.7]"')
s = s.replace('card6 "dyw3td_s" "DYNA TD+CEM terminal [pre-Dyna held 35.7]"',
              'card6 "dyw3td_s" "DYNA TD+CEM window [pre-Dyna held 11.7]"')
s = s.replace('"DYNA LIP terminal s${s} plain [pre-Dyna held: best 57.0 / pooled 48.6]"',
              '"DYNA LIP window s${s} plain [pre-Dyna held pooled 44.2 | Latent+CEM-w 42.7]"')

# ---- 2. checkpoint normalisation right after the fine-tune block
ANCHOR = '''[ -n "$FT" ] && [ -e "$FT" ] || die "no fine-tuned checkpoint found"'''
NORM = '''# normalise the trainer's output: weights_epoch_N.pt -> weights.pt (newest
# epoch wins), extras shelved -- the loader refuses dirs with multiple .pt files
if [ -n "$FT" ] && [ ! -f "$FT/weights.pt" ]; then
  latest=$(ls -t "$FT"/weights_epoch_*.pt 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    mkdir -p "$FT/_extra_ckpts"
    for f in "$FT"/weights_epoch_*.pt; do mv "$f" "$FT/_extra_ckpts/"; done
    cp "$FT/_extra_ckpts/$(basename $latest)" "$FT/weights.pt"
    log "normalised checkpoint: $(basename $latest) -> weights.pt"
  fi
fi
[ -n "$FT" ] && [ -e "$FT" ] || die "no fine-tuned checkpoint found"'''
assert s.count(ANCHOR) == 1, f"norm anchor x{s.count(ANCHOR)}"
s = s.replace(ANCHOR, NORM)

open(P, "w").write(s)
checks = ["window3_dyna2_lejepa", "train_window.py", "l2window3.pt",
          "normalised checkpoint", "pre-Dyna held 42.7"]
print("dyna driver re-pointed to window+held; checks:",
      {c: (c in s) for c in checks})
