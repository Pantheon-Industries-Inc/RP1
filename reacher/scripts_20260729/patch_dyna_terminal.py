"""Re-point the Dyna driver at the 1-FRAME TERMINAL critic (user directive).

The driver was written when the 3-frame window was the leading cost. Held-at-end
then showed the window was a latched-only win -- for the learned quasimetric it
went from held 39.3 (terminal) to 11.7 (window) -- so Dyna must be evaluated with
the terminal critic instead, or its result would not be comparable to the current
baseline (LIP terminal pad: 57.0 best / 48.6 pooled, n=6).

Swaps train_window.py -> train_metric.py for the post-fine-tune value, and drops
the window-L2 arm (which is a 3-frame object) for a terminal Latent+CEM arm.
Also releases the amax gate: the 3-seed Pre card settled amax at 2.2 (held 44.2
vs 1.8's 33.4), so the sweep marker is written directly rather than waited on.
"""

P = "/workspace/run_dyna2w3.sh"
s = open(P).read()
if "train_metric.py" in s and "TERMINAL critic" in s:
    print("already re-pointed")
    raise SystemExit

# 1. terminal TD instead of a 3-frame window value
OLD_W3 = '''W3=/workspace/metrics/window3_dyna2_lejepa.pt'''
NEW_W3 = '''W3=/workspace/metrics/td_dyna2_lejepa.pt   # TERMINAL critic (1-frame quasimetric)'''
assert s.count(OLD_W3) == 1, f"W3 anchor x{s.count(OLD_W3)}"
s = s.replace(OLD_W3, NEW_W3)

OLD_TRAIN = '''  log "window3 value on FT latents"
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 /workspace/train_window.py \\
    --cache "$C1" --lag 5 --frames 3 --expectile 0.1 --n-step 50 --steps 6000 --seed 0 \\
    --out "$W3" > "$L/w3.log" 2>&1 || die "window value failed"'''
NEW_TRAIN = '''  log "terminal TD quasimetric on FT latents (canonical tau 0.1 / n 50)"
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$PLAN/train_metric.py" \\
    --cache "$C1" --learner td --head quasimetric \\
    --expectile 0.1 --n-step 50 --steps 6000 --seed 0 \\
    --out "$W3" > "$L/td.log" 2>&1 || die "terminal TD failed"'''
assert s.count(OLD_TRAIN) == 1, f"train anchor x{s.count(OLD_TRAIN)}"
s = s.replace(OLD_TRAIN, NEW_TRAIN)

# 2. terminal Latent+CEM arm instead of the 3-frame window-L2 arm
s = s.replace(
    'ev 1 "dyw3l2_s${seed}" $seed solver=cem solver.n_steps=10 "+metric=/workspace/metrics/l2window3.pt"',
    'ev 1 "dyw3l2_s${seed}" $seed solver=cem solver.n_steps=10')
s = s.replace('card6 "dyw3l2_s" "DYNA Latent+CEM window [pre-Dyna 89.7]"',
              'card6 "dyw3l2_s" "DYNA Latent+CEM terminal [pre-Dyna held 41.7]"')
s = s.replace('card6 "dyw3td_s" "DYNA TD+CEM window [pre-Dyna 86.7]"',
              'card6 "dyw3td_s" "DYNA TD+CEM terminal [pre-Dyna held 35.7]"')
s = s.replace('"DYNA LIP window s${s} plain [pre-Dyna pooled 90.6]"',
              '"DYNA LIP terminal s${s} plain [pre-Dyna held: best 57.0 / pooled 48.6]"')

# 3. record HELD as the primary metric
s = s.replace('echo "${nm},latched=${ever:-FAIL},held=${held:-FAIL}" >> "$SUM"',
              'echo "${nm},held=${held:-FAIL},latched=${ever:-FAIL}" >> "$SUM"')
s = s.replace('''    e=$(grep "^${pre}${seed}," "$SUM" | tail -1 | grep -oE "latched=[0-9.]+" | grep -oE "[0-9.]+")''',
              '''    e=$(grep "^${pre}${seed}," "$SUM" | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")''')
s = s.replace('log "CARD6 ${tag}: latched h25', 'log "CARD6 ${tag}: HELD h25')
open(P, "w").write(s)
print("run_dyna2w3.sh re-pointed: terminal TD, held-primary, terminal CEM arms")
