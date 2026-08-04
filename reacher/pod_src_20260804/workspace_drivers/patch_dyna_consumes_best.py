"""Chain Dyna behind the joint sweep so it runs on the SWEPT winner.

run_dyna_full.sh hardcoded the old 48.9 recipe (--amax 1.8 --actor-lr 1e-3,
expectile 0.1, replay/expand 0). The user asked for Dyna to be queued behind the
sweep and to run automatically on the best configuration, so it has to read the
winner instead of assuming it.

The sweep writes, per base:
    /workspace/_BEST_<base>   ->  "<base> <amax> <lr> <EX> <replay> <expand>"
where EX is the tag form (005 | 01), mapped back to 0.05 | 0.1 here.

Three edits:

1. A gate at the top: block until _BEST_lejepa exists (i.e. the sweep reached
   stage 3), polling on-pod every 2 min with a hard ceiling so a dead sweep
   cannot leave this parked forever.

2. The window value for the fine-tuned WM is retrained at the SWEPT expectile,
   not a hardcoded 0.1 -- expectile is a hyper of the value itself, so reusing a
   0.1 value under a 0.05 winner would silently evaluate the wrong recipe. The
   output path carries the expectile so the two cannot collide on disk.

3. The LIP training step uses the swept amax / lr / replay / expand.

Deliberately NOT changed: the fine-tune itself (lr 1e-5, 2 epochs) and the split
discipline (expert 0:1500, on-policy collected 0:8000, eval 8000:10000). Those
are properties of the Dyna experiment, not of the LIP recipe under test.
"""

import re
import subprocess

P = "/workspace/run_dyna_full.sh"
s = open(P).read()

if "_BEST_lejepa" in s:
    print("already patched")
    raise SystemExit

# ---------------------------------------------------------------- 1. gate + read winner
OLD1 = '''# ---------------------------------------------------------------- action stats pin'''

NEW1 = '''# ---------------------------------------------------------------- 0. wait for the sweep
# Dyna must run on the configuration the joint sweep actually selected, so it is
# gated on the sweep reaching stage 3. Ceiling 6 h: a dead sweep should fail
# loudly here rather than silently park the GPUs.
BESTF=/workspace/_BEST_lejepa
if [ ! -f "$BESTF" ]; then
  log "waiting for the joint sweep to write ${BESTF} (ceiling 6 h)"
  waited=0
  while [ ! -f "$BESTF" ]; do
    sleep 120; waited=$((waited + 120))
    [ $((waited % 1800)) -eq 0 ] && log "  still waiting, ${waited}s elapsed"
    [ "$waited" -ge 21600 ] && die "sweep never produced ${BESTF}"
  done
fi
read -r _b SW_AMAX SW_LR SW_EX SW_RP SW_EW < "$BESTF"
case "$SW_EX" in 005) SW_EXV=0.05;; 01) SW_EXV=0.1;; *) die "bad expectile tag ${SW_EX}";; esac
SW_LRF=1e-4; [ "$SW_LR" = "3e-4" ] && SW_LRF=3e-5
log "swept winner for lejepa: amax ${SW_AMAX} lr ${SW_LR} expectile ${SW_EXV} replay ${SW_RP} expand ${SW_EW}"

# ---------------------------------------------------------------- action stats pin'''

assert s.count(OLD1) == 1, f"gate anchor x{s.count(OLD1)}"
s = s.replace(OLD1, NEW1, 1)

# ---------------------------------------------------------------- 2. window value at swept expectile
OLD2 = '''W3=/workspace/metrics/window3_dyna_lejepa.pt'''
NEW2 = '''W3=/workspace/metrics/window3_dyna_lejepa_e${SW_EX}.pt'''
assert s.count(OLD2) == 1, f"W3 anchor x{s.count(OLD2)}"
s = s.replace(OLD2, NEW2)

OLD3 = '''  /workspace/train_window.py --cache "$C1" --lag 5 --frames 3 --expectile 0.1 \\
  --n-step 50 --steps 6000 --seed 0 --out "$W3" > "$L/w3.log" 2>&1 || die "window value failed"; }'''
NEW3 = '''  /workspace/train_window.py --cache "$C1" --lag 5 --frames 3 --expectile "$SW_EXV" \\
  --n-step 50 --steps 6000 --seed 0 --out "$W3" > "$L/w3.log" 2>&1 || die "window value failed"; }'''
assert s.count(OLD3) == 1, f"train_window anchor x{s.count(OLD3)}"
s = s.replace(OLD3, NEW3)

# ---------------------------------------------------------------- 3. LIP at the swept recipe
OLD4 = '''log "LIP x3 seeds: 1000 steps, lr 1e-3, uniform, amax 1.8, mw 0.1 (the 48.9 recipe)"'''
NEW4 = '''log "LIP x3 seeds at the SWEPT recipe: 1000 steps, amax ${SW_AMAX}, lr ${SW_LR}, expectile ${SW_EXV}, replay ${SW_RP}, expand ${SW_EW}"'''
assert s.count(OLD4) == 1, f"log anchor x{s.count(OLD4)}"
s = s.replace(OLD4, NEW4)

OLD5 = '''    --arch v4 --amax 1.8 --iters 8 --horizon 5 --max-delta 12 \\
    --steps 1000 --batch 128 --n-step 50 \\
    --expectile 0.1 --expectile-final 0.03 \\
    --critic-lr 1e-3 --critic-lr-final 1e-4 \\
    --actor-lr 1e-3 --actor-lr-final 1e-4 \\
    --lambda-schedule uniform --mean-weight 0.1 --seed "$s" \\'''
NEW5 = '''    --arch v4 --amax "$SW_AMAX" --iters 8 --horizon 5 --max-delta 12 \\
    --steps 1000 --batch 128 --n-step 50 \\
    --expectile 0.1 --expectile-final 0.03 \\
    --critic-lr 1e-3 --critic-lr-final 1e-4 \\
    --actor-lr "$SW_LR" --actor-lr-final "$SW_LRF" \\
    --replay-prob "$SW_RP" --expand-weight "$SW_EW" \\
    --lambda-schedule uniform --mean-weight 0.1 --seed "$s" \\'''
assert s.count(OLD5) == 1, f"LIP recipe anchor x{s.count(OLD5)}"
s = s.replace(OLD5, NEW5)

# actor paths must not collide with the pre-sweep dyna actors
s = s.replace("/workspace/actors/lip4_dyna_s${s}.pt",
              "/workspace/actors/lip4_dyna_sw_s${s}.pt")
s = s.replace("/workspace/metrics/lip4_dyna_s${s}_value.pt",
              "/workspace/metrics/lip4_dyna_sw_s${s}_value.pt")
s = s.replace("/workspace/actors/lip4_dyna_s${sd}.pt",
              "/workspace/actors/lip4_dyna_sw_s${sd}.pt")

# the baseline bar in the final line is now whatever the sweep carded
s = s.replace('log "DYNA_DONE"',
              'log "DYNA_DONE -- compare against the sweep\'s POOLED lejepa card, not the old 48.9"')

open(P, "w").write(s)
r = subprocess.run(["bash", "-n", P], capture_output=True, text=True)
assert r.returncode == 0, f"syntax error:\n{r.stderr}"
print(f"patched: gated on _BEST_lejepa, window value at swept expectile, "
      f"LIP at swept amax/lr/replay/expand ({len(re.findall(r'SW_', s))} refs); bash -n clean")
