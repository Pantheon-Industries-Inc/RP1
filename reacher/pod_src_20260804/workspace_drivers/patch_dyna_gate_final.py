"""Repoint Dyna's gate, and fix the held10 regex.

Two unrelated defects, both cheap to fix while Dyna is idle at its gate.

1. GATE. The objective changed: the user wants BOTH bases above their own
   Latent+CEM bar (lejepa 44.7, pldm 39.3), not the best cross-base mean. The
   sweep's stage 3 still writes _BEST_<base> by cross-base mean, and Dyna is
   watching that file -- so the moment the sweep finishes, Dyna would start on
   the wrong winner.

   The running sweep is NOT edited to fix this. bash reads a script
   incrementally, so rewriting a file mid-execution shifts byte offsets under
   the interpreter; the sweep has ~60 trainings still to go and its trainings
   are valid under either criterion (the criterion only affects which cell is
   chosen at the end). So the selection is redone afterwards from
   joint_screen.txt, and Dyna waits on a file only this side writes.

   _BEST_lejepa        -> written by the sweep, cross-base mean, now ignored
   _BEST_FINAL_lejepa  -> written by the min-margin rescore, what Dyna consumes

2. HELD10. meanof() does
       grep -oE "held10=[0-9.]+" | grep -oE "[0-9.]+"
   and the second grep matches the "10" inside the KEY before it ever reaches
   the value, so every @0.1 mean printed a constant 10.0. Seen in the PLDM bars
   run: the six per-seed rows are 80/82/80/82/80/66 (mean 78.3) but the summary
   line claimed 10.0. Switched to cut -d= -f2, which takes the value.
"""

import subprocess

# ---------------------------------------------------------------- gate
P = "/workspace/run_dyna_full.sh"
s = open(P).read()

if "_BEST_FINAL_lejepa" in s:
    print("gate: already repointed")
else:
    OLD = 'BESTF=/workspace/_BEST_lejepa'
    NEW = ('# written by the min-margin rescore, NOT by the sweep\'s cross-base-mean\n'
           '# stage 3 -- the objective is "both bases above their own Latent+CEM bar"\n'
           'BESTF=/workspace/_BEST_FINAL_lejepa')
    assert s.count(OLD) == 1, f"gate anchor x{s.count(OLD)}"
    s = s.replace(OLD, NEW)
    s = s.replace('log "waiting for the joint sweep to write ${BESTF} (ceiling 6 h)"',
                  'log "waiting for the min-margin winner at ${BESTF} (ceiling 6 h)"')
    print("gate: now waits on _BEST_FINAL_lejepa")

# ---------------------------------------------------------------- held10 regex
OLD_M = '''    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | grep -oE "[0-9.]+")'''
NEW_M = '''    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "${key}=[0-9.]+" | cut -d= -f2)'''
if OLD_M in s:
    s = s.replace(OLD_M, NEW_M)
    print("held10: regex fixed (was returning the 10 from the key name)")
else:
    print("held10: anchor not present, skipped")

open(P, "w").write(s)
r = subprocess.run(["bash", "-n", P], capture_output=True, text=True)
assert r.returncode == 0, f"syntax error:\n{r.stderr}"
print("run_dyna_full.sh: bash -n clean")

# ---------------------------------------------------------------- same fix, bars script
B = "/workspace/run_pldm_bars.sh"
try:
    b = open(B).read()
    if OLD_M in b:
        open(B, "w").write(b.replace(OLD_M, NEW_M))
        r = subprocess.run(["bash", "-n", B], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        print("run_pldm_bars.sh: held10 fixed, bash -n clean")
except FileNotFoundError:
    print("run_pldm_bars.sh: absent, skipped")
