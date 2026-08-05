"""Let Dyna pick up the value-steps ladder result late, so it can start now.

Dyna's long pole is the WM fine-tune (hours). Its window value is only rebuilt
AFTER that fine-tune, and the value-steps ladder finishes well inside that
window -- so there is no reason to keep Dyna parked until the ladder lands.

The step count is therefore read at the point of use, not at the top of the
script: /workspace/_VSTEPS if it exists, else the current 6000. The ladder
writes _VSTEPS when it picks a winner; if it has not written by the time Dyna
gets there, Dyna proceeds on the established 6000 rather than blocking.

The step count also goes into the value's filename, so a 6000-step value and a
60000-step value cannot silently collide on disk and be reused for each other.
"""

import subprocess

P = "/workspace/run_dyna_full.sh"
s = open(P).read()

if "SW_VSTEPS" in s:
    print("already patched")
    raise SystemExit

OLD_W3 = 'W3=/workspace/metrics/window3_dyna_lejepa_e${SW_EX}.pt'
NEW_W3 = '''# read at point of use, not at the gate: the value-steps ladder is still running
# when Dyna starts, and Dyna does not need this until after the fine-tune
SW_VSTEPS=6000
[ -f /workspace/_VSTEPS ] && SW_VSTEPS=$(cat /workspace/_VSTEPS)
W3=/workspace/metrics/window3_dyna_lejepa_e${SW_EX}_st${SW_VSTEPS}.pt'''
assert s.count(OLD_W3) == 1, f"W3 anchor x{s.count(OLD_W3)}"
s = s.replace(OLD_W3, NEW_W3)

OLD_T = '--n-step 50 --steps 6000 --seed 0 --out "$W3"'
NEW_T = '--n-step 50 --steps "$SW_VSTEPS" --seed 0 --out "$W3"'
assert s.count(OLD_T) == 1, f"steps anchor x{s.count(OLD_T)}"
s = s.replace(OLD_T, NEW_T)

OLD_L = 'log "window3 value on fine-tuned latents"'
NEW_L = 'log "window3 value on fine-tuned latents (${SW_VSTEPS} steps)"'
assert s.count(OLD_L) == 1, f"log anchor x{s.count(OLD_L)}"
s = s.replace(OLD_L, NEW_L)

open(P, "w").write(s)
r = subprocess.run(["bash", "-n", P], capture_output=True, text=True)
assert r.returncode == 0, f"syntax error:\n{r.stderr}"
print("run_dyna_full.sh: value steps read from _VSTEPS at point of use "
      "(default 6000), step count in the filename; bash -n clean")
