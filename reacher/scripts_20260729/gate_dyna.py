"""Gate the Dyna driver on (a) a leak-free mix and (b) the amax sweep result.

(a) The r1 mix carries 400k rows (10.1%) from eval episodes 8000:10000; the
    overnight fine-tune used it and is discarded. Point MIX at the clean rebuild
    and give the run a new NAME so the leaky checkpoint can never be confused
    with it (the *_R1OLD lesson).
(b) User ordering: sweep amax first, then Dyna with the winning recipe.
"""

P = "/workspace/run_dyna2w3.sh"
s = open(P).read()
if "_CLEAN_MIX_READY" in s:
    print("dyna driver: already gated")
    raise SystemExit

s = s.replace(
    "MIX=/workspace/dyna_data/reacher_mix_r1_5050.lance",
    "MIX=/workspace/dyna_data/reacher_mix_r2_clean5050.lance",
)
s = s.replace("NAME=dyna_reacher_r2_5050", "NAME=dyna_reacher_r2_clean5050")

OLD_WAIT = '''log "queued: waiting for W3SWEEP_DONE"
while ! grep -q "W3SWEEP_DONE" /workspace/logs/w3sweep_driver.log 2>/dev/null; do sleep 120; done'''
NEW_WAIT = '''log "queued: waiting for the clean mix AND the amax sweep"
while [ ! -f /workspace/_CLEAN_MIX_READY ]; do sleep 60; done
while [ ! -f /workspace/_AMAX_SWEEP_DONE ]; do sleep 60; done
log "gates cleared: leak-free mix + amax winner available"
if [ -f /workspace/_AMAX_BEST ]; then
  read -r BEST_AMAX BEST_ITERS < /workspace/_AMAX_BEST
  log "recipe from sweep: amax ${BEST_AMAX} iters ${BEST_ITERS}"
else
  BEST_AMAX=2.2; BEST_ITERS=8
  log "no sweep winner file; falling back to canonical amax 2.2 iters 8"
fi'''
assert s.count(OLD_WAIT) == 1, f"wait anchor x{s.count(OLD_WAIT)}"
s = s.replace(OLD_WAIT, NEW_WAIT)

# use the swept recipe for the Dyna LIP actors
s = s.replace("--arch v4 --amax 2.2 --max-delta 12 --iters 8 --horizon 5",
              '--arch v4 --amax "$BEST_AMAX" --max-delta 12 --iters "$BEST_ITERS" --horizon 5')
open(P, "w").write(s)
print("dyna driver gated: clean MIX, new NAME, swept recipe")
