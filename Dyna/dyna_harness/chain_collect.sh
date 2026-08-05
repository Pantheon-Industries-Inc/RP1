#!/bin/bash
# FAILURE-STATE CHAINING collection: build the chain task dataset from the 289
# labeled failures (start = mid-failure state, goal = the original task goal),
# then roll the PRE actors from those starts. 867 chains = 17 slices x 51; each
# call's ep_range slice is covered deterministically (draw = permutation of the
# slice's first 50), so no cross-call duplicates. See build_chain_dataset.py.
#
# PREREQUISITE: the pod's world.py must have the recorder outcome column
# (SWM_RECORD_OUTCOME) -- the relabel fallback does NOT apply to chain draws.
# Preflight dies if it is missing.
#
# ~2 min build + 17 x ~2.3 min collection ~= 45 min, serial on GPU 0 (egl).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; D=/workspace/dyna_split; DRV=$L/driver_chain.log
V2WM=/workspace/models/v2WM
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
CHAIN=$D/chain_tasks.lance
EPHI=8000; SLICE=51
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
cd "$CODE"

# ------------------------------------------------------------------ preflight
log "=== FAILURE-STATE CHAINING (pid $$) ==="
grep -q "SWM_RECORD_OUTCOME" stable_worldmodel/world/world.py \
  || die "pod world.py lacks the recorder outcome column -- sync the repo world.py first"
for a in 0 1 2; do
  [ -d "$D/onpolicy_full_a${a}_lab.lance" ] || die "labeled lance a$a missing"
  [ -f "/workspace/actors/lip4_dsp_pre_s${a}.pt" ] || die "PRE actor s$a missing"
done
[ -d "$EXPERT" ] || die "expert lance missing"
[ -f /workspace/build_chain_dataset.py ] || die "builder not deployed"
[ -f /workspace/relabel_onpolicy.py ] || die "relabel module not deployed (builder imports it)"
if [ "${SKIP_PGREP:-0}" != 1 ]; then
  pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null && die "preflight: something is already running"
fi
log "P0: preflight OK"
[ "${1:-}" = check ] && { log "check mode: exiting"; exit 0; }

# ------------------------------------------------------------- P1 build chains
if [ ! -f "$D/CHAIN_BUILD_DONE" ]; then
  log "P1: building chain task dataset (mids 15/25/35)"
  python3 /workspace/build_chain_dataset.py --expert "$EXPERT" \
    --labeled $D/onpolicy_full_a0_lab.lance $D/onpolicy_full_a1_lab.lance $D/onpolicy_full_a2_lab.lance \
    --out "$CHAIN" --mids 15 25 35 --ep-range "0:${EPHI}" \
    > "$L/chain_build.log" 2>&1 || die "chain build failed -- see $L/chain_build.log"
  grep -q BUILD_CHAIN_DONE "$L/chain_build.log" || die "chain build incomplete"
  touch "$D/CHAIN_BUILD_DONE"
fi
PLAN=$(grep -h "SLICE_PLAN" "$L/chain_build.log" | tail -1)
log "P1: $PLAN"
NSLICE=$(echo "$PLAN" | grep -oE "n_slices=[0-9]+" | cut -d= -f2)
[ -n "$NSLICE" ] && [ "$NSLICE" -ge 1 ] || die "could not parse slice plan"

# --------------------------------------------------------------- P2 collection
# Slices round-robin across the 3 PRE actors: mostly cross-actor chaining --
# a different actor attempts the stuck state. Legitimate for the WM (the
# critic never sees this data) and adds recovery-mode diversity.
log "P2: $NSLICE collection calls, terminate_at_goal=False, stats pinned to expert"
for i in $(seq 0 $((NSLICE-1))); do
  a=$((i % 3)); lo=$((i*SLICE)); hi=$(((i+1)*SLICE))
  lg="$L/chaincol_i${i}.log"
  grep -q "kept=" "$lg" 2>/dev/null && { log "  call $i cached"; continue; }
  SWM_RECORD_PATH=$D/chained_a${a}.lance SWM_RECORD_OUTCOME=1 \
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" \
    --config-name cube seed=$((3000 + i)) eval.dataset_name="$CHAIN" \
    dataset.stats="$EXPERT" \
    ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
    eval.num_eval=50 "+eval.ep_range=${lo}:${hi}" world.terminate_at_goal=False \
    policy="$V2WM" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_dsp_pre_s${a}.pt" \
    output.filename="chaincol_i${i}.txt" > "$lg" 2>&1
  grep -q "ep_range ${lo}:${hi}" "$lg" || die "call $i: ep_range not applied"
  rec=$(grep -h "\[record\]" "$lg" | tail -1)
  echo "$rec" | grep -q "kept=50 dropped=0" || die "call $i: unexpected keep/drop -- $rec"
  echo "$rec" | grep -q "kept_success=" || die "call $i: outcome label MISSING -- $rec"
  log "  call $i (actor s$a, slice ${lo}:${hi}): $(echo "$rec" | grep -oE 'kept_success=[0-9]+')"
done
log "P2: collection done"

# ------------------------------------------------------------------- P3 report
python3 - "$D" <<'PY' 2>&1 | tee -a "$DRV"
import sys, glob, numpy as np, lance
D = sys.argv[1]
tot = {"rows": 0, "eps": 0, "fail_eps": 0, "fail_rows": 0}
for p in sorted(glob.glob(f"{D}/chained_a*.lance")):
    ds = lance.dataset(p)
    t = ds.to_table(columns=["episode_idx", "success"])
    ep = t.column(0).to_numpy().reshape(-1)
    ok = t.column(1).to_numpy().reshape(-1).astype(bool)
    eps, inv = np.unique(ep, return_inverse=True)
    ever = np.zeros(len(eps), bool); np.logical_or.at(ever, inv, ok)
    rows = np.bincount(inv, minlength=len(eps))
    tot["rows"] += len(ep); tot["eps"] += len(eps)
    tot["fail_eps"] += int((~ever).sum()); tot["fail_rows"] += int(rows[~ever].sum())
    print(f"  {p.split('/')[-1]}: eps={len(eps)} succ={int(ever.sum())} "
          f"fail={int((~ever).sum())}")
print(f"CHAINED TOTAL: eps={tot['eps']} rows={tot['rows']} "
      f"fail_eps={tot['fail_eps']} fail_rows={tot['fail_rows']} "
      f"(chain success rate {100*(1-tot['fail_eps']/max(tot['eps'],1)):.1f}%)")
# K table for the combined pool (original full arm + chained)
E, F0, S0 = 1_608_000, 14_450, 75_550
F = F0 + tot["fail_rows"]; S = S0 + (tot["rows"] - tot["fail_rows"])
T = E / 0.5
for phi in (0.08, 0.13, 0.20):
    kf = phi * T / F; ks = (0.5 - phi) * T / S
    print(f"  phi={phi:.2f}: K_fail={kf:.1f} K_succ={ks:.1f}"
          + ("   <-- now clean" if kf <= 30 and ks <= 30 else ""))
print("CHAIN_COLLECT_DONE")
PY
log "done"
