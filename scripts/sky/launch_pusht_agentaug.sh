#!/usr/bin/env bash
# PushT critic fix (2026-09-08): counterfactual agent augmentation on the
# config-B actors (pusht-v2l05-s*-20260902 recipe, unchanged) — see
# docs/campaigns/2026-09-03/PUSHT_DIAG.md (E9 diagnosis, E10 fix).
#
#   scripts/sky/launch_pusht_agentaug.sh A 0        # arm A, seed 0: builds the aug cache
#   scripts/sky/launch_pusht_agentaug.sh A 1 2      # seeds 1/2 reuse the seed-0 cache
#   scripts/sky/launch_pusht_agentaug.sh B 0        # arm B (label-preserving, transit 0)
#   DRY=1 scripts/sky/launch_pusht_agentaug.sh A 0  # print only
#
# Arms:  A  AGENT_AUG=0.5 AUG_TRANSIT=1.0   (displaced query, label d + walk-back)
#        B  AGENT_AUG=0.5 AUG_TRANSIT=0     (displaced query, label d)
#        C  AGENT_AUG=0.25 AUG_TRANSIT=1.0
#        D  AGENT_AUG=1.0 AUG_TRANSIT=1.0   (every critic query displaced)
# Eval conditions: rlp (the deployed planner), cem_value / cem_tdvalue (the
# critics as CEM objectives, E4 pairing), plus the E9 agent-vs-block probe on
# the new critics as a POST script. Report draws 42/43/44, one K=8 pass.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.sky/bin:$PATH"

ARM=${1:?arm A|B|C|D}; shift
SEEDS=${*:-0}
DATE=${DATE:-20260908}
CACHE_OWNER=${CACHE_OWNER:-pusht-augA-s0-$DATE}   # tag whose caches/ holds the aug cache
case $ARM in
  A) P=0.5;  TR=1.0;;
  B) P=0.5;  TR=0;;
  C) P=0.25; TR=1.0;;
  D) P=1.0;  TR=1.0;;
  *) echo "unknown arm $ARM" >&2; exit 1;;
esac

POST_B64=$(python3 - <<'PY'
import base64, pathlib
head = (
    'import os\n'
    'D = os.environ["D"]\n'
    'os.environ.update(DIAG=D, ACTOR=D, WM=D + "/lewm_pusht_official", OUT=D + "/e9", GRID=os.environ.get("GRID", "24"), NT=os.environ.get("NT", "15"))\n'
)
body = pathlib.Path("scripts/pusht_diag/block_sensitivity.py").read_text().replace("from __future__ import annotations\n", "")
tail = (
    '\nimport base64, pathlib\n'
    'for name in ("block_sensitivity.json", "e9_agent_vs_block.png"):\n'
    '    data = base64.b64encode(pathlib.Path(os.environ["OUT"], name).read_bytes()).decode()\n'
    '    print(f"[e9-b64] {name} {data}", flush=True)\n'
)
print(base64.b64encode((head + body + tail).encode()).decode())
PY
)

for S in $SEEDS; do
  TAG="pusht-aug${ARM}-s${S}-${DATE}"
  if [ "$TAG" = "$CACHE_OWNER" ]; then AUGTAG=""; WAIT=0; else AUGTAG=$CACHE_OWNER; WAIT=${WAIT_CACHE_MIN:-240}; fi
  cmd=(sky jobs launch scripts/sky/counterstrike_pusht.yaml
    -n "rlp-$TAG" --priority p1 -y --async
    --env EXPERIMENT_TAG="$TAG"
    --env CACHE_TAG=counterstrike --env WAIT_CACHE_MIN="$WAIT"
    --env AGENT_AUG="$P" --env AUG_TRANSIT="$TR" --env AUG_CACHE_TAG="$AUGTAG"
    --env SEED="$S" --env TD_MODE=cube --env ITERS=8
    --env VALUE_GAMMA=0.98 --env VALUE_NSTEP=1 --env VALUE_EXPECTILE=0.03
    --env MAX_DELTA=20 --env MEAN_WEIGHT=0.1 --env AMAX=2.5
    --env CKPT_SELECT=1 --env CKPT_VAL_SEEDS="50 51"
    --env TRAIN_OVERRIDES="value.window_frames=4 value.window_lag=5 planner.ac_weight=0.5 planner.ckpt_every=2000"
    --env EVAL_SEEDS="42 43 44" --env EVAL_CONDS="rlp cem_value cem_tdvalue" --env SMOKE=0
    --env POST_SCRIPT_B64="$POST_B64"
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer
    --env PANTHEON_USER=armin@pantheon.inc)
  if [ "${DRY:-0}" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}" 2>&1 | tail -2; fi
  echo "-> $TAG (aug p=$P transit=$TR cache=${AUGTAG:-own})"
done
