#!/usr/bin/env bash
# Ship + start the six PushT MODE=report jobs via the pod runner (fresh job dirs; the copied-teacher-dir approach ran the
# teacher's hardcoded launch.sh). usage: pusht_report_ship.sh <T> <STEP>   (env POD, TAG as in pusht_lock.sh)
set -uo pipefail
T=$1; STEP=$2; POD=${POD:-root@69.30.85.162:22186}; TAG=${TAG:-rlp-pt-uniJ-h25-20260923}; H=${POD%%:*}; P=${POD##*:}
for s in 0 1 2 3 4 5; do
  # env of the shipped teacher job (all UNI_* etc.), minus the keys we override
  ENVARGS=$(ssh -o BatchMode=yes -o ConnectTimeout=25 -i ~/.ssh/id_ed25519 -p $P $H "grep -E '^export [A-Z_0-9]+=' /root/podjobs/rlp-pt-uniJ-h25-s${s}-teacher/env.sh" \
    | sed -E "s/^export //; s/^([A-Z_0-9]+)='(.*)'$/\1=\2/" | grep -vE "^(MODE|GPU|FIXED_ROW|FIXED_STEP|ROW)=" | sed 's/^/--env /' | tr '\n' ' ')
  eval python3 scripts/sky/pod/run_yaml_on_pod.py --host $H --port $P --key ~/.ssh/id_ed25519 --yaml scripts/sky/pusht_uniJ.yaml \
    --name rlp-pt-uniJ-h25-s${s}-report --home /root/hpusht $ENVARGS --env MODE=report --env GPU=$s --env FIXED_ROW=t$T --env FIXED_STEP=$STEP 2>&1 | grep -E "pod-runner|rror"
done
