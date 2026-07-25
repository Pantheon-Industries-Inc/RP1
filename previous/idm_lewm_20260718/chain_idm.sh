#!/bin/bash
# Wait for the augment control run to finish, then launch the IDM-LeWM campaign.
# Completion signal = "AUGMENT DONE" in the LOG FILE (grepping a file, never a
# process list -> no pgrep self-match). Fallback: GPUs idle for two checks
# (covers the augment run dying without the marker). Max wait ~3h.
for i in $(seq 1 240); do
  grep -q "AUGMENT DONE" /workspace/logs/driver_augment.log 2>/dev/null && break
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1500{c++} END{print c+0}')
  if [ "$busy" = "0" ]; then
    sleep 30
    busy2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1500{c++} END{print c+0}')
    [ "$busy2" = "0" ] && break
  fi
  sleep 45
done
echo "[chain] launching IDM campaign $(date)" >> /workspace/logs/driver_idm.log
bash /workspace/run_idm_campaign.sh
