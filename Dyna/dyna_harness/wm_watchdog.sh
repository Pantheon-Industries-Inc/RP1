#!/bin/bash
# Watchdog for WM training on pod A (87.120.211.204). Exits (=notifies) on FIRST
# event: log error / workers gone / new checkpoint / ~12min stall. Re-arm after
# handling each event. Usage: wm_watchdog.sh <baseline_ckpt_mtime>
SSH="ssh -o ConnectTimeout=15 -o BatchMode=yes -p 17531 -i $HOME/.ssh/id_ed25519 root@87.120.211.204"
BASE_MT=${1:-0}
prev_step=""; stall=0; fails=0
while true; do
  out=$($SSH bash /workspace/watch_probe.sh 2>/dev/null)
  if [ -z "$out" ]; then
    fails=$((fails+1))
    if [ "$fails" -ge 10 ]; then echo "SSH_UNREACHABLE x10 (pod down?)"; exit 0; fi
    sleep 60; continue
  fi
  fails=0
  step="${out%%###*}"
  rest="${out#*###}";  err="${rest%%###*}"
  rest="${rest#*###}"; mt="${rest%%###*}"
  rest="${rest#*###}"; procs="${rest%%###*}"
  ckpt="${rest#*###}"
  if [ "$err" != "0" ]; then echo "TRAIN_ERROR_IN_LOG (count=$err); last step: $step"; exit 0; fi
  if [ "$procs" = "0" ]; then echo "TRAIN_PROCS_GONE; last step: $step"; exit 0; fi
  if [ "$mt" -gt "$BASE_MT" ] 2>/dev/null; then echo "NEW_CHECKPOINT: $ckpt (mtime $mt); last step: $step"; exit 0; fi
  if [ -n "$step" ] && [ "$step" = "$prev_step" ]; then stall=$((stall+1)); else stall=0; fi
  if [ "$stall" -ge 6 ]; then echo "STALL ~12min: $step (procs=$procs)"; exit 0; fi
  prev_step="$step"
  sleep 120
done
