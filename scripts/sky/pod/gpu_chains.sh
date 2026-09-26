#!/usr/bin/env bash
# Run pod jobs (shipped with run_yaml_on_pod.py --no-start) as per-GPU sequential chains, optionally after a
# gate job finishes.  usage: gpu_chains.sh <host:port> <gate-job-or-> <name:gpu> [<name:gpu> ...]
#   e.g. gpu_chains.sh root@1.2.3.4:22186 rlp-pt-uniJ-h25-prepare rlp-pt-uniJ-h25-s0-teacher:0 rlp-pt-uniJ-h25-s0-t3000:1 ...
# Each chain waits for the gate job's exit_code (must be 0), then runs its jobs in the listed order.
set -euo pipefail
HP=$1; GATE=$2; shift 2
declare -A CH
for spec in "$@"; do n=${spec%%:*}; g=${spec##*:}; CH[$g]="${CH[$g]:-} $n"; done
SCRIPT=""
for g in "${!CH[@]}"; do
  SCRIPT+="( "
  # optional: WAIT_PIDFILE=<file> -> also wait until the process in that pid file (e.g. the TwoRoom chain) has exited
  [ -n "${WAIT_PIDFILE:-}" ] && SCRIPT+="while [ -f $WAIT_PIDFILE ] && kill -0 \$(cat $WAIT_PIDFILE) 2>/dev/null; do sleep 60; done; echo \"[gpu$g] $WAIT_PIDFILE released \$(date -u +%FT%TZ)\"; "
  [ "$GATE" != - ] && SCRIPT+="while [ ! -f /root/podjobs/$GATE/exit_code ]; do sleep 60; done; [ \"\$(cat /root/podjobs/$GATE/exit_code)\" = 0 ] || { echo \"[gpu$g] gate $GATE failed\"; exit 1; }; "
  for n in ${CH[$g]}; do
    SCRIPT+="echo \"[gpu$g] start $n \$(date -u +%FT%TZ)\"; chmod +x /root/podjobs/$n/launch.sh; bash /root/podjobs/$n/launch.sh; echo \"[gpu$g] $n rc=\$(cat /root/podjobs/$n/exit_code 2>/dev/null) \$(date -u +%FT%TZ)\"; "
  done
  SCRIPT+="echo \"[gpu$g] chain done \$(date -u +%FT%TZ)\" ) & "
done
SCRIPT+="wait; echo \"[gpu-chains] all done \$(date -u +%FT%TZ)\""
ssh -o BatchMode=yes -o ConnectTimeout=25 -i "${POD_KEY:-$HOME/.ssh/id_ed25519}" -p "${HP##*:}" "${HP%%:*}" \
  "cat > /root/podjobs/gpu_chains.sh <<'EOS'
#!/bin/bash
$SCRIPT
EOS
chmod +x /root/podjobs/gpu_chains.sh; setsid nohup bash /root/podjobs/gpu_chains.sh >> /root/podjobs/gpu_chains.log 2>&1 < /dev/null & echo \$! > /root/podjobs/gpu_chains.pid; sleep 1; echo \"gpu_chains pid \$(cat /root/podjobs/gpu_chains.pid) alive=\$(kill -0 \$(cat /root/podjobs/gpu_chains.pid) 2>/dev/null && echo yes || echo no); chains: ${!CH[*]}\""
