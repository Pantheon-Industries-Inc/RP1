#!/bin/bash
# Upload the migration bundle to a new pod. Usage: bash upload_to_newpod.sh <HOST> <PORT>
set -e
HOST=${1:?usage: upload_to_newpod.sh HOST PORT}
PORT=${2:?usage: upload_to_newpod.sh HOST PORT}
KEY=~/.ssh/id_ed25519
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "== checking pod =="
ssh -p "$PORT" -i "$KEY" root@"$HOST" 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c; df -h /workspace | tail -1'

echo "== uploading bundle (1.59GB) + scripts =="
scp -P "$PORT" -i "$KEY" "$DIR/pod_bundle_20260709.tgz" "$DIR/replicate84_newpod.sh" root@"$HOST":/workspace/

echo "== unpacking =="
ssh -p "$PORT" -i "$KEY" root@"$HOST" 'cd /workspace && tar -xzf pod_bundle_20260709.tgz && md5sum pod_bundle_20260709.tgz && rm pod_bundle_20260709.tgz && ls'

echo "done. Next: environment + dataset download per BOOTSTRAP_NEWPOD.md sections 2-3."
