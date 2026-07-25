#!/bin/bash
# wait for round G to finish, then run round H (no-gate ablation)
while ! grep -q "DONE\." /workspace/logs/driver_ac90g.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90h_ogbench.sh > /workspace/logs/ac90h_nohup.log 2>&1
