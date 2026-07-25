#!/bin/bash
# wait for round Z to finish, then run round G (small-init, scalar iteration)
while ! grep -q "DONE\." /workspace/logs/driver_ac90z.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90g_ogbench.sh > /workspace/logs/ac90g_nohup.log 2>&1
