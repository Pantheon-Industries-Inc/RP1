#!/bin/bash
# wait for round S1 to finish, then run round S2 (actor-loss ablation)
while ! grep -q "DONE\." /workspace/logs/driver_ac90s1.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90s2_ogbench.sh > /workspace/logs/ac90s2_nohup.log 2>&1
