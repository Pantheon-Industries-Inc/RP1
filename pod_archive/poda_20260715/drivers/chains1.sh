#!/bin/bash
# wait for round H to finish, then run round S1 (sweep around H)
while ! grep -q "DONE\." /workspace/logs/driver_ac90h.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90s1_ogbench.sh > /workspace/logs/ac90s1_nohup.log 2>&1
