#!/bin/bash
# wait for round V to finish, then run round Z (zero-init comparison)
while ! grep -q "DONE\." /workspace/logs/driver_ac90v.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90z_ogbench.sh > /workspace/logs/ac90z_nohup.log 2>&1
