#!/bin/bash
while ! grep -q "DONE\." /workspace/logs/driver_ac90s4.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90s5_ogbench.sh > /workspace/logs/ac90s5_nohup.log 2>&1
