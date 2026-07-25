#!/bin/bash
while ! grep -q "DONE\." /workspace/logs/driver_ac90s3.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90s4_ogbench.sh > /workspace/logs/ac90s4_nohup.log 2>&1
