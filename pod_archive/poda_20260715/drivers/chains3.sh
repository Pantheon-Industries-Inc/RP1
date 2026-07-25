#!/bin/bash
while ! grep -q "DONE\." /workspace/logs/driver_ac90s2.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90s3_ogbench.sh > /workspace/logs/ac90s3_nohup.log 2>&1
