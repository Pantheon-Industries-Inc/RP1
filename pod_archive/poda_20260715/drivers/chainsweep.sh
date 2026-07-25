#!/bin/bash
# input-sweep chain: S3 (zg0+champion-ng) -> S5 (z00+min0) -> S6 (winner gate test)
while ! grep -q "DONE\." /workspace/logs/driver_ac90s3.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90s5_ogbench.sh > /workspace/logs/ac90s5_nohup.log 2>&1
while ! grep -q "DONE\." /workspace/logs/driver_ac90s5.log 2>/dev/null; do sleep 60; done
bash /workspace/run_ac90s6_ogbench.sh > /workspace/logs/ac90s6_nohup.log 2>&1
