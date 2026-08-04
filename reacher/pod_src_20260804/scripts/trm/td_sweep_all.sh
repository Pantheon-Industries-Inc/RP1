#!/bin/bash
# TD n x gamma sweep on ALL trajectories (full mixed cache), eval hard n100 at CEM ~150.
# References: latent, regression-TRM (all trajectories), shuffled, oracle.
cd /root/stable-worldmodel
export TQDM_DISABLE=1
CACHE=caches/lewm_tworoom.pt   # fs1 mixed = ALL logged trajectories (expert+random)
EVAL="--wm lewm_tworoom --dataset tworoom_pixels.lance --action-block 5 --horizon 10 --eval-budget 100 --n-per-class 50 --min-geo 90 --max-geo 180 --num-samples 30 --cem-steps 5 --topk 5 --seed 1"

# --- references (all-trajectory regression + latent + shuffled + oracle) ---
CUDA_VISIBLE_DEVICES=0 python3 scripts/trm/train_metric.py --cache $CACHE --learner regression --out metrics/all_regression.pt --steps 8000 --device cuda 2>&1 | grep -iE "saved|Error"
CUDA_VISIBLE_DEVICES=0 python3 scripts/trm/train_metric.py --cache $CACHE --learner regression --labels shuffled --out metrics/all_shuffled.pt --steps 8000 --device cuda 2>&1 | grep -iE "saved|Error"
CUDA_VISIBLE_DEVICES=0 python3 scripts/trm/eval_hard.py $EVAL --condition latent --condition regression=metrics/all_regression.pt --condition shuffled=metrics/all_shuffled.pt --condition oracle --device cuda --out results/all_refs.txt 2>&1 | grep -iE "n100=|Error"

run_cfg() { # gpu head n gamma
  local gpu=$1 head=$2 n=$3 g=$4
  local tag="${head:0:1}_n${n}_g${g//./}"
  local cond="${head:0:1}n${n}g${g//./}"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/trm/train_metric.py --cache $CACHE \
    --learner td --head "$head" --n-step "$n" --gamma "$g" --out "metrics/atd_$tag.pt" \
    --steps 6000 --device cuda 2>&1 | grep -iE "Error|Traceback"
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/trm/eval_hard.py $EVAL \
    --condition "$cond=metrics/atd_$tag.pt" --device cuda --out results/td_sweep_all.txt 2>&1 \
    | grep -iE "n100=|Error"
}
( for c in "quasimetric 1 0.9" "quasimetric 5 0.9" "quasimetric 10 0.9" "quasimetric 20 0.9" "quasimetric 5 0.95" "quasimetric 1 0.99" "mlp 5 0.99"; do run_cfg 0 $c; done ) &
( for c in "quasimetric 1 0.95" "quasimetric 10 0.95" "quasimetric 20 0.95" "quasimetric 5 0.99" "quasimetric 10 0.99" "quasimetric 20 0.99" "mlp 10 0.95"; do run_cfg 1 $c; done ) &
wait
echo "=== TD_SWEEP_ALL_DONE ==="
