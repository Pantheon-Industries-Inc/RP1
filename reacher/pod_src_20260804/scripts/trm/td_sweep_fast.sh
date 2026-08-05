#!/bin/bash
# High-parallelism TD n x gamma sweep on ALL trajectories, eval hard n100 at CEM ~150.
# Eval is CPU-bound (GPU ~0%, per-step 100-env pixel pipeline), so run many jobs concurrently
# across the 2 idle GPUs + 144 cores. Each job is self-contained (trains its metric, then evals)
# and writes its own result file (no append races).
cd /root/stable-worldmodel
export TQDM_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CACHE=caches/lewm_tworoom.pt
EVAL="--wm lewm_tworoom --dataset tworoom_pixels.lance --action-block 5 --horizon 10 --eval-budget 100 --n-per-class 50 --min-geo 90 --max-geo 180 --num-samples 30 --cem-steps 5 --topk 5 --seed 1 --solver-batch 1024"
MAXJOBS=10
mkdir -p results/sweep_all
sem() { while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 3; done; }

# job: gpu, tag, condition-spec, [head n gamma] (if td, train first)
job() {
  local gpu=$1 tag=$2 cond=$3 head=$4 n=$5 g=$6
  if [ -n "$head" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/trm/train_metric.py --cache $CACHE --learner td \
      --head "$head" --n-step "$n" --gamma "$g" --out "metrics/atd_$tag.pt" --steps 6000 --device cuda >/dev/null 2>&1
  fi
  CUDA_VISIBLE_DEVICES=$gpu python3 scripts/trm/eval_hard.py $EVAL --condition "$cond" --device cuda \
    --out "results/sweep_all/$tag.txt" >/dev/null 2>&1
  echo "done $tag"
}

# reference metric trains (quick relative to evals; do up front)
CUDA_VISIBLE_DEVICES=0 python3 scripts/trm/train_metric.py --cache $CACHE --learner regression --out metrics/all_regression.pt --steps 8000 --device cuda >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=1 python3 scripts/trm/train_metric.py --cache $CACHE --learner regression --labels shuffled --out metrics/all_shuffled.pt --steps 8000 --device cuda >/dev/null 2>&1

i=0
launch() { sem; job "$@" & i=$((i+1)); }
# references
launch $((i%2)) latent     "latent"
launch $((i%2)) regression "regression=metrics/all_regression.pt"
launch $((i%2)) shuffled   "shuffled=metrics/all_shuffled.pt"
launch $((i%2)) oracle     "oracle"
# TD grid: quasimetric n in {1,5,10,20} x gamma in {0.9,0.95,0.99} + 2 MLP
for n in 1 5 10 20; do for g in 0.9 0.95 0.99; do
  launch $((i%2)) "q_n${n}_g${g//./}" "qn${n}g${g//./}=metrics/atd_q_n${n}_g${g//./}.pt" quasimetric $n $g
done; done
launch $((i%2)) "m_n5_g099"  "mn5g099=metrics/atd_m_n5_g099.pt"   mlp 5 0.99
launch $((i%2)) "m_n10_g095" "mn10g095=metrics/atd_m_n10_g095.pt" mlp 10 0.95
wait
echo "=== FAST_SWEEP_DONE ==="
