"""One-command TRM sweep for a task: collect -> train WM -> cache -> train the
three metrics (+ shuffled control) -> evaluate all cost conditions with CEM-MPC.

Chains the other ``scripts/trm`` entrypoints so a whole task reproduces in one
call. Steps already completed can be skipped with ``--skip``.

Example (TwoRoom state path)::

    python scripts/trm/run_sweep.py --task tworoom --num-eval 40 --cross-wall
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable

TASKS = {
    "tworoom": {
        "env": "swm/TwoRoom-v1",
        "dataset": "tworoom_expert.lance",
        "obs_key": "proprio",
        "wm": "statewm_tworoom",
        "scale": 60,
    },
}


def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="tworoom", choices=list(TASKS))
    p.add_argument("--num-eval", type=int, default=40)
    p.add_argument("--cross-wall", action="store_true")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--receding", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--cem-steps", type=int, default=8)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--eval-budget", type=int, default=60)
    p.add_argument("--wm-steps", type=int, default=4000)
    p.add_argument("--metric-steps", type=int, default=5000)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip", default="", help="comma list of steps to skip: collect,wm,cache,metrics,eval")
    args = p.parse_args()

    t = TASKS[args.task]
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    cache = f"caches/{args.task}_state.pt"
    mdir = Path("metrics")

    if "collect" not in skip:
        run([PY, HERE / "../data/collect_tworooms.py", "num_traj=400",
             "world.num_envs=10", "world.max_episode_steps=100"])
    if "wm" not in skip:
        run([PY, HERE / "train_state_wm.py", "--dataset", t["dataset"],
             "--run-name", t["wm"], "--obs-key", t["obs_key"],
             "--steps", args.wm_steps, "--device", args.device])
    if "cache" not in skip:
        run([PY, HERE / "cache_latents.py", "--wm", t["wm"], "--dataset", t["dataset"],
             "--out", cache, "--device", args.device])
    if "metrics" not in skip:
        for learner in ("regression", "td", "contrastive"):
            run([PY, HERE / "train_metric.py", "--cache", cache, "--learner", learner,
                 "--out", mdir / f"{args.task}_{learner}.pt", "--steps", args.metric_steps,
                 "--scale", t["scale"], "--device", args.device])
        run([PY, HERE / "train_metric.py", "--cache", cache, "--learner", "regression",
             "--labels", "shuffled", "--out", mdir / f"{args.task}_shuffled.pt",
             "--steps", args.metric_steps, "--scale", t["scale"], "--device", args.device])

    if "eval" not in skip:
        cmd = [PY, HERE / "eval_trm.py", "--env", t["env"], "--wm", t["wm"],
               "--dataset", t["dataset"], "--num-eval", args.num_eval,
               "--goal-offset", args.goal_offset, "--eval-budget", args.eval_budget,
               "--horizon", args.horizon, "--receding", args.receding,
               "--num-samples", args.num_samples, "--cem-steps", args.cem_steps,
               "--condition", "latent",
               "--condition", f"trm_regression={mdir}/{args.task}_regression.pt",
               "--condition", f"trm_td={mdir}/{args.task}_td.pt",
               "--condition", f"trm_contrastive={mdir}/{args.task}_contrastive.pt",
               "--condition", f"hybrid={mdir}/{args.task}_regression.pt",
               "--condition", f"shuffled={mdir}/{args.task}_shuffled.pt",
               "--condition", "oracle", "--scsa",
               "--device", args.device, "--out", f"results/{args.task}_sweep.txt"]
        if args.cross_wall:
            cmd.append("--cross-wall")
        run(cmd)
    print("\nSweep complete.")


if __name__ == "__main__":
    main()
