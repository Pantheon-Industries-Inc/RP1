"""Add --lambda-schedule / --lambda-base to train_lip_ac.py.

The actor loss is currently

    L = v_K + mean_weight * mean(v_0 .. v_K)

i.e. UNIFORM weight on every refinement iteration, plus a double count on the
terminal v_K. This generalises it to

    L = v_K + sum_k lambda_k * v_k

with three schedules:

  uniform     lambda_k = mean_weight / (K+1)          (exactly the current loss)
  geom-early  lambda_k = mean_weight * base**k        (early iterations dominate)
  geom-late   lambda_k = mean_weight * base**(K-1-k)  (late iterations dominate)

Motivation, from the seed-0 post-mortem (2026-07-31): across 20 actors with
complete cards, corr(E_final, HELD) = +0.583 -- a LOWER final imagined cost
reliably predicts WORSE real performance. The best-imagined actor (E_final 0.778)
held 11.0; the worst-imagined (1.921) held 51.3. That is world-model
exploitation: the 25-step open-loop error (~0.11 rad) exceeds the 0.05 rad
tolerance, so an actor that pushes the imagined cost far below what the model can
support is optimising fiction.

A terminal-heavy objective is exactly what encourages that, so `geom-early`
(reward being good EARLY, stop rewarding further refinement) is the principled
counter-measure, and `geom-late` is its falsification twin -- it should make the
pathology worse. If the mechanism is real, geom-early lifts the low-E_final
seeds (seed 0) without hurting the others.
"""

import ast

P = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
s = open(P).read()
if "--lambda-schedule" in s:
    print("already patched")
    raise SystemExit

OLD_ARG = '    p.add_argument("--mean-weight", type=float, default=0.1)'
NEW_ARG = '''    p.add_argument("--mean-weight", type=float, default=0.1)
    p.add_argument("--lambda-schedule", choices=["uniform", "geom-early", "geom-late"],
                   default="uniform",
                   help="how the per-iteration values v_k are weighted in the actor "
                        "loss. uniform = the historical mean(); geom-early weights the "
                        "FIRST refinement iterations (regularises against world-model "
                        "exploitation, which anti-correlates with success at r=+0.58); "
                        "geom-late weights the LAST ones (the falsification twin)")
    p.add_argument("--lambda-base", type=float, default=0.5,
                   help="geometric ratio for the non-uniform schedules")'''
assert s.count(OLD_ARG) == 1, f"arg anchor x{s.count(OLD_ARG)}"
s = s.replace(OLD_ARG, NEW_ARG)

OLD_LOSS = "        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()"
NEW_LOSS = '''        _stack = torch.stack(e_path)
        if a.lambda_schedule == "uniform":
            loss = e_path[-1] + a.mean_weight * _stack.mean()
        else:
            _K = len(e_path)
            if a.lambda_schedule == "geom-early":
                _w = [a.lambda_base ** k for k in range(_K)]
            else:                                    # geom-late
                _w = [a.lambda_base ** (_K - 1 - k) for k in range(_K)]
            _n = sum(_w) or 1.0                      # normalise so mean_weight keeps its scale
            _wt = torch.tensor([wi / _n for wi in _w], device=_stack.device,
                               dtype=_stack.dtype)
            loss = e_path[-1] + a.mean_weight * (_wt * _stack).sum()'''
assert s.count(OLD_LOSS) == 1, f"loss anchor x{s.count(OLD_LOSS)}"
s = s.replace(OLD_LOSS, NEW_LOSS)
open(P, "w").write(s)
ast.parse(open(P).read())
print("train_lip_ac.py: --lambda-schedule {uniform,geom-early,geom-late} + --lambda-base; syntax OK")
