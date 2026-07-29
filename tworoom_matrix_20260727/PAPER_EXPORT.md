# LaTeX export — LIPv4 vs native-latent planning (paper-budget protocol)

All numbers: success rate (%), n = 50 episodes/cell, `swm/TwoRoom-v1`, std surface.
CEM at the paper's budget (300 samples × **10 iterations**, top-30, var 1.0 — §9.3);
LIPv4 at K = 8 refinement passes, deploy `restarts 1, n_steps 0` (pure learned
planner). h25 / h50 = goal offset 25 / 50 primitive steps. Superscripts: number of
independent draws (³ = 3 task seeds; ⁹ = 3 actor/critic seeds × 3 task seeds).
Search cost in world-model rollout-equivalents per planning step.

## Main table

```latex
\begin{table}[t]
\centering
\caption{Amortized planning (LIPv4) versus native latent-cost search (CEM) on
TwoRoom, at the planning budget of \citet{lewm}: 300 candidates $\times$ 10 CEM
iterations. LIPv4 replaces search with $K{=}8$ learned refinement passes
($\sim$16 rollout-equivalents per planning step, versus 3{,}000 for CEM,
$\sim$190$\times$ less). Success \% over 50 episodes per cell; $h25$/$h50$: goal
25/50 primitive steps ahead. LIPv4 cells aggregate 3 actor seeds $\times$ 3 task
seeds; latent-CEM cells 3 task seeds.}
\label{tab:lip-vs-latent}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{LeWM} & \multicolumn{2}{c}{PLDM} & \multicolumn{2}{c}{DINO-WM} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
method & $h25$ & $h50$ & $h25$ & $h50$ & $h25$ & $h50$ \\
\midrule
Latent + CEM (3{,}000 ro.) & 86.7 & 54.0 & 97.3 & 76.7 & \textbf{100.0} & 99.3 \\
LIPv4, $K{=}8$ ($\sim$16 ro.) & \textbf{100.0} & \textbf{100.0} & \textbf{97.1} & \textbf{98.9} & 99.3 & 99.3 \\
\bottomrule
\end{tabular}
\end{table}
```

## Appendix tables

```latex
% ---------------------------------------------------------------- budget sweeps
\begin{table}[h]
\centering
\caption{Planning-budget sensitivity. CEM is flat between 30 and 10 iterations
(max cell shift $-2.6$). LIPv4's budget knob (deploy-time truncation of the
weight-tied refiner to $K$ passes, no retraining) is free at $K{=}4$ on both
world models and degrades gracefully at $K{=}2$, and only where the latent is
weak. DINO-WM $K{=}4$ (separately retrained, not truncated) matched $K{=}8$
identically (99.3 six-cell mean).}
\label{tab:budget}
\begin{tabular}{llcccc}
\toprule
& rollout-eq. & \multicolumn{2}{c}{LeWM} & \multicolumn{2}{c}{PLDM} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
method / budget & per step & $h25$ & $h50$ & $h25$ & $h50$ \\
\midrule
CEM, 30 iters      & 9{,}000 & 100.0 & 99.8  & 98.7 & 99.6 \\
CEM, 10 iters      & 3{,}000 & 100.0 & 100.0 & 98.2 & 99.1 \\
LIPv4 $K{=}8$      & $\sim$16 & 100.0 & 100.0 & 97.1 & 98.9 \\
LIPv4 $K{=}4$      & $\sim$8  & 100.0 & 100.0 & 96.7 & 99.6 \\
LIPv4 $K{=}2$      & $\sim$4  & 99.1  & 100.0 & 93.3 & 97.8 \\
\bottomrule
\end{tabular}
\end{table}

% ------------------------------------------------- single-pass amortization
\begin{table}[h]
\centering
\caption{Iterative refinement is load-bearing: a reactive policy
$\pi(z, z_g)$ trained offline by first-order extraction through the frozen
world model (PWM-style~\citep{pwm}, but scored by the same MRN quasimetric
LIPv4 refines against) collapses, while two refinement passes over the same
value landscape ($K{=}2$, comparable deploy budget) hold 93--100.
\emph{proto}: all 5 action blocks executed per replan, tail imagined;
\emph{rh1}: replan every block (zero deploy rollouts). TD+CEM at the paper
budget shown for reference. DINO-WM: 1 training seed (others: 3).}
\label{tab:pwm}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{LeWM} & \multicolumn{2}{c}{PLDM} & \multicolumn{2}{c}{DINO-WM} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
method & $h25$ & $h50$ & $h25$ & $h50$ & $h25$ & $h50$ \\
\midrule
reactive $\pi$, proto & 35.3 & 21.1 & 30.4 & 9.3  & 36.0 & 20.7 \\
reactive $\pi$, rh1   & 38.0 & 22.9 & 31.8 & 14.9 & 36.0 & 20.7 \\
LIPv4 $K{=}2$         & 99.1 & 100.0 & 93.3 & 97.8 & --- & --- \\
TD + CEM, 10 iters    & 100.0 & 100.0 & 98.2 & 99.1 & 100.0 & 100.0 \\
\bottomrule
\end{tabular}
\end{table}
```

## Numbers provenance

| table | journal | driver |
|---|---|---|
| main + budget (CEM rows) | `results/it10_driver.log` | `code/it10_rerun.sh` |
| budget (K rows) | `results/lipk_driver.log` | `code/lipk_sweep.sh` |
| PWM | `results/pwm_driver.log` | `code/tworoom_pwm.sh` |
| LIPv4 K=8 / CEM-30 rows | campaign §9.2 | `code/tworoom_matrix.sh` |

Caveat carried from RESULTS §11.1: no train/eval split — TD/LIP/PWM numbers are
in-distribution upper bounds; the latent-CEM rows are unaffected (no learned
component beyond the frozen WM).
