# LIP + Dyna — Serialization & Parallelization Audit

Audience: someone who optimizes kernels / systems. Written 2026-07-25 against
`galilai-group/stable-worldmodel` @ `b298aa70` plus `Dyna/dyna_harness`.
Reusable LIP/TRM code now lives under `src/rlp/`; campaign-specific shell
drivers were removed from the application tree. References prefixed with
`main:Dyna/dyna_harness/` identify their historical locations on Git `main`.

Every claim is anchored to `file:line`. Numbers are labelled **[measured]** (from the repo's own
benchmarks/logs, or from experiments run for this audit) or **[derived]** (static FLOP count from
`assets/core/world_model/lewm_cube/config.json`). Nothing here is a guess about what the code *probably* does.

---

## 0. TL;DR — the one number

The flagship planner, LIP, does **16.9 GFLOP of work per decision and takes 560 ms on an H100.**

That is **0.004% of peak**. [derived FLOPs / measured wall-clock]

The repo's own benchmark table (`rlp_writeup_20260717/rlp.tex:200-203`) makes the diagnosis for us:

| per decision | CEM | MPPI | LIP |
|---|---|---|---|
| WM rollouts | 9,000 | 9,000 | 17 fwd + 8 bwd |
| wall-clock, **1 env** | 0.42 s | 0.38 s | **0.56 s** |
| wall-clock, **50 envs** | 5.8 s | 19.4 s | **0.58 s** |

LIP's batch-50 call costs the same as its batch-1 call. That is the signature of a workload that is
**100% kernel-launch / Python-dispatch bound and 0% compute bound** — roughly **25,000–35,000
sequential kernel launches per decision**, each on a `(B·3, 192)` tensor.

**The algorithmic win (~530× fewer rollouts than CEM) is currently cashed at ~1× (B=1) to ~10× (B=50).**

But **the planner is not where the wall-clock is.** It is invoked twice per 50-step episode. Four
independent things are all true and all fixable, and they are not in the same subsystem:

1. **Collection/eval runs the GPU at ~1.5% duty** — 119 s per 50-env call, of which ~1.8 s is
   planning. The inner loop steps 50 MuJoCo envs and renders 50 frames in a **single-threaded Python
   `for`**. On-policy collection is ~45% of a Dyna round. (§4.1)
2. **Its 3× process parallelism is disabled by an abort whose root cause is now confirmed:**
   `MUJOCO_EGL_DEVICE_ID` is never set anywhere in the repo, so `CUDA_VISIBLE_DEVICES=0/1/2` spreads
   CUDA while **all** EGL render contexts pile onto physical GPU 0. One abort then latches
   `PAR=3 → 1` permanently. (§4.1b, §5.2)
3. **~47% of LIP's forward rollout work is provably dead** — a duplicate rollout that the *trainer*
   already eliminated and the *solver* never did. One line. (§3.1)
4. **Nothing is compiled, graphed, or run in reduced precision** on the LIP path — although the eval
   entrypoint, the DINO world-model, and the original WM training all use bf16/`torch.compile`. (§6)

---

## 0.5 Update 2026-08-12/13 — the launch-bound diagnosis, cashed

Fixes #3 and #17 of §7 have landed in the LIPv4 (kind `lip4`) solve path; measured on an
architecture-matched benchmark, job 4840, 1×H200, fp32, B=1:

| LIPv4 decision | ms/decision | vs eager |
|---|---|---|
| eager, duplicate rollout (the audited path) | 272.6 | 1.0× |
| **CUDA-graphed** (`core.solver.graphed=true`) | **29.3** | **9.3×** |

- **Trajectory reuse (#3) is now the solver default** (`core.solver.reuse_trajectory=true`,
  since 2026-08-12): the gradient unroll's trajectory doubles as the actor features, so one
  decision costs **9 forward + 8 backward unrolls** instead of 17 fwd + 8 bwd. Bit-exact:
  `reuse_trajectory="verify"` logs `max_difference == 0` per iteration, and end-task parity
  held on the reacher winners (jobs 4927/4930 — 3 of 4 report cells identical, the fourth
  differs by a single episode in 450). `reuse_trajectory=false` reproduces pre-2026-08-12 evals.
- **CUDA graphs (#17) are implemented opt-in** (`rlp/core/solver/graphed.py`,
  `core.solver.graphed = false | true | "verify"`): the per-iteration score computation
  (WM unroll → trajectory value, forward AND backward) is graph-captured once per batch size
  and replayed per refinement iteration. Final-plan deviation vs the eager path: **5.96e-8**
  (one float32 ulp — identical to the trajectory-reuse deviation, i.e. capture itself adds
  zero). Capture costs seconds, once per (batch, horizon, action_dim); shrinking eval batches
  must be padded by the caller or left on the eager path (the #12 constraint, handled at the
  capture boundary). Covered by `tests/core/test_graphed_refinement.py`.
- **Planner-vs-planner, same benchmark** (H200, fp32): LIPv4 graphed **30.7 ms/decision** vs
  CEM 218.6 ms graphed / 241.9 ms eager — the ~530× rollout advantage now cashes at **~7.1×**
  wall-clock at B=1 instead of the ~1× this audit measured, and the gap widens with batch
  (the samplers' 9,000 rollouts scale with controllers; LIP's 9 do not).

The remaining §7 items on the LIP path (#7, #8, #9, #10) are subsumed for *inference* by the
graph capture (a replayed graph has no per-kernel launch or sync cost) but still stand for
*training* (`rlp/train/lip_ac.py`), which reuses the scoring rollout
(`reuse_refinement_rollouts=true`, same 2026-08-12 default) but is not graph-captured.

---

## 1. Three regimes, three different bottlenecks

Do not optimize "LIP". Optimize one of these; they share code but nothing else.

| regime | what dominates | GPU utilization | scale |
|---|---|---|---|
| **A. Deploy / eval / Dyna collection** | serial per-env MuJoCo `step()` + 224×224 render | **~1.5%** [measured, §4.1] | 119 s per 50-env call ≈ 48 ms/env-step |
| **B. LIP actor–critic training** | the **40-deep frozen-WM autoregressive chain**, launch-bound | 4–14% of fp32 peak [derived] | 0.78 s/step → 1.3–1.7 h per actor × 3 seeds × every round |
| **C. Dyna round orchestration** | shell-level **barriers and idle GPUs** | ~48% process occupancy [measured, §5.1] | 4–6 h/round (14 h for round 2) |

---

## 2. Shape / cost reference card

From `assets/core/world_model/lewm_cube/config.json` and `configs/inference/benchmark/cube.yaml`. Everything is small — that is the problem.

```
encoder    ViT-tiny, patch 14, 224px   -> 257 tokens, d=192, depth 12    2.73 GFLOP / image  [derived]
           (confirmed at runtime: hidden_size 192, num_hidden_layers 12,
            num_attention_heads 3, intermediate_size 768, patch_size 14)
predictor  depth 6, d=192, heads 16 x dim_head 64 (inner 1024), mlp 2048
           window T = 3 frames, causal SDPA (wm/lewm/module.py:53,66)
           -> 69.3 MFLOP per predict() call per sample                                       [derived]
pred_proj / projector   MLP 192 -> 2048 -> 192 (BatchNorm1d)
action_encoder          Conv1d(25->10, k=1) + Linear(10->768) + Linear(768->192)

value head QuasimetricHead(192, hidden 256, embed 128)  ~147k params    — negligible
actor      PlannerNet v4: Linear(251,512)/(512,512)/(512,125) ~455k     — negligible

horizon H = 5 action blocks (a_dim = 25 = 5 env dims x frameskip 5)
LIP iterations K = 8            history_len = 1 (policy.py:30)  <- padded to 3 by repetition
num_envs = 50                   eval_budget = 50 -> 2 replans/episode
```

Per-decision budget at K=8, H=5 **[derived]**:

```
forward predict() calls   85   = K x 2 rollouts x H  +  H final selection
backward                  40   = K x H            (costs ~2x forward)
encoder                    2 images (1 obs + 1 goal)

B=1    encoder   5.5 + predictor  11.4 =  16.9 GFLOP  / 0.56 s ->    30 GFLOP/s = 0.004% peak
B=50   encoder 272.9 + predictor 571.8 = 844.6 GFLOP  / 0.58 s ->  1456 GFLOP/s = 0.21%  peak
```

GEMM shapes are `M = B·3, K = 192, N ∈ {3072, 1024, 2048, 1152}`. **K=192 is far too skinny for
tensor cores**, and at B=1, M=3. This will never be compute-bound at these batch sizes; the only
levers are *launches per decision* and *bytes moved*.

---

## 3. Regime B — LIP actor–critic training (`rlp/train/lip_ac.py`)

**[historical measurement]** 0.77–0.79 s/step; 6000 steps ≈ **1.3 h
per actor**; ~2.2 TFLOP/step at B=128 → **~2.8 TFLOP/s effective**, i.e. 4–14% of fp32 peak.
~30,000 kernel launches per training step ≈ **26 µs of wall-clock per kernel**.

### The serial spine

Two nested Python loops, 40 deep, with only B=128 of parallelism inside each level:

```python
# rlp/train/lip_ac.py:588
for k in range(a.iters):                       # K = 8, genuinely sequential (learned optimizer)
    traj = rollout_traj(wm, zh, ah, A_in)      # :591   H=5 serial WM steps
    (gA,) = torch.autograd.grad(...)           # :592   backward through those 5
    tr = rollout_traj(wm, zh, ah, A)           # :611   H=5 serial WM steps
```

```python
# rlp/core/rollout.py:35
for t in range(plan.shape[1]):                 # H = 5, autoregressive — irreducibly serial
    win_e = torch.stack(embs[-3:], dim=1)      # :37   reallocates the window every step
    win_a = torch.stack(acts[-3:], dim=1)      # :38
    nxt = wm.predict(win_e, wm.action_encoder(win_a))[:, -1]   # :39
```

The **depth** is irreducible. The **constant factor** is very reducible.

### 3.1 The duplicate rollout — a one-line, ~2× win, already written elsewhere in the repo

`rlp/core/solver/lip.py:251-255`, the **inference** path:

```python
251    A_in = A.detach().requires_grad_(True)
252    traj = rollout_traj(wm, zh_r, a_hist, A_in)          # rollout #1 (grad)
253    (gA,) = torch.autograd.grad(self.lip_value(traj[:, -1], zg_r).sum(), A_in)
254    with torch.no_grad():
255        traj_f = rollout_traj(wm, zh_r, a_hist, A)       # rollout #2 — IDENTICAL VALUES
```

`A` is not modified between 251 and 255, `A_in` holds `A`'s values, and the WM is deterministic in
`.eval()` (dropout off, BatchNorm on running stats). So `traj_f == traj.detach()`, exactly.
**40 of the solver's 85 forward predict calls (47%) recompute a tensor it already has.** `E` at
`:256` is likewise the scalar already summed at `:253`.

The trainer already knows this — `rlp/train/lip_ac.py:594-596`:

```python
593    # A_in holds A's values, so the grad pass's trajectory IS the feature
594    # trajectory — reuse it instead of a third WM rollout (value-identical)
595    traj_f = traj.detach()
```

**Fix: port line 595 into `rlp/core/solver/lip.py:255`.** An env-gated patch that also verifies value-identity is in
`rlp/tools/patch/patch_lip_reuse_traj.py` (`LIP_REUSE_TRAJ=1` to enable, `=verify` to print
`max|traj_f - traj.detach()|` per iteration).

A second, *inter*-iteration instance exists that neither has fixed: in the trainer, `traj` at `:591`
in iteration `k+1` recomputes what `tr` at `:611` produced in iteration `k`. Taking `gA` off the
existing `:611` graph via `autograd.grad(..., inputs=A, retain_graph=True)` cuts the trainer from
80 → 41 forward WM calls and removes 7 of 8 extra backward passes.

### 3.2 Discarded compute inside every rollout step

`wm/lewm/lewm.py:44-52` runs the predictor and `pred_proj` over **all 3 window tokens**; the caller
keeps only `[:, -1]` (`rlp/core/rollout.py:39`, `lewm.py:100`).

- `pred_proj` (192→2048→192) over 3 tokens when 1 is needed — **free 2/3 saving**, one line.
- The transformer body genuinely needs all 3 positions at every layer, so you cannot trim it
  within a call.
- **Across** rollout steps you also cannot cache: attention *is* causal (`module.py:53,66`), but the
  window slides while `pos_embedding[:, :T]` is indexed from 0 (`module.py:291`), so a frame's
  positional offset changes every step and its K/V is not reusable. RoPE or a time-invariant slot
  convention would unlock a ~3× KV-cache win — but it changes numerics and needs a retrain.
- `wm.action_encoder(win_a)` (`rlp/core/rollout.py:39`) re-encodes all 3 action blocks each step
  when 2 were encoded the previous step. LeWM's own `rollout()` batches this correctly
  (`lewm.py:88-90`); `rollout_traj` does not.
- `Embedder.forward` (`wm/lewm/module.py:200-209`) does `permute → Conv1d(k=1) → permute`. A
  kernel-size-1 Conv1d **is** a Linear; the permutes are pure memory shuffling.

### 3.3 Host↔device syncs on the critical path

Three `.item()` calls per training step, **24,000 over a run**, for values printed once every 500:

```python
train_lip_ac.py:500    return loss.item()                              # critic_step
train_lip_ac.py:626    return e_path[0].item(), e_path[-1].item(), ... # actor_step
train_lip_ac.py:665    if step % 500 == 0:                             # ...only read here
```

### 3.4 Single-threaded Python samplers with no DataLoader

**No `torch.utils.data.DataLoader` anywhere in the LIP training path** — no `num_workers`, no
`prefetch_factor`, no `pin_memory`, no `non_blocking=True`. Both samplers are per-item Python loops:

```python
rlp/core/value/samplers.py:195   for b in range(batch_size):     # 1024 iterations, per critic step
rlp/core/value/samplers.py:180       edges = np.linspace(...)    # a fresh allocation PER SAMPLE
train_lip_ac.py:398          for _ in range(B):              # 128 iterations, per actor step
train_lip_ac.py:412              [blocks(e, ...) for k in range(a.horizon)]   # 640 calls/step
```

~6,000 Python-level RNG calls per critic step; ~37M over a run. Gathers come from a **1.54 GB
CPU-resident** fs1 cache (2.01M × 192 fp32 — `LatentCache.load` pins to CPU at
`rlp/data/latent_cache.py:75` and nothing moves it), then **8 blocking pageable H2D copies per
step** (`train_lip_ac.py:422,423,473,475-477`).

**Dead work:** `aref` is computed unconditionally at `:411` (640 of 896 per-step `blocks()` calls)
but consumed only under `if a.bc_weight > 0` at `:618`, and `bc_weight=0.0` by default.

**Cold start:** `LatentCache.episodes()` (`rlp/data/latent_cache.py:53-55`) is O(E × N) — 10,000
full boolean masks over 2.01M rows ≈ 2×10¹⁰ comparisons, single-threaded, before step 0.

### 3.5 Memory caps the batch at 128

All 8 differentiable rollouts stay alive until `loss.backward()` at `:621`, because `e_path`
(`:586`) is reduced by `torch.stack(e_path).mean()` at `:617` → **~2 GB of retained WM activations**.
Per-rollout-step activation checkpointing (the WM is frozen — no weight grads) frees most of it.
**A bigger batch costs almost nothing in wall-clock in a launch-bound regime.**

The DINO world-model already does this — `rlp/core/world_model/dinowm/tokens.py:167-169`:

```python
167    # the K-step LIP unroll keeps every iteration's rollout graph alive for
168    # the final backward — full fp32 activations OOM an 80GB H100 at B=128,
169    # so recompute predictor activations in backward instead of storing them
```

### 3.6 No multi-GPU inside a run

No DDP, no FSDP, no gradient accumulation. Single-GPU by construction
(`train_lip_ac.py:336-341`). All multi-GPU use is shell-level fan-out over seeds — a 4-GPU box gets
4× *sweep* throughput and **zero** speedup on any single actor's 1.3 h latency.

---

## 4. Regime A — deploy / eval / collection rollout

### 4.1 The dominant cost: 50 serial software renders per step

`world/env_pool.py:134-140` is the whole story:

```python
134    for i, env in enumerate(self.envs):
135        if mask is not None and not mask[i]:
136            continue
137        _, rewards[i], terminateds[i], truncateds[i], info = env.step(actions[i])
140        _write_env_info(self._stacked_infos, i, info)
```

Single process, single thread, no `SubprocVecEnv`, no `AsyncVectorEnv`, no thread pool. The class
docstring calls itself "a lightweight replacement for `gymnasium.vector.SyncVectorEnv`".

Each `env.step` triggers an unconditional 224×224 render (`wrapper/default.py:484-486` → `:433-434`),
one `mujoco.Renderer` per env — **50 GL contexts per process**.

**[measured]** One 50-env × 50-step call takes **119 s** — CEM mean 132.0 s, LIP mean 110.3 s across
15 evals in `logs/dyna_r1_results/driver_r1_ladder.log`. That is **~48 ms per env-step**, of which
the planner accounts for ~1.8 s *in the entire call* (2 replans; the log shows exactly two
`solve time: 0.9199 / 0.8214` lines). **GPU duty cycle ≈ 1.5%.**

The original campaign driver states the consequence outright:

```
# Eval is CPU-bound (GPU ~0%, per-step 100-env pixel pipeline), so run many jobs concurrently
```

Free instrumentation is already there and unused: `wrapper/default.py:433-435` records
`info['render_time']` per frame, and nothing outside tests ever reads it. **Wire that up first.**

### 4.1b The segfault's root cause: `MUJOCO_EGL_DEVICE_ID` — **CONFIRMED, 2026-07-25**

> **Status: verified from source *and* measured on the 4× H100 loop pod.** This was a hypothesis in
> the first draft; it is now an experimental result. Probe: `rlp/tools/probe/egl_device_probe.py`,
> driver: `main:Dyna/dyna_harness/egl_device_test.sh`.

**Source-level mechanism** (`mujoco/egl/__init__.py`, mujoco 3.10.0):

```python
def create_initialized_egl_device_display():
  all_devices = EGL.eglQueryDevicesEXT()
  selected_device = os.environ.get('MUJOCO_EGL_DEVICE_ID', None)
  if selected_device is None:
    candidates = all_devices                              # <-- ALL of them, fixed order
  else:
    candidates = all_devices[device_idx:device_idx + 1]
  for device in candidates:
    ...
    if initialized == EGL.EGL_TRUE and ...:
      return display                                      # <-- FIRST one that works
```

With the variable unset, **every process walks the same list in the same order and takes the same
first device.** Nothing consults `CUDA_VISIBLE_DEVICES`. `EGL_DISPLAY` is also a module-level global
initialized once per process, so a 50-env process opens **one display and 50 contexts on it** —
3 processes ⇒ 150 contexts, all on one physical GPU.

**Measurement 1 — EGL enumeration ignores `CUDA_VISIBLE_DEVICES`:**

| `CUDA_VISIBLE_DEVICES` | `MUJOCO_EGL_DEVICE_ID` | `n_egl_devices` seen |
|---|---|---|
| 0 | unset | **5** |
| 2 | unset | **5** |
| 2 | 2 | **5** |

Identical no matter what CUDA is restricted to. (5 devices for 4 GPUs — the extra is a
software/other EGL device, which is its own reason to pin the index explicitly.)

**Measurement 2 — where bare contexts land.** 3 processes × 50 EGL contexts, `CUDA_VISIBLE_DEVICES=0/1/2`:

| arm | GPU0 | GPU1 | GPU2 |
|---|---|---|---|
| `MUJOCO_EGL_DEVICE_ID` **unset** | 1096 → **1117** | 1 → **1** | 1 → **1** |
| `MUJOCO_EGL_DEVICE_ID=$gpu` | 1096 | 1 → **6** | 1 → **6** |

**Measurement 3 — under a real 3-way 50-env eval** (`main:Dyna/dyna_harness/egl_3way_eval_test.sh`, CEM, var unset):

```
t0    0: 1096 MiB | 1:   1 | 2:   1        <- baseline
t1    0: 2693 MiB | 1: 576 | 2: 576        <- CUDA contexts appear on 1 and 2 as expected...
t2    0: 5262 MiB | 1: 576 | 2: 576        <- ...but only GPU0 grows
t3    0: 6559 MiB | 1: 576 | 2: 576
t4    0: 8641 MiB | 1: 576 | 2: 576
t5    0:13619 MiB | 1: 576 | 2: 576
```

**CUDA is spread correctly (576 MiB of torch context on GPU1/GPU2, never moving again); all render
allocation lands on GPU 0 and climbs monotonically.**

Two consequences beyond the abort itself:
- **The growth is monotonic**, so the failure is load- and duration-dependent — which explains an
  intermittent abort partway into a batch rather than a clean startup error, and why it worsened
  whenever anything else was resident on GPU 0 (hence the "quiet pod" rule).
- Even when it *doesn't* abort, 3-way EGL funnels all rendering through one GPU's graphics engine,
  so the three "parallel" collectors contend and self-serialize.

**Fix:** set `MUJOCO_EGL_DEVICE_ID=$gpu` next to every `CUDA_VISIBLE_DEVICES=$gpu`. Applied in
`main:Dyna/dyna_harness/collect_r1_fixed.sh`.

**⚠ Not yet closed:** the with-fix/without-fix *exit codes* for a full 3-way eval were never
captured — the run was killed to protect an unrelated experiment sharing the pod. The mechanism is
confirmed; **the "does the fix prevent the abort" test still needs one clean run.**

Historical bracket, consistent with all of the above (the archived campaign
artifacts are no longer part of this repository):
- 2 concurrent EGL evals: fine — `main:Dyna/dyna_harness/capture_videos.sh:5`
- 3 concurrent EGL evals: aborts — `main:Dyna/dyna_harness/collect_r1.sh:56`
- 3 concurrent **osmesa** evals: fine
- 12 concurrent **osmesa** evals, `OMP_NUM_THREADS=18`: **~40× wall-clock**, numerically identical

osmesa "worked" because software rendering never touches a GPU — it sidestepped the bug by
abandoning the accelerator, at the cost of putting rendering on the CPU critical path (§4.1).

### 4.2 `batch_size: 1` turns one batched solve into 50 sequential ones

`configs/inference/benchmark/solver/cem.yaml` sets `batch_size: 1`; the packaged CEM solver loops over it.
With 50 envs that is **50 sequential CEM solves**, each 30 iterations × 300 samples →
**7,500 strictly sequential predictor calls per replan**, plus 100 separate batch-1 ViT forwards.

The fix already exists elsewhere — `rlp/train/trm_pipeline/eval_hard.py:87-94`:

```python
    solver = swm.solver.CEMSolver(..., batch_size=args.solver_batch)  # batch ALL envs together (GPU, not serial)
```

with `core.solver.batch_size=1024`. **The evaluation solver configs never got it.**
Same loop in `gd.py:192`, `mppi.py:149`, `icem.py:179`, `predictive_sampling.py:121`.

LIP's *proposal* correctly batches all envs (`rlp/core/solver/lip.py:231-236`), but
`rlp/core/solver/lip.py:362` reuses the chunk loop for its MPPI tail — and with `n_steps: 0` that
loop body builds an `expanded` dict including `np.repeat(..., 300)` at `:375`, then does nothing
with it.

### 4.3 D2H syncs inside the innermost optimization loop

```python
solver/cem.py:263      final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()   # INSIDE the 30-iter loop
solver/mppi.py:249     (same pattern)
rlp/core/solver/lip.py:396   final_cost = (w * costs.float()).sum(dim=1).cpu().tolist()
solver/gd.py:271       batch_cost_history.append(cost.item())
```

Overwritten every iteration; only the last is used. **30 hard syncs per env × 50 envs = 1,500 wasted
syncs per replan.** `icem.py:324` has the identical line correctly hoisted *outside* the loop.

### 4.4 Per-step CPU work on all envs, including finished ones

```python
policy.py:135-137   torch.stack([self.transform[k](tv_tensors.Image(x)) for x in v])
```

A Python loop calling torchvision v2 **once per image**: 50 envs × {pixels, goal} = **100 transform
invocations per env step**, ~60 MB of fresh float32 per step, on all 50 steps though only 2 plan.
`EnvPool` masks dead envs (`env_pool.py:135`), but `policy.py:375` is unmasked.

More per-step host memcpy on the critical path:
- `world/world.py:570` — `deepcopy(goal_snapshot)` of an **invariant** tensor block, ~7.5 MB/step.
- `world/world.py:574-576` — per-env frame copies accumulated in RAM for the whole run.
- `wrapper/default.py:553-565` — `ResizeGoalWrapper` PIL-resizes the **constant** goal every step.

### 4.5 The goal latent is re-encoded every single replan

`rlp/core/solver/lip.py:218-229` encodes the goal image on every call; the goal is fixed for the episode. CEM
at least caches it within a solve (`lewm.py:132`); LIP does not cache it at all. With
`history_len=1` that is **50% of all encoder FLOPs, thrown away.**

(PLDM is worse: `wm/pldm/pldm.py:78-81` lacks both the `if 'emb' not in info` guard and the
`.detach()` that `lewm.py:77-82` has → the ViT re-runs on **every CEM iteration**.)

### 4.6 Ragged batches defeat compilation

`terminate_at_goal: True` plus `replan_idx` built by list comprehension (`policy.py:393-397`) means
the solve batch size **changes step to step**. That invalidates CUDA-graph capture and forces
`torch.compile` recompiles. Pad to a fixed B and mask, or bucket to a few static sizes.

Related latent bug: `world.py:414` marks done on `terminated | truncated`, but the frozen info slot
carries `info['terminated']` from `wrapper/default.py:305`, which is `False` for a pure truncation.
So a truncated env keeps `dead[i] == False` and keeps triggering solves.

---

## 5. Regime C — Dyna round orchestration

Round: **collect → mix → fine-tune WM → probe-gate → eval ladder.** ~4–6 h (round 2: ~14 h).
Pod is **4× H100, 64 cores**. Almost nothing uses more than one GPU.

### 5.1 Measured breakdown of one ladder

**[measured]** `logs/dyna_r1_results/driver_r1_ladder.log`, 2026-07-24:

| phase | duration | GPUs busy | share |
|---|---|---|---|
| fs1 latent cache | **22 m 02 s** | 1 / 4 | 21.9% |
| fs5 subsample + TD value | 1 m 44 s | 1 / 4 | 1.7% |
| LIP actor ×3 (parallel) | **47 m 01 s** | 3 / 4 | 46.7% |
| 15 evals (sequential) | **29 m 45 s** | 1 / 4 | 29.6% |
| divergence probe | 4 s | 0 | — |
| **total** | **100 m 36 s** | ~48% occupancy | |

The fs1 cache stage is the clearest miss: **22 min on one GPU**, when the 4-way sharded version of
the same job measured **6 m 19 s**. See §9 — the sharding script was broken and has now been fixed.

### 5.2 Collection is ~45% of a round at 1.5% GPU duty

Collection is the eval path with a lance-writer hook spliced in (`rlp/tools/patch/patch_world_record.py`,
gated on `SWM_RECORD_PATH`), so it inherits every cost in §4.

- Round 1: 24 calls → ~50 min sequential (~17 min at PAR=3).
- **Round 2: 180 calls, strictly sequential (`main:Dyna/dyna_harness/r2_driver.sh:73-77`) ≈ 6.3 h — the single
  largest serial block in the pipeline.**

And the 3× parallelism gets latched off permanently by one abort:

```bash
# main:Dyna/dyna_harness/collect_r1.sh:43-64
44    if [ "$PAR" = 3 ]; then
52      wait $P0 || R=1  ...                       # lockstep barrier: batch i+1 waits for the slowest
56        log "parallel batch $i had a failure -> degrading to sequential + retrying"
57        PAR=1                                    # <-- LATCHED, never reset
59        collect_call "$a" 0 "$i"                 # <-- retry pins GPU 0
64      collect_call "$a" 0 "$i"                   # <-- sequential path pins GPU 0 too
```

Fixed in `main:Dyna/dyna_harness/collect_r1_fixed.sh`: EGL device pinning, bounded per-job retry on the worker's
own GPU, and one independent worker per actor instead of the lockstep batch barrier (one worker per
actor is *required* — concurrent writers must not share a lance).

### 5.3 Fine-tune uses 2 of 4 GPUs and no DDP

`trainer.devices=1` in every arm; two arms concurrently on 2 GPUs, ~5 h wall (**[measured]** 3.4 it/s
single-GPU ≈ 56 min/epoch). A working 4-GPU recipe exists — **[measured]** 5.9 it/s ≈ 33 min/epoch,
a **1.7×** return — but `grep -rn "NCCL"` over the harness returns **zero hits**. The config is
fragile: NVLS must be **off** (`NCCL_NVLS_ENABLE=0`) *and* P2P must stay **on** (a leftover
`NCCL_P2P_DISABLE=1` forced sync through SHM and hung the run ~70 min in, exit 134).

Watch the **cgroup pid quota** when raising rank count: `pids.max = 16896`, threads count, and
uncapped OMP × 77-thread dataloader workers has already exhausted it.

### 5.4 Dataset mixing: ~65 GB of single-threaded rewrite, arms back-to-back

`rlp/tools/data/build_dyna_mix.py` is one generator feeding one `lance.write_dataset`. JPEG passes
through undecoded (good), but:

- arm 5050 = 2,010,000 expert + **25×** 80,746 on-policy = 4,028,650 rows ≈ **40 GB written**
- arm 8020 = 2,010,000 + **6×** = 2,494,476 rows ≈ **25 GB written**
- both built **sequentially** before either GPU starts
- `:36-43` — a **per-row Python loop** to remap `episode_idx`, repeated K=25 times
  (`build_arm_dataset.py:57-58` already does this with vectorized `pyarrow.compute`)
- `:92-96` — a full re-read of the 4 M-row output to assert monotonicity

The 25× duplication exists **only** because the loader has no sample weights (stated in the
docstring). A weighted sampler deletes ~20 GB of writes per round.

### 5.5 Barriers: real vs. scripted

**Real:** collect→mix, mix→fine-tune, fine-tune→gate, TD→LIP actor.

**Scripted only** — all independent, all pinned to GPU 0: the 15 ladder evals
(`main:Dyna/dyna_harness/dyna_r1_ladder.sh:102-118`, comment `# 4. evals — sequential (quiet pod)`), the 2 gate
arms (`main:Dyna/dyna_harness/dyna_r1_gate.sh:72-73`, `# All evals SEQUENTIAL (quiet-pod rule).`), the 2 mix
builds, the 6 P1 validation evals.

An N-way worker **already exists** (`main:Dyna/dyna_harness/run_evals_b.sh:81-101`), and every eval driver is
idempotent per row, so aggressive re-running is safe.

Most damning: `main:Dyna/dyna_harness/r2_driver.sh:3` forbids overlapping eval-path work with training.
**Collection is ~98% CPU and fine-tune is ~100% GPU — the ideal overlap pair, and the harness bans
it.** The ban is a workaround for §4.1b, not a fundamental constraint; an earlier version
(`main:Dyna/dyna_harness/dyna_stage1.sh:118-124`) *did* overlap them.

### 5.6 Polling gates

`main:Dyna/dyna_harness/dyna_r1_finetune.sh:26` — `while ! grep -q "COLLECT_R1_DONE" ...; do sleep 300; done`:
**up to 5 minutes of pure idle** at the collection→fine-tune handoff. Cross-machine "IPC" is
grep-a-logfile-for-a-string; `main:Dyna/dyna_harness/r2_driver.sh:89` ends with a **human-in-the-loop barrier**.

### 5.7 Cache building has no prefetch

`rlp/data/latent_cache.py:98-133` (`encode_dataset`) is a synchronous read → jpeg-decode →
GPU-encode → `.cpu()` loop with no DataLoader, no workers, no double buffering.

### 5.8 Ops trap worth preserving

Never write pod scripts via `ssh 'cat > f << EOF'` when the body contains patterns you later
`pgrep -f` — the lingering `bash -c cat…` writer keeps the whole body in its cmdline, so a
`while pgrep -f X` wait self-matches forever. `scp` the scripts instead. *(Corollary learned the
hard way during this audit: `pkill -f <pattern>` run over SSH also matches the `bash -c` carrying
that pattern, and kills your own session. Kill by PID.)*

---

## 6. What is simply switched off

| | LIP trainer | LIP solver | eval default |
|---|---|---|---|
| bf16 / autocast | absent | absent | `cfg.get('bf16', False)` — **off**, not set in `cube.yaml` |
| `torch.compile` | absent | absent | `cfg.get('compile', False)` — **off**, not set in `cube.yaml` |
| TF32 / `set_float32_matmul_precision` | absent | absent | absent |
| CUDA graphs | absent | absent | absent |
| `cudnn.benchmark` | absent | absent | absent |
| `pin_memory` / `non_blocking` | absent | absent | absent |

The production path runs **fp32 without even TF32**, uncompiled — while the machinery exists and is
used elsewhere in the same repo:

- `src/rlp/inference/benchmark/world_model.py` — bf16 cast, `torch.compile` on encoder +
  predictor, autocast. All gated behind flags `cube.yaml` never sets.
- `rlp/core/world_model/dinowm/tokens.py:165,172` — `torch.compile` + bf16 autocast + grad
  checkpointing.
- `wm/lewm/lewm.py`, `wm/lewm/module.py` — **zero** occurrences. This is the path used for every
  cube result and the entire Dyna campaign.
- The WM was **trained** at `precision: bf16-mixed`, so bf16 inference is in-distribution.

**LIP's inner loop is an unusually good CUDA-graph target**: K=8 and H=5 are compile-time constants,
shapes are fully static, no data-dependent control flow — *provided* the batch dim is pinned (§4.6).

---

## 7. Ranked fix list

Ordered by (wall-clock won) ÷ (effort). "Free" = no numerics change.

| # | fix | where | payoff | risk |
|---|---|---|---|---|
| 1 | **Set `MUJOCO_EGL_DEVICE_ID=$gpu`** next to `CUDA_VISIBLE_DEVICES`; retest 3-way collection | `main:Dyna/dyna_harness/collect_r1.sh:27`, all cube Dyna scripts | unblocks 3× on **the** longest stage (~45% of a round) | **confirmed root cause**; historical fix in `main:Dyna/dyna_harness/collect_r1_fixed.sh` |
| 2 | Un-latch `PAR`, stop pinning retries to GPU 0, drop the batch barrier | `main:Dyna/dyna_harness/collect_r1.sh:57,59,64` | one abort currently costs the whole run's parallelism | low; historical fix in `main:Dyna/dyna_harness/collect_r1_fixed.sh` |
| 3 | ~~Reuse `traj.detach()` instead of re-rolling~~ **DONE 2026-08-12** — solver default, bit-exact (§0.5) | `rlp/core/solver/lip.py` | **−47% of LIP forward rollout** | **free**, exact |
| 4 | `batch_size: 1` → `>= num_envs` for CEM/GD/MPPI/iCEM | `config/solver/cem.yaml:3` | 50× fewer sequential solves per replan | **free**, one line, proven at `rlp/train/trm_pipeline/eval_hard.py:94` |
| 5 | Wire in 4-way sharded cache build | `main:Dyna/dyna_harness/dyna_r1_ladder.sh:55-64` | **[measured]** 22 m → 6 m 19 s, 22% of the ladder | script was broken; **fixed + validated**, see §9 |
| 6 | Generalize the historical N-way eval worker; run gate arms + mix builds concurrently | `main:Dyna/dyna_harness/run_evals_b.sh:81-101` | ~30 m → ~8 m ladder; 3 idle GPUs recovered | gated on #1 |
| 7 | Hoist `.cpu()/.item()` out of inner loops | `cem.py:263`, `mppi.py:249`, `rlp/core/solver/lip.py:396`, `gd.py:271`, `train_lip_ac.py:500,625` | −1,500 syncs/replan; −24,000/training run | **free**; `icem.py:324` shows the correct form |
| 8 | Cache the goal latent per episode | `rlp/core/solver/lip.py:218-229`; `pldm.py:78-81` | −50% encoder FLOPs (LIP) | **free** |
| 9 | `pred_proj` on last token only; batch the action encoder; ring-buffer the window; Conv1d(k=1)→Linear | `lewm.py:50-51`, `rlp/core/rollout.py:37-39`, `module.py:200-209` | −2/3 of `pred_proj`; −160 kernels/actor step | **free** |
| 10 | Turn on bf16 + `torch.compile` + TF32 | `cube.yaml`, `set_float32_matmul_precision('high')` | large in a launch-bound regime | low; needs #12 for compile to stick |
| 11 | Delete dead work | `train_lip_ac.py:412`; `rlp/core/solver/lip.py:362-375`; `build_dyna_mix.py:92-96` | small but pure | **free** |
| 12 | Pin the solve batch (pad + mask) | `policy.py:393-397` | prerequisite for CUDA graphs | low |
| 13 | Export the documented NCCL env, 2 GPUs/arm | `main:Dyna/dyna_harness/dyna_r1_finetune.sh:54` | **[measured]** 1.7× on ~36% of a round | medium; NVLS off + P2P on; watch the pid quota |
| 14 | Parallelize env stepping + rendering in-process | `world/env_pool.py:134` | attacks the 48 ms/env-step directly | medium; needs #1 first |
| 15 | Weighted sampler instead of 25× row duplication | `rlp/tools/data/build_dyna_mix.py:66-68` | deletes ~20 GB of writes per round | medium |
| 16 | Activation-checkpoint the rollout; raise B | `train_lip_ac.py:612`; pattern at `rlp/core/world_model/dinowm/tokens.py:167-175` | frees ~2 GB → larger batch ≈ free throughput | low |
| 17 | ~~CUDA-graph the K×H unroll~~ **DONE (inference) 2026-08-12** — opt-in `core.solver.graphed`, 272.6→29.3 ms/decision, 5.96e-8 deviation (§0.5); training unroll still eager | `rlp/core/solver/graphed.py` | **[measured]** 9.3× per decision | landed; per-batch capture handles #12 |
| 18 | Vectorize samplers; `argsort`-based `episodes()`; DataLoader + pinned memory; vectorize the remap | `rlp/core/value/samplers.py`, `rlp/data/latent_cache.py:53`, `rlp/tools/data/build_dyna_mix.py:36-43` | ~37M Python RNG calls → 0 | **free** |
| 19 | Drop the collection→fine-tune poll from `sleep 300` to `sleep 15` | `main:Dyna/dyna_harness/dyna_r1_finetune.sh:26` | up to 5 min/round | **free** |
| 20 | RoPE / time-invariant positions → KV-cache the window | `wm/lewm/module.py:269,291` | ~3× rollout transformer FLOPs | **high — changes numerics, needs a retrain** |

**Do first:** #1 and #2 are written and ready. Then the free, correctness-preserving batch: #3, #4,
#7, #8, #9, #11, #18. Instrument before committing to #14 or #17 — `render_time` at
`wrapper/default.py:433` is already being recorded and thrown away.

**One structural note.** Everything in the "artificially serialized" column traces back to the same
root: the eval-path concurrency abort (§4.1b). It caused the quiet-pod rule, which caused the
sequential ladders, which caused the ban on overlapping CPU-bound collection with GPU-bound
training. Fix the renderer concurrency and a large amount of scheduling freedom is restored at once.

---

## 8. What is *not* a bottleneck — don't waste time here

- **The actor network.** 455k params, ~0.04% of WM FLOPs. The historical
  measurement found that swapping the MLP actor for a transformer moved step
  time only 2.6% (0.77 vs 0.79 s/step).
- **The critic / teacher / value head.** 147k params, ~1.2 GFLOP/step at B=1024.
- **The EMA update** (`train_lip_ac.py:497-499`) — 12 tiny kernels.
- **top-k / elite selection** (`cem.py:225-245`) — already GPU-side on a `(1,300)` tensor.
- **`expanded_infos` tensor expansion** (`cem.py:168-176`) — stride-0 `.expand()`, never materialized.

---

## 9. Open items and corrections

1. **The 3-way abort fix is not yet proven end-to-end.** §4.1b confirms the *mechanism*
   (`MUJOCO_EGL_DEVICE_ID` controls placement; `CUDA_VISIBLE_DEVICES` does not), but the
   with-fix/without-fix *exit codes* for a full 3-way eval were never captured — the run was killed
   to protect an unrelated experiment sharing the pod. **Needs one clean run of
   restoring `main:Dyna/dyna_harness/egl_3way_eval_test.sh` or an equivalent
   driver on a quiet pod.**

2. **The launch-count estimate (~25–35k/decision) is static counting, not Nsight.** It is the number
   here that most deserves direct measurement before anyone commits to fix #17.

3. ~~`rlp/tools/data/cache_lance_shard.py:50` looks wrong~~ — **CONFIRMED BROKEN, FIXED, AND VALIDATED.**
   `get_row_data(self, i): return full.get_row_data(s + i)` is called from
   `rlp/data/latent_cache.py:125` with `i` as a **list**, and `int + list` raises
   `TypeError: unsupported operand type(s) for +: 'int' and 'list'` — on the *first batch*, for
   *every* shard including shard 0. This script could never have run. Corroborated on the pod:
   `/workspace/caches/` contained only whole-dataset `fs1`/`fs5` caches, **no shard files**, and no
   log referenced it. The measured 4-way cache time quoted elsewhere came from the older
   `cache_cube_full.py` (`episode_start` / `episode_end`), not this one.

   This matters because fix #5 — the 22 min → 6 min cache win, 22% of a ladder — *is delivered by
   this script*. It was staged as a "round-2+ speed lever (scripts ready)" but never exercised, so
   the bug sat undetected.

   **Fixed** element-wise in `rlp/tools/data/cache_lance_shard.py` and **validated on the pod**: a
   10-episode shard (rows 402000–404010) encoded on GPU3 and compared against the reference fs1
   cache — `episode_idx` and `step_idx` exact, latents matching to **1.7e-6** (ordinary fp32
   batch-tiling nondeterminism, not a logic error).

4. **Two rollout code paths differ.** CEM goes through `LeWM.rollout` (`lewm.py:96`), LIP uses
   `rollout_traj` (`rlp/core/rollout.py:30`) with a zero-filled 2-block action history
   (`rlp/core/solver/lip.py:236`). Both land
   on 5 predict calls here, but the equivalence is incidental to `history_len=1`. Anyone changing
   the rollout must change both.

---

*Method note: FLOP figures are static counts from `assets/core/world_model/lewm_cube/config.json`, not profiler output. They are
consistent with the measured 0.56 s @ B=1 / 0.58 s @ B=50 and 0.78 s/training-step to within the
precision the argument needs.*
