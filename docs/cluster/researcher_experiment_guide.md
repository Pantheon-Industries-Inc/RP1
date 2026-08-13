# Pantheon GPU Cluster — Experiment Guide

<!--
This file lives in the modal-skypilot repo, and that is deliberate.

It used to live in a Google Doc and as a Slack attachment. Both drifted: on 2026-08-09 the
Doc copy was still telling agents to set `disk_size` in every template, which is the single
largest cause of failed launches on this cluster. The numbers below are the same numbers the
shim holds as constants in `src/modal_skypilot/mapping.py`, and `tests/test_docs.py` asserts
that they still match. If a test in that file fails, one of the two is stale — find out
which before editing either.

Date every NEW measurement you add here. That is the repo's rule for cluster facts, not a
stylistic choice: a number with no date cannot be told apart from a number that was true
once. Several figures arrived with this document undated — the 47 preemptions behind the
SIGTERM rule, the 4 vCPU / 16 GB an unspecified job is given, the three-preemption resume
— and they are left as they were found rather than stamped with a date nobody measured
them on.
-->

## FOR AGENTS: Read this section first

Complete reference for submitting experiments to Pantheon's 128× H200 cluster via SkyPilot.
Install the SkyPilot Agent Skill: `Fetch and follow https://github.com/skypilot-org/skypilot/blob/HEAD/agent/INSTALL.md`

Constraints:
- GPU type is `H200` — remap if the user says H100/A100
- Default priority is `p3` — never set higher without the user asking. `p4` (10) sits *below* it, for work that should yield to everything else
- Use `sky jobs launch` for all unattended work, never `sky launch`
- Do NOT hardcode `infra:` — SkyPilot selects automatically
- Always set `cpus` and `memory` proportional to GPU count: **20 CPU + 230 GB per GPU**. This is the starting point, not a floor — see below
- **Never request more than the table without a measurement.** A node is 175.8 CPU / 1928 GiB across 8 GPUs (≈22 CPU and 241 GiB per GPU, ≈1 CPU per 11 GiB). Asking above that share on a sub-node job leaves GPUs on your node that nobody — including you — can use, and skewed CPU:memory strands whichever resource you under-ask for. Both are invisible to you and surface as somebody else's job failing to schedule
- **Request what the job uses, not what looks safe.** After a first run, size from *anonymous* memory in `/sys/fs/cgroup/memory.stat` plus headroom — not from `memory.current`, which counts reclaimable page cache. On 2026-08-09 two jobs requested 1630 GiB and used 54 and 10 GiB of real memory. Do not reduce below the table without that measurement: under-requesting memory OOMs the job and under-requesting CPU starves the dataloader
- Checkpoints go to `/checkpoints/{PANTHEON_USER}/{EXPERIMENT_TAG}/` (shared volume, survives eviction)
- **Never write output to `/workspace`, `~`, `/root`, or a relative path.** Those land in the container's writable layer, which is shared node disk with no quota — one job wrote 692 GiB there on 2026-08-09 and pushed a node into disk pressure, excluding 8 GPUs from scheduling for everyone. Write to `/checkpoints/...`, `/mnt/raid0`, or `/tmp`
- Bulk/temp data goes to node-local scratch (`/tmp`, or `/mnt/raid0` for anything large) — NEVER set `disk_size`, never put an `ephemeral-storage` request in `pod_config`, and never put bulk data on `/checkpoints`
- Platform: https://pantheon.svc.skypilot.co

---

---

# Part I — What every job must do

*(This part is Sambhav's; it's about how to write a job. Part II is the cluster
mechanics — how to size, submit and store.)*

## Writing code for a job

Jobs should be written as Modal jobs, so they can execute on either our infra or
Modal. Modal jobs run on our infra via the
[`modal-skypilot`](https://github.com/Pantheon-Industries-Inc/modal-skypilot)
package — a shim that converts Modal code into the equivalent SkyPilot YAML and
launches it. Expect SkyPilot infra to be the main target.

## Preemption safety

**Expect all jobs to be preemptible.** A job must lose no more than **5 minutes
of compute** if preempted. For training, that means the dataloader, optimizer and
model are all checkpointed to non-ephemeral disk. For data processing, workers
must register completed work atomically and resume from arbitrary states.

This is what allows aggressive scheduling, including defragmenting jobs that take
less than a node.

Jobs must also be robust to hardware failure — especially multi-node. A job must
not corrupt itself or its past results when anything other than non-ephemeral
storage fails.

⚠️ **Do not write a SIGTERM handler and rely on it.** Measured across 47 real
preemptions on this cluster: no signal reaches the process, it ends at SIGKILL.
Durability has to come from checkpoint frequency, not from a graceful-shutdown
path. See §5.

## MFU

Jobs should attain high MFU — at worst 10%, ideally 40%+. Make CPU work async
from GPU work.

- CPU-only data processing should run on CPU instances. **Note:** there is no CPU
  nodepool in this Kubernetes cluster today — every node is an 8×H200 machine —
  so a CPU-only job here consumes CPU on a GPU node. Standalone `c1a.*` VMs exist
  outside the cluster for genuinely CPU-only work.
- Dataloaders should prefetch and never block the GPU. If dataloading blocks the
  GPU, resize the model or the GPU count until it doesn't. `/mnt/raid0` (§5) is
  16.6× faster than the shared volume on small random reads and is usually the
  right fix.

## Performance and durability via queueing

Use producer–consumer queues:

- **Performance** — things that needn't wait for each other shouldn't. Checkpoint
  writes should be async.
- **Durability** — queues checkpoint progress durably.
- **Consistency** — always append data before updating metadata, so preemption
  cannot leave a ghost shard that never gets processed.

**Do not use Python threading for parallelism — only multiprocessing.** Keep it
simple and deadlock-free; prefer vendored queue abstractions.

## W&B

W&B logging governs the quality of information an experiment yields.

On preemption, or a deliberate pause and resume, **reuse the same W&B run** — not
a fresh one. See the `WANDB_RESUME` / `WANDB_RUN_ID` pattern in §5.

Be precise with naming. No vague run names like "training", and no extraneous
hyperparameters that aren't being tuned. Run and project names should convey what
the project is and what the run is testing.

## Training telemetry contract (`training-v1`)

A custom SkyPilot dashboard expects telemetry from every running job, and
complains loudly without it.

Every active GPU training job must:

- Publish a W&B run URL discoverable by SkyPilot.
- Refresh these W&B summary fields every ~10s:
  - `pantheon_telemetry/contract_version: training-v1`
  - `pantheon_telemetry/heartbeat_at` — Unix timestamp
  - `pantheon_telemetry/total_steps` — positive integer
  - `pantheon_telemetry/active_step` — current step
  - `pantheon_telemetry/step_time_s` — latest step duration
  - `pantheon_telemetry/effective_mfu` — fraction in `[0, 1]`
  - `pantheon_telemetry/wandb_url` — canonical run URL
- Log scalar history under the first metric matching `train/loss*`; log
  `val/loss` when validation begins.
- Report **only from global rank 0**. Heartbeats older than 30s are violations.

Progress is `active_step / total_steps`. Missing links, missing fields, missing
train loss, invalid values or stale heartbeats all produce loud dashboard errors.

Non-training jobs must explicitly opt out with a reason:

```yaml
resources:
  labels:
    telemetry-contract: opt-out
    telemetry-opt-out-reason: embedding-extraction
```

**MFU must be end-to-end "effective MFU", not compute-only.** Over all steps, the
sum of step times in the denominator must be nearly equal to the length of the
run. A compute-only MFU is not an accurate reflection of how well we are using
the compute. A separate compute-only MFU may be logged to assess the ceiling once
dataloading stalls are eliminated, but must not go to telemetry.

## Data processing jobs

Data processing jobs should be producer–consumer queues: consumers take many
small units of work, process, write out, then atomically update metadata. An
orchestrator assigns tasks dynamically so no worker waits in a long tail.

## InfiniBand

InfiniBand setup is involved — **use `modal-skypilot` and let it emit the
config**. Validated 2026-08-09 at 437.5 GB/s busbw with GDRDMA on a converted
2-node job. Do not hand-copy the block; see §6.

---

# Part II — Cluster reference

## 1. Setup (~5 minutes)

```bash
pip install "skypilot[kubernetes]"
export SKYPILOT_API_SERVER_ENDPOINT=https://pantheon.svc.skypilot.co
sky api login
echo 'export SKYPILOT_API_SERVER_ENDPOINT=https://pantheon.svc.skypilot.co' >> ~/.zshrc
```

---

## 2. Cluster

| Detail | Value |
|---|---|
| GPUs | 128× NVIDIA H200 (16 nodes × 8 GPUs) |
| Accelerator string | `H200` |
| Per-node resources | 175.8 CPU, 1928 GiB RAM, ~14 TiB of NVMe (see below) |
| Per-GPU share | 21.9 CPU, 241 GiB — the divisor everything in §4 is about |
| Platform + Dashboard | https://pantheon.svc.skypilot.co |

The NVMe figure is **not** an `ephemeral-storage` quota and must never be requested as one:
~1.8 TiB of it is the container filesystem you already have at `/tmp`, and 12.2 TiB is the
opt-in `/mnt/raid0` array (§5). What Kubernetes advertises as `ephemeral-storage` is a
different, much smaller device — the node's boot disk, of which **111.5 GiB is
allocatable**. See §5.

---

## 3. Priority

| Name | Value | Use | Default |
|---|---|---|---|
| p0 | 1000 | Large PFM runs only (Mo/Sambhav) | |
| p1 | 80 | Time-critical ablations ≤32 GPU | |
| p2 | 60 | Standard ablations & sweeps | |
| p3 | 20 | Single-GPU experiments, sweeps | ✓ |
| p4 | 10 | Below the default — backfill and opportunistic work that should yield to everything else | |

```bash
sky jobs launch experiment.yaml                        # p3 (default)
sky jobs launch experiment.yaml --priority p1          # higher priority
```

**Always use the tier names, never bare numbers.** The CLI silently accepts any
integer in ±1000: a bare `--priority 2` lands *below* p3 (=20) — last in the
admission line and first evicted — and a hand-rolled `900` silently outranks
every p1 on the cluster. Both happened. Names only.

---

## 4. CPU / RAM sizing

Kubernetes gives unspecified jobs almost nothing (observed: 4 vCPU / 16 GB on a
1-GPU job) — enough to starve any real dataloader. Size to GPU count:

| GPUs | cpus | memory (GB) |
|---|---|---|
| 1 | 20 | 230 |
| 2 | 40 | 460 |
| 4 | 80 | 920 |
| 8 (full node) | 160 | 1840 |

Rule: `cpus = 20 × N`, `memory = 230 × N`. The templates below already carry
these. Going higher than the table risks an unschedulable pod (node allocatable
is ~176 CPU / ~1928 GiB).

### Keep CPU and memory in the node's proportion

A node is **175.8 CPU : 1928 GiB — roughly 1 CPU per 11 GiB.** Request far from
that ratio and you strand the *other* resource on your node for everyone else.

Measured 2026-08-09: jobs running 64 CPU / 1630 GiB (1:26) left nodes with 46–74
CPU free and 5–22 GiB of memory, while guide-shaped jobs left the mirror image —
212 GiB free with 14 CPU. A 1-GPU job asking 24 CPU + 168 GiB found **zero of 16
nodes** could take it, with nine GPUs sitting idle. It looks like a scheduler
problem and is not.

The `20 × N` / `230 × N` table is already in proportion. Stay on it unless you
have measured a reason not to.

### Anything you write outside a mounted volume fills a shared node disk

Your container's filesystem is **not yours** — it is a writable layer on the
node's 1.8 TiB NVMe, shared with every other pod on that machine, with no quota.
That includes:

- **`/workspace`** — the NGC PyTorch image's default `WORKDIR`, so *plain
  relative paths land here*
- `/root`, `~`, `/opt`, and anywhere else not listed under `volumes:`

When that disk passes 85% used, kubelet declares disk pressure and **the whole
node stops accepting work** — every other job on it is affected, and the node is
excluded from scheduling until it recovers.

Measured 2026-08-09: one job held **692.5 GiB in `/workspace`**, 93% of its
node's container-filesystem usage. Another node crossed the threshold the same
evening and had **8 GPUs excluded** from scheduling while it drained.

Write instead to:

| For | Path | Notes |
|---|---|---|
| Anything you need after the job | `/checkpoints/$PANTHEON_USER/$EXPERIMENT_TAG/` | durable, replicated |
| Large scratch, dataset cache | `/mnt/raid0` | 12.2 TiB/node, free, fast — see §5 |
| Small temp files | `/tmp` | node NVMe, shared and unpoliced — keep it small |

Check your own job with `du -sh /workspace` before you scale it up.

### Why going over the table costs other people GPUs

A node has **1928 GiB across 8 GPUs = 241 GiB per GPU.** That is the physical
share. Ask for more than 241 GiB per GPU and the excess is taken from the GPUs
you did *not* request — they stay physically idle and **no other job can ever
land on them**, because there is no memory left to run with.

Measured on 2026-08-09: two 4-GPU jobs requesting `memory: 1750` (1630 GiB,
407 GiB/GPU) each stranded 4 GPUs on their node. Across six nodes, **11 of 16
free GPUs were unusable by any 1-GPU job** while 44 workloads sat pending. CPU
was never the constraint — every node had 55+ cores spare. It looked like a
scheduler problem and was not.

It got worse before it got better. Measured **2026-08-10**: twelve sub-node jobs were
asking **1676 GiB for 4 GPUs** (419 GiB/GPU), one of them 1676 GiB for **3** (559
GiB/GPU). Result: **24 of 128 GPUs idle**, with about 1,088 GiB of free memory spread
across the nodes holding them — enough for roughly five of the twenty-four. **Zero of
sixteen nodes** could have fitted another 4-GPU job at this guide's `memory: 920`.

Five of those twelve were checked from the inside, `/sys/fs/cgroup/memory.stat`: **11,
11, 0, 58 and 9 GiB anon**, with `memory.events` `max = 0` on every one. None had ever
come near its limit. All twelve were hand-written SkyPilot YAML rather than converted
Modal code, which is why `modal-skypilot convert` never got a chance to say so — and why
`modal-skypilot submit` now reads `resources:` out of a hand-written document and warns
with the same arithmetic.

If your memory-per-GPU needs to exceed ~241 GiB, the honest options are to take
a **full node** (8 GPUs, 1840) or to use **fewer GPUs with the same memory** —
not to reserve a neighbour's share.

### Do you actually need that much? Usually not — check before asking

Linux counts **page cache** in a container's memory usage, and page cache is
**reclaimable** — the kernel drops it under pressure and your job keeps running.
Only *anonymous* memory (real allocations) is unreclaimable. A job showing
700 GiB of "usage" may be holding 50 GiB of real memory and 650 GiB of cached
file reads it never needed reserved.

Check your own running job:

```bash
kubectl exec <your-pod> -- sh -c '
  echo "current: $(( $(cat /sys/fs/cgroup/memory.current) / 1073741824 )) GiB"
  awk "/^anon /{printf \"anon (real, unreclaimable): %d GiB\n\", \$2/1073741824}" /sys/fs/cgroup/memory.stat
  awk "/^file /{printf \"file cache (reclaimable): %d GiB\n\", \$2/1073741824}" /sys/fs/cgroup/memory.stat
  echo "--- has it ever hit the ceiling? max>0 means yes ---"
  cat /sys/fs/cgroup/memory.events'
```

Size `memory` to **anon plus headroom**, not to `current`. If `max` in
`memory.events` is `0`, your job has never come close to its limit and the
request is too high.

Real examples from 2026-08-09, same cluster:

| Job | GPUs | requested | anon (real) | file cache | verdict |
|---|---|---|---|---|---|
| A | 4 | 1630 GiB | **54 GiB** | 630 GiB | 30× over |
| B | 5 | 1630 GiB | **10 GiB** | 584 GiB | 163× over |
| C | 4 | 1118 GiB | **643 GiB** | 299 GiB | honest, leave alone |

Job C is a legitimate exception — 643 GiB of real memory with a sensible margin.
The rule is not "always request less," it is **request what you use.**

---

## 5. Storage & checkpoints

Three tiers — use the right one:

| Tier | What | Survives eviction | Use for |
|---|---|---|---|
| `/checkpoints` volume | 10 TiB shared filesystem, all nodes | ✅ | checkpoints and small durable state ONLY |
| Node scratch (`/tmp`) | ~1.5 TiB free local NVMe, already mounted | ❌ | datasets, temp files, anything bulk |
| `s3://encodings` bucket | Crusoe object storage | ✅ | bulk data in/out of the cluster |

### Choosing: local vs volume vs bucket

The underlying difference: a **volume is a filesystem**, a **bucket is a
warehouse with an API**. Everything else follows. Ask three questions in order:

**1. Does the data only matter while this job runs?** → node-local NVMe: just write to `/tmp`.
Fastest option on the cluster, costs nothing, and eviction wiping it is fine
because the job was its whole lifetime. Staged datasets, decompression scratch,
intermediate files.

**2. Must the job find it again after eviction, at a normal file path?** →
volume (`/checkpoints`). It's a real filesystem: `torch.save` works directly,
renames are atomic, no credentials, and the eviction-recovery contract (job
relaunches → `load_latest_checkpoint()` → resumes) depends on the files being
at the same path on whatever node the job lands on. Checkpointing to a bucket
instead is actively worse: SIGTERM mid-upload leaves a truncated object and a
corrupt resume.

**3. Does it cross the cluster boundary, or is it bulk?** → bucket
(`s3://encodings`). The bucket is the *only* door in or out of the cluster —
hades staging, embeddings from external machines, artifacts you keep after the
job. It's also the only place bulk belongs: effectively unlimited (256 TiB org
quota, billed per byte stored), whereas the volume is a fixed 10 TiB **shared
by everyone** and billed on provisioned size — one 5 TiB dataset parked there
puts every job's checkpoint saves on the cluster at ENOSPC risk simultaneously.

Rule of thumb: **local for during-the-job, volume for across-evictions, bucket
for across-the-boundary and anything big.** A typical training job uses all
three in one YAML — sync a slice from the bucket to `/tmp` scratch, train
reading local, checkpoint to the volume.

(The friction difference is deliberate: the volume is zero-ceremony because
checkpoints are touched constantly; the bucket needs the three env exports
because its contents are touched once per run.)

One hard rule: **checkpoints go on `/checkpoints`, bulk data does not.** The
volume is shared and capped — filling it breaks checkpoint saves for every job
on the cluster at once.

**Creating volumes** (if you make your own): size must be ≥ 1 TiB in whole-Ti
steps — Crusoe's provisioner rejects anything smaller (binary TiB: 1000Gi
fails, 1024Gi works), and an invalid size wedges permanently NOT_READY instead
of erroring. And **never enable Auto Mount**: it injects the volume into every
new job on the cluster, so one broken volume blocks all scheduling (this took
the queue down on Aug 5). Mount explicitly in your YAML instead.

### Shared checkpoint volume

Mount it in your YAML:
```yaml
volumes:
  /checkpoints: checkpoints
```

Path convention:
```
/checkpoints/{your-email}/{experiment-tag}/step_000500.pt
```

Example: `/checkpoints/mo@pantheon.inc/vit-lr3e4-aug04/step_001000.pt`

- **PANTHEON_USER** = your @pantheon.inc email (no collision with other researchers)
- **EXPERIMENT_TAG** = unique per experiment. Reusing a tag deliberately = resuming
  that run's checkpoints. Reusing one accidentally = corrupting them. When in
  doubt, mint a new tag and date-stamp it.

⚠️ **`chmod 777` every directory you create on `/checkpoints`, immediately.**
Container images differ in default user — `nvcr.io/nvidia/pytorch` runs as
**root**, SkyPilot's default image runs as **`sky` (uid 1000)**. Whichever image
first creates `/checkpoints/<you>/` owns it, and a later job under the other
image gets `mkdir: Permission denied` *inside its own directory*. The volume
root is already 777; keep your subdirectories the same:

```bash
mkdir -p "$D" && chmod 777 "$D"
```

### Creating your own volume

Only if `/checkpoints` doesn't fit — it's RWX, 10 Ti, and mostly empty.

```yaml
name: my-volume
type: k8s-pvc
infra: kubernetes
size: 1Ti                              # 1 TiB MINIMUM, whole-Ti steps only
config:
  namespace: default
  storage_class_name: crusoe-sharedfs  # RWX-capable class
  access_mode: ReadWriteMany           # REQUIRED for multi-node
```

```bash
sky volumes apply my-volume.yaml
sky volumes ls          # confirm READY before submitting anything
```

Three ways this bites if you skip the explicit lines:

- **Omit `access_mode`** → you get `ReadWriteOnce`, and any multi-node task is
  rejected at submit: `Volume X with access mode ReadWriteOnce is not supported
  for multi-node tasks`. **Access mode is fixed at creation** — the only fix is
  a new volume.
- **Omit `storage_class_name`** → you may land on an RWO-only class.
- **Size below 1 TiB, or not a whole-Ti step** → the PVC goes permanently
  `NOT_READY` **silently**, with no error at creation. Every job that mounts it
  then holds GPU quota it can never use. This took the queue down on Aug 5.

Never enable **Auto Mount** — it injects the volume into every new pod
cluster-wide.

### Local scratch (node NVMe)

Nothing to request — **the container filesystem already is the node's NVMe.**
Write to `/tmp` (or anywhere on `/`) and you get roughly **1.5 TiB free**:

```
/dev/nvme0n1   1.8T  170G  1.5T  11%  /
```

**Do not set `disk_size`, and never request `ephemeral-storage` in
`pod_config` either.** `disk_size` maps to Kubernetes *ephemeral-storage*, of which
the node has very little — and the two numbers `kubectl describe node` prints are not the
same number:

| | Value | What it is |
|---|---|---|
| `capacity` | `129886128Ki` = **123.9 GiB** | the boot disk `/dev/vda1` — not the NVMe |
| `allocatable` | `119703055367` = **111.5 GiB** | capacity × 0.9, after the default 10% `nodefs.available` eviction reserve |

**Your request is scheduled against `allocatable`, so 111.5 GiB is the ceiling that
matters** — a `disk_size` between the two looks like it fits and does not. That is not a
reading of the docs, it is what the cluster did: `disk_size: 100` launches and `128` fails,
and 128 GB is under capacity (133.0 decimal GB) and over allocatable (119.7). If capacity
were the constraint, 128 would have worked.

Anything above ~100 is rejected at submit with `FAILED_PRECHECKS` (this
silently cost the team 20+ failed launches over two days), and smaller values reserve
from the scarcest per-node quota there is: about **12.5 GB per GPU**, so a request that
does schedule strands the H200s it did not ask for exactly as an oversized `memory:`
does (§4). Crusoe confirmed on 2026-08-06 that this is expected CMK behaviour with no
setting to repoint it at the NVMe.

`modal-skypilot` refuses Modal's `ephemeral_disk=` at **every** size for this reason, and
`modal-skypilot submit` scans a hand-written YAML for both spellings and warns.

⚠️ **The `pod_config` form of this is worse than `disk_size`.** An explicit
`resources.requests.ephemeral-storage` in `config.kubernetes.pod_config`
*passes* prechecks, then never schedules: kube-scheduler reports `0 pods fit`
on every node and the job pends indefinitely with no error anywhere. On
2026-08-09 this held 53 jobs for 7+ hours while half the cluster sat idle, and
it presented as a bin-packing problem. Pods are immutable, so the only fix is
to cancel, remove the stanza, `sky down`, and relaunch.

Fast, free, node-local — and wiped on eviction. Stage datasets here, write temp
files here.

### `/mnt/raid0` — 12.2 TiB of free node-local NVMe (new, 2026-08-09)

Each node has 8 NVMe drives. One backs the container filesystem (that's what
`/tmp` is); the other seven are now a RAID0 array mounted at `/mnt/raid0`,
**12.2 TiB per node, free** — the capacity is already paid for in the node.
Live on **all 16 nodes**, ~195 TiB cluster-wide. (It was 13 of 16 for most of 2026-08-09;
the rollout's second phase finished the same afternoon. Keep the `nodeSelector` below
anyway — it selects nothing out today and it is what keeps the `hostPath` honest the day a
node is rebuilt or added without the array.)

Measured on the array (fio, 8 jobs, `direct=1`):

| | `/mnt/raid0` | `/tmp` (1 drive) | `/checkpoints` (NFS) |
|---|---|---|---|
| Sequential write | **20.1 GB/s** | 2.88 GB/s | 5.15 GB/s |
| Sequential read | **39.3 GB/s** | 6.05 GB/s | 21.3 GB/s* |
| **Random read 4k** | **7458 MiB/s** | 4390 MiB/s | 448 MiB/s |

\* warm cache, treat as an upper bound.

**Random read is the number that matters for a dataloader — 16.6× better than
`/checkpoints`.** If your job is I/O bound on small reads, this is the fix.

To use it, add a `nodeSelector` and a `hostPath` mount:

```yaml
config:
  kubernetes:
    pod_config:
      spec:
        nodeSelector:
          pantheon.inc/raid0: "enabled"
        volumes:
          - name: raid0
            hostPath:
              path: /mnt/raid0
              type: Directory        # load-bearing, see below
        containers:
          - volumeMounts:            # no `name:` key on this entry
              - mountPath: /mnt/raid0
                name: raid0
```

`type: Directory` is not optional. Without it, kubelet silently creates an empty
directory on any node lacking the array and your job writes to the boot disk
while looking healthy. With it you get a loud `FailedMount`.

⚠️ **This is a cache, not storage. Everything on it must be reconstructible.**

- **RAID0 across 7 drives** — one drive failure loses the whole node's array.
- **It does not survive a reboot or node replacement.** The array is recreated
  **empty**. A job that finds its data missing must be able to re-fetch it.
- **No quotas, `chmod 777`.** One greedy job fills 12.2 TiB and evicts its
  neighbours' data. Clean up after yourself.
- **The scheduler doesn't know which node holds what.** A pod can land on a node
  with a cold cache. Write for a miss: pull from object storage or `/checkpoints`
  and populate on first use. Budget **~1.3 GB/s** for the object-storage leg
  (measured 2026-08-10; see §5's bucket section) — a cold 1 TiB slice is about
  fifteen minutes, so a miss is survivable but not free.
- **Same 777 subdirectory trap as `/checkpoints`** — a directory created by a
  root-default image (`nvcr.io/...`) is unwritable by `sky`-default jobs
  (uid 1000). `chmod 777` anything you create for shared use.

Good fits: dataset slices, HF/model caches, embedding shards, checkpoint staging
before a durable write. Bad fits: the only copy of anything. Anything you want to keep goes to `/checkpoints` or a bucket before
the job ends.

⚠️ **Both node-local tiers are unpoliced**, and they are different sizes: `/tmp` has
~1.5 TiB free and `/mnt/raid0` has 12.2 TiB. Neither has a per-job quota, and filling
either triggers disk-pressure evictions for your neighbours as well as yourself. Clean up
large intermediates as you go.

### Reading the encodings bucket (bulk data)

```yaml
run: |
  export AWS_ACCESS_KEY_ID=$CRUSOE_S3_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$CRUSOE_S3_SECRET_KEY
  export AWS_ENDPOINT_URL=https://object.eu-iceland1-a.crusoecloudcompute.com
  aws s3 sync s3://encodings/<prefix>/ /tmp/data/
  python train.py --data /tmp/data
```

Keep the slice you pull under the ~1.5 TiB of free node NVMe. For prefixes with thousands of
objects, `s5cmd` is ~10× faster than `aws s3 sync`.

**Budget ~1.3 GB/s per client, and do not plan around more.** Measured 2026-08-10 on both
a GPU node and a `c1a.32x` with a 200 Gbps NIC: **~1.3 GB/s each**, with the NIC at 5%
utilisation. Throughput plateaus at 64 workers and *declines* beyond it, so adding workers
past that point makes it worse. Two nodes pulling concurrently reached 2.84 GB/s in
aggregate, which says the ceiling is **per client, not global** — the way to move more data
is more readers, not a bigger one. (A figure roughly six times this one circulated before
2026-08-10; it does not reproduce on either machine type and should not be planned
against.)

### Checkpoint code pattern

```python
import os, torch

CKPT_DIR = f"/checkpoints/{os.environ['PANTHEON_USER']}/{os.environ['EXPERIMENT_TAG']}"
os.makedirs(CKPT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, step):
    path = f"{CKPT_DIR}/step_{step:06d}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)

def load_latest_checkpoint(model, optimizer):
    if not os.path.exists(CKPT_DIR):
        return 0
    files = sorted(f for f in os.listdir(CKPT_DIR) if f.startswith("step_"))
    for f in reversed(files):
        try:
            ckpt = torch.load(f"{CKPT_DIR}/{f}", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            return ckpt["step"]
        except Exception:
            continue
    return 0

# Training loop
start_step = load_latest_checkpoint(model, optimizer)
for step in range(start_step, total_steps):
    train_one_step()
    if step % 500 == 0:
        save_checkpoint(model, optimizer, step)
```

Keep the volume lean: rotate old step files (keep last N + milestones). 10 TiB
is shared across everyone. There is **no automatic drain to hades** yet —
long-term checkpoint retention is manual for now; treat the volume as working
state, not an archive.

### What happens on eviction

1. Higher-priority job submitted → Kueue preempts your workload
2. Your pod is killed → SkyPilot marks the job RECOVERING
3. GPUs free up → SkyPilot re-launches on a new node
4. Your job starts, mounts `/checkpoints`, loads the latest checkpoint, resumes

No re-queuing logic needed from you. Save + load checkpoints. **Verified on this
cluster:** a job preempted three times resumed from step 81, then 122, each time
picking up exactly where the previous life ended.

### You get no warning — do not write a SIGTERM handler

Measured across 47 real preemptions on this cluster (three separate jobs), **a
`SIGTERM` handler in your training script never runs.** Traps on both `TERM` and
`EXIT` produced no output in any case. The user script is not PID 1, the signal
is not forwarded to it, and the process ends at SIGKILL — which no handler can
intercept.

So there is **no save window**. Plan for the process to stop between one
instruction and the next:

- Checkpoint on a **step interval you can afford to lose** (e.g. every 500
  steps), not on a shutdown hook.
- Write atomically — `torch.save` to a temp path, then `os.replace` — so a kill
  mid-write can't leave a truncated checkpoint as `latest`.
- Assume the last partial interval is gone. That is the cost of running at p3,
  and it is normal.

There is a second reason this matters: a job that ends on a **nonzero exit code**
is classified by SkyPilot as a *user program failure* and is **not retried**,
while a job whose pod simply disappears is recovered. A slow or blocking
shutdown path makes the bad outcome more likely, not less.

### wandb run continuity

Add these so eviction doesn't fragment your wandb charts:
```yaml
envs:
  WANDB_RESUME: allow
  WANDB_RUN_ID: my-experiment-tag   # same as EXPERIMENT_TAG
```

### Secrets

Pre-configured and global, so **you do not create these**:
`CRUSOE_S3_ACCESS_KEY`, `CRUSOE_S3_SECRET_KEY`, `WANDB_API_KEY`. Corroborated by a live
job on 2026-08-06, and the list `modal-skypilot`'s `config/secret-map.yaml` is allowed to
name — `tests/test_config.py::PLATFORM_SECRETS` holds it to exactly these three, because a
mapped secret suppresses the "create this first" action an unmapped one would raise, and a
stale entry there puts "Ready to submit" over a launch that fails.

Reference one by name; do not paste its value. That is the `secrets:NAME` form, and it is
what `modal-skypilot` emits:

```yaml
secrets:
  - secrets:WANDB_API_KEY          # a REFERENCE; the API server resolves it at launch
  - secrets:CRUSOE_S3_ACCESS_KEY
  - secrets:CRUSOE_S3_SECRET_KEY
```

Anything else you need, add at
https://pantheon.svc.skypilot.co/dashboard/secrets-manager — "Add Secret" → "Env Var" →
name it → paste the value — and then tell `modal-skypilot` about it by adding a line to
`config/secret-map.yaml`, or `convert` will raise an ACTION REQUIRED for it every time.

---

## 6. Templates

### Single-GPU experiment (start here)

```yaml
name: my-experiment

resources:
  accelerators: H200:1
  cpus: 20
  memory: 230

volumes:
  /checkpoints: checkpoints

envs:
  PANTHEON_USER: yourname@pantheon.inc
  EXPERIMENT_TAG: my-experiment-aug04
  WANDB_PROJECT: pfm-experiments
  WANDB_RESUME: allow
  WANDB_RUN_ID: my-experiment-aug04

secrets:
  - secrets:WANDB_API_KEY          # a reference, not a value -- see §5

setup: |
  pip install torch wandb --index-url https://download.pytorch.org/whl/cu128

run: |
  python train.py --lr 1e-4 --epochs 10
```

### Multi-GPU single-node

```yaml
name: my-ablation

resources:
  accelerators: H200:4
  cpus: 80
  memory: 920

volumes:
  /checkpoints: checkpoints

envs:
  PANTHEON_USER: yourname@pantheon.inc
  EXPERIMENT_TAG: ablation-lr1e4-bs64
  WANDB_PROJECT: pfm-ablations
  WANDB_RESUME: allow
  WANDB_RUN_ID: ablation-lr1e4-bs64

secrets:
  - secrets:WANDB_API_KEY          # a reference, not a value -- see §5

setup: |
  pip install torch torchvision wandb accelerate --index-url https://download.pytorch.org/whl/cu128

run: |
  torchrun --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    train.py --lr 1e-4 --batch-size 64
```

### Multi-node distributed

⚠️ **A partial-node multi-node job puts every "node" on one machine, and you
cannot stop it.** Measured 2026-08-09 across three separate `2x[H200:1]` jobs:
all three placed *both* pods on a single machine (twice on node-4, once on
node-10). The third carried an explicit
`topologySpreadConstraints` block with `maxSkew: 1` and
`whenUnsatisfiable: DoNotSchedule` on `kubernetes.io/hostname` — **it had no
effect.**

**Why the spread constraint did nothing** (measured 2026-08-10): every admitted Kueue
Workload carries a `status.admission.podSetAssignments[].topologyAssignment` naming an
exact host, and the cluster's `Topology` object is
`{"levels":[{"nodeLabel":"kubernetes.io/hostname"}]}`. Kueue's Topology-Aware Scheduling
picks the specific machine **at admission time, at hostname granularity**. By the time
kube-scheduler sees the pod the host is already chosen, so nothing in a `pod_config` —
spread constraints, affinity, anti-affinity — is ever evaluated.

Nothing forces separation when GPUs-per-node is below 8 — at 8 the arithmetic
does it, below that nothing does.

**The job cannot detect this.** `socket.gethostname()` returns `node-0` and
`node-1` regardless of physical placement, and `container_ips` are pod IPs, so a
co-located pair looks identical to a distributed one from inside. Consequences:
no real network traffic, so a "works multi-node" test proves nothing; both ranks
share a failure domain, so fault-tolerance tests are meaningless; and any
bandwidth figure measured this way is loopback.

**The only reliable way to get genuine separation is `H200:8` per node**, where
two pods cannot fit on one machine. Affinity and spread rules do not work here.
Verify any multi-node run with `kubectl get pods -o wide | grep <jobname>` and
check the NODE column actually differs before trusting a cross-node result.

Multi-node **requires** an InfiniBand `pod_config` — without it, pods get one
shared NIC with GPUDirect disabled and DDP runs ~40× slower (~11 GB/s per rank,
~2.3% MFU). With it, measured all-reduce busbw through a SkyPilot job is
**437.5 GB/s**, validated **2026-08-09** on a converted 2-node × 8 H200 job
(1.07 GB all_reduce, all 8 HCAs, IB not sockets, GDRDMA on all 16 channels). That is the
date `modal-skypilot` stamps into every multi-node header, and it is the number to quote.

(An older 479 GB/s figure from job 446 on 2026-08-06 is still true and is **not** a
ceiling this regressed from: different message size, a different pod builder, a busier
cluster. Neither number is comparable to the other, which is why only the dated one the
shim emits belongs in a template.)

> ### Do not hand-copy the IB block
>
> **`modal-skypilot` emits it for you.** Run your conversion through the shim
> and it produces a correct, dated block plus a node-side assertion that fails
> loudly at second zero if IB is missing — instead of silently training at 2.3%
> MFU for six hours.
>
> If you are hand-writing SkyPilot YAML rather than converting Modal code, copy
> the block from **the shim's output**, not from this document. There is one
> source of truth and this is not it. Every time someone has copied a block from
> a doc it has drifted: one fork added `priorityClassName: system-cluster-critical`
> (a *Kubernetes* pod priority class, unrelated to `--priority`, which can evict
> other people's work), added a `k8s.v1.cni.cncf.io/networks` annotation (inert
> here — Multus runs but there are zero NetworkAttachmentDefinitions), and the
> job never ran.
>
> ⚠️ **Never add a `dshm` volume or a `/dev/shm` mount.** SkyPilot already
> provides `/dev/shm` as a memory-backed emptyDir sized to node RAM (measured
> 1.9 TiB). Adding your own is a duplicate volume name *and* a duplicate
> mountPath — the API server rejects the pod with a `422` and the job sits in
> PENDING with nothing in `sky jobs queue` to say why. It is also ~30× smaller
> than what you already had.

The rest of a multi-node task looks like this. The `config:` block is omitted
deliberately — get it from the shim:

```yaml
name: distributed-run

resources:
  accelerators: H200:8
  cpus: 160
  memory: 1840

num_nodes: 2

# config.kubernetes.pod_config: emitted by modal-skypilot — see the box above.

volumes:
  /checkpoints: checkpoints

envs:
  PANTHEON_USER: yourname@pantheon.inc
  EXPERIMENT_TAG: distributed-aug04
  WANDB_PROJECT: pfm-distributed
  WANDB_RESUME: allow
  WANDB_RUN_ID: distributed-aug04
  NCCL_DEBUG: INFO
  NCCL_TOPO_FILE: /opt/nccl_topo/h200-141gb-sxm-ib-cloud-hypervisor.xml
  UCX_RNDV_SCHEME: get_zcopy
  UCX_TLS: self,sm,cuda_copy
  NCCL_IB_PCI_RELAXED_ORDERING: "1"
  NCCL_IB_SPLIT_DATA_ON_QPS: "0"
  NCCL_IB_QPS_PER_CONNECTION: "2"
  NCCL_IB_MERGE_VFS: "0"
  NCCL_IB_HCA: "^mlx5_0:1"
  NCCL_NVLS_ENABLE: "1"
  NCCL_IB_SL: "1"
  NCCL_IBEXT_DISABLE: "1"

secrets:
  - secrets:WANDB_API_KEY          # a reference, not a value -- see §5

setup: |
  pip install torch wandb --index-url https://download.pytorch.org/whl/cu128

run: |
  MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
  torchrun \
    --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    --nnodes=$SKYPILOT_NUM_NODES \
    --node_rank=$SKYPILOT_NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=12345 \
    train.py --lr 1e-4
```

Note: in distributed, only rank 0 should save checkpoints. All ranks load + `dist.barrier()`.

### Parameter sweep

```bash
for lr in 1e-3 3e-4 1e-4 3e-5 1e-5; do
  sky jobs launch sweep.yaml --env LR=$lr --env EXPERIMENT_TAG=sweep-lr-$lr --env WANDB_RUN_ID=sweep-lr-$lr -y
done
```

```yaml
name: sweep

resources:
  accelerators: H200:1
  cpus: 20
  memory: 230

volumes:
  /checkpoints: checkpoints

envs:
  PANTHEON_USER: yourname@pantheon.inc
  EXPERIMENT_TAG: sweep-lr-1e-4
  LR: "1e-4"
  WANDB_PROJECT: pfm-sweeps
  WANDB_RESUME: allow
  WANDB_RUN_ID: sweep-lr-1e-4

secrets:
  - secrets:WANDB_API_KEY          # a reference, not a value -- see §5

setup: |
  pip install torch wandb --index-url https://download.pytorch.org/whl/cu128

run: |
  python train.py --lr $LR
```

---

## 6.5 My job died — what happened?

Work down this list; each step is cheap and rules out a whole class.

| What you see | Likely cause | What to do |
|---|---|---|
| `FAILED_PRECHECKS` in ~2s, no pod ever created | Resource shape the cluster can't satisfy | `sky jobs queue -v` → DETAILS. Common: `disk_size` above 100 (don't set it at all), cpus above ~175, more than 8 GPUs on one node |
| Job ran, then `FAILED` with a nonzero exit | Your script | `sky jobs logs <id>` — the error is usually in the last 20 lines |
| Job `PENDING` for hours, nothing on the cluster | Waiting for capacity, or provisioning backoff | `sky jobs queue -v` → DETAILS says `Job is in backoff` if it's retrying. Normal when the cluster is full |
| `RECOVERING`, then resumes | Preempted by a higher-priority job | Nothing to do — this is expected at p3. Make sure you checkpoint |
| Died mid-run, no error in your logs | Preemption, or a node/GPU event | Check `sky jobs queue -v` for recoveries; ping #training-infra if several jobs died together |
| `CUDA error: an illegal memory address` / XID 31 in cluster logs | **Your code** — a null or out-of-bounds pointer in a kernel | `CUDA_LAUNCH_BLOCKING=1` plus `compute-sanitizer` on a single-GPU repro. Harmless to the hardware, but results after the fault are suspect |
| Multi-node job trains but very slowly | Missing the IB pod_config | Your NCCL log will show one HCA and `GDR Disabled`. Get the block from `modal-skypilot convert` and paste it in — §6's template deliberately does not contain one, and neither does any other document |

**If several people's jobs die around the same time, it's probably not you.**
GPUs can be pulled out of the scheduler by node-level health events without any
notification. Say so in #training-infra rather than debugging alone — that's an
infra check, not a code one.

---

## 6.7 Telemetry contract (`training-v1`) — required

The SkyPilot dashboard expects telemetry from every running GPU training job and
**complains loudly if it's missing**. Contract owner: Sambhav.

Every active GPU training job must:

- Publish a **W&B run URL** discoverable by SkyPilot
- Refresh these W&B **summary** fields every ~10s, **from global rank 0 only**:

```python
wandb.run.summary.update({
    "pantheon_telemetry/contract_version": "training-v1",
    "pantheon_telemetry/heartbeat_at":     time.time(),      # unix ts
    "pantheon_telemetry/total_steps":      total_steps,      # positive int
    "pantheon_telemetry/active_step":      step,
    "pantheon_telemetry/step_time_s":      last_step_seconds,
    "pantheon_telemetry/effective_mfu":    mfu,              # 0..1
    "pantheon_telemetry/wandb_url":        wandb.run.url,
})
```

- Log scalar history under the first metric matching `train/loss*`; log `val/loss`
  when validation begins.

**Heartbeats older than 30s are violations.** Missing links, missing fields,
missing train loss, invalid values, or stale heartbeats all produce dashboard
errors.

**`effective_mfu` must be end-to-end**, not compute-only: summed step times in
the denominator must nearly equal the wall-clock length of the run. A
compute-only MFU hides dataloader stalls, which is exactly what this is meant to
surface. You may log a separate compute-only MFU for ceiling analysis — but not
under `pantheon_telemetry/`.

Non-training jobs opt out explicitly, with a reason:

```yaml
resources:
  labels:
    telemetry-contract: opt-out
    telemetry-opt-out-reason: embedding-extraction
```

### Related job requirements

- **Assume every job is preemptible.** Lose no more than **5 minutes** of
  compute to a preemption — and note there is *no* SIGTERM warning (§6.6), so
  that budget has to come from step-interval checkpointing, not a shutdown hook.
  Dataloader, optimizer and model state all need checkpointing.
- **Survive hardware failure** without corrupting past results. Nothing outside
  non-ephemeral storage should be load-bearing.
- **CPU-only data processing belongs on CPU instances**, not on GPU nodes.
- **Dataloaders must prefetch and never block the GPU.** If dataloading stalls
  training, resize the model or GPU count rather than accepting the stall.
- **Use producer–consumer queues** for anything that shouldn't serialise —
  checkpoint writes especially. Update append-only data *before* metadata so a
  preemption can't leave a ghost shard.
- **Multiprocessing, not threading**, for parallelism in Python.

---

## 7. Commands

| Action | Command |
|---|---|
| Submit | `sky jobs launch experiment.yaml` |
| With priority | `sky jobs launch experiment.yaml --priority p1` |
| With env override | `sky jobs launch experiment.yaml --env LR=1e-3` |
| View queue | `sky jobs queue` |
| Stream logs | `sky jobs logs <job_id>` |
| Cancel | `sky jobs cancel <job_id>` |
| Dashboard | https://pantheon.svc.skypilot.co |

### Job states

| State | Meaning |
|---|---|
| PENDING | Queued, waiting for GPUs |
| RUNNING | Executing |
| RECOVERING | Evicted, auto re-launching |
| SUCCEEDED | Exit code 0 |
| FAILED | Check `sky jobs logs <id> --tail 100` |

---

## 8. Important notes
- **CUDA:** host driver is 12.8. Use `--index-url https://download.pytorch.org/whl/cu128` for PyTorch — it matches the driver exactly. CUDA 13+ wheels install cleanly and then fail at runtime.
- **CPU/RAM:** always set `cpus`/`memory` per §4 — unset jobs get ~4 CPU / 16 GB and starve. Node allocatable is 175.8 CPU / ~1928 GiB, so 176 CPU is unschedulable. **A node is 241 GiB per GPU; exceeding that on a sub-node job strands your neighbours' GPUs (§4).** SkyPilot renders `memory` as decimal `G` and Kubernetes reads `G` as 10⁹ — `memory: 1200` is **1117 GiB**, not 1200 GiB, so the number you write is about 6.9% larger than the binary figure you probably meant. The `241 GiB per GPU` share above is binary; divide a `memory:` value by 1.074 before comparing it.
- **NGC images pin their own packages.** `nvcr.io/nvidia/pytorch` ships `/etc/pip/constraint.txt` pinning `transformer-engine`; a `pip install` of any other version dies in ~24s with `ResolutionImpossible`. Set `PIP_CONSTRAINT=` (empty) in `setup:` before installing. Note also that TransformerEngine's build queries the GPU driver, so it cannot be compiled on a GPU-less job.
- **Scratch vs durable:** bulk/temp data → `/tmp` (node NVMe, ~1.5 TiB free, no request needed); checkpoints → `/checkpoints`. Never set `disk_size` (see §5) and never dump datasets on the checkpoint volume.
- **A slow job start is unpack, not download.** Measured 2026-08-10 on a 5.14 GiB image:
  **23 s to download, 74 s to unpack**, with containerd's `max_concurrent_unpacks = 1`. An
  in-region pull-through cache was tried and made no difference (3m16.5s against 3m15.5s
  direct). So a smaller image helps and a closer registry does not — and if you are waiting
  on a pull, you are waiting on one CPU decompressing layers in series.
- **Days 1–3:** 1-GPU and 1-node only. No multi-day runs until queue is proven.
- **Eviction is normal** for p3 jobs. Checkpoint early and often, rotate old checkpoints.
- **Don't use p0** without Mo/Sambhav approval.

---

## 9. For agents: mistakes to avoid

| Mistake | Fix |
|---|---|
| `sky launch` instead of `sky jobs launch` | Always `sky jobs launch` for unattended work |
| Accelerators H100/A100 | `H200:N` only |
| Missing `cpus` / `memory` | Set `20 × N` / `230 × N` GB per GPU |
| `memory` above `230 × N` on a sub-node job | Reduce it. A node is 241 GiB per GPU; the excess strands the GPUs you did not request and no other job can use them. Size to *anonymous* memory (`/sys/fs/cgroup/memory.stat`), not to `memory.current`, which counts reclaimable page cache. If `memory.events` shows `max 0`, the request has never been approached |
| Missing `volumes: /checkpoints: checkpoints` | Always mount for checkpoint survival |
| Missing PANTHEON_USER / EXPERIMENT_TAG | Required for checkpoint isolation |
| Same EXPERIMENT_TAG across different experiments | Checkpoints collide — unique per experiment |
| Datasets/temp files written to `/checkpoints` | Write to `/tmp` (node NVMe); the checkpoint volume is shared and capped |
| Setting `disk_size` at all | Remove it. >100 fails precheck silently; you already get ~1.5 TiB at `/tmp` |
| PyTorch with cu130 | Use `cu128` — host driver is 12.8 |
| `infra:` set in YAML | Remove — SkyPilot handles placement |
| Priority above p3 without approval | Default p3 |
| Writing output to `/workspace`, `~`, or a relative path | Write to `/checkpoints/...` or `/mnt/raid0` — the container filesystem is shared node disk and fills it for everyone |
| `cpus`:`memory` far from 1:11 | Keep the `20 × N` / `230 × N` ratio; skewed requests strand the other resource cluster-wide |
| Long backfill or opportunistic work at p3 | Use `p4` (10) — below default, yields to everything else |
| Bare numeric priority (`--priority 2`) | Tier names only — bare numbers land *below* p3 and get evicted first |
| Large inline payload in `setup:` (e.g. a base64 tarball) | Use `file_mounts:` — a long command line fails with `exec /bin/bash: argument list too long` before anything runs, and the job then retries forever |
| Relaunching after editing `config: kubernetes: pod_config` | `sky down <cluster>` first, or SkyPilot silently reuses the old pod config |
| Multi-node without an IB `pod_config` | Get it from `modal-skypilot` — without IB you get one NIC and ~40× slower collectives. Do not copy a block out of a doc |
| Adding a `dshm` volume or `/dev/shm` mount to `pod_config` | Remove it. SkyPilot already provides `/dev/shm` (memory-backed, ~1.9 TiB). Yours is a duplicate volume name *and* mountPath — the API server returns `422`, no pod is created, and the job sits PENDING with no visible cause |
| Any `ephemeral-storage` request in `pod_config` | Remove it. Same trap as `disk_size` but worse: it passes prechecks and then never schedules. Nodes allocate 111.5 GiB of ephemeral-storage (123.9 GiB capacity, less the 10% eviction reserve); a 1 Ti request means `0 pods fit` on every node, forever |
| `pip install` in an NGC image failing with `ResolutionImpossible` | NGC images ship `/etc/pip/constraint.txt` pinning `transformer-engine`. Set `PIP_CONSTRAINT=` (empty) before installing |
| Building TransformerEngine on a GPU-less job | Doesn't work — the build queries the driver and fails with `0 active drivers`. Allocate at least 1 GPU |

---

Questions? #training-infra or DM Mo.
