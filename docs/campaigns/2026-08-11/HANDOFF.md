# RLP campaign handoff — 2026-08-11

## Source of truth

- Protocol: `configs/campaign/rlp_20260811.yaml`
- Matrix validator: `python -m rlp.campaigns.rlp_20260811 validate`
- Current results: `docs/campaigns/2026-08-11/RESULTS.md`
- Implementation: `src/rlp/`
- Historical launch record: `docs/campaigns/2026-08-11/launch_state.json`

The running jobs were submitted from the historical `eval-sweep` snapshot.
They must not be relaunched from this migration branch. The refactored package
owns future execution; dated vendored source trees are provenance only.

## Protocol invariants

- 16 jobs x 4 H200 = 64 H200 maximum.
- Every job is p1 with 80 CPUs and 920 GiB RAM.
- TwoRoom LeWM and PLDM actors are trained separately at h25 and h100 with
  `tel-exact`, frozen critics, receding horizon 1, and max-delta 5/20.
- Reacher reruns only three-frame latent/value CEM, receding horizon 1, with
  the predicted chunk landing on graded step 50.
- OGBench replicates the established terminal recipe at receding horizon 5;
  h25 and h100 use distinct actors with max-delta 10/20.
- Selection uses draws 50/51. Headline reporting uses draws 42/43/44.

Previous OGBench terminal results to reproduce are LeWM 89.1/82.4 and PLDM
82.9/77.1 at h25/h100. The later 66.9/59.6 and 60.3/51.7 runs used receding
horizon 1 plus other recipe changes and are not terminal replications.

## Active managed jobs

Status snapshot: 2026-08-11 06:54 CEST. All 16 were running.

| IDs | work |
|---|---|
| 4164–4166 | TwoRoom LeWM h25 shards a/b/c |
| 4167–4169 | TwoRoom PLDM h25 shards a/b/c |
| 4173–4175 | TwoRoom LeWM h100 shards a/b/c |
| 4176–4178 | TwoRoom PLDM h100 shards a/b/c |
| 4179 | Reacher LeWM deadline CEM |
| 4180 | Reacher PLDM deadline CEM |
| 4181 | OGBench LeWM terminal h25/h100 replication |
| 4183 | OGBench PLDM terminal h25/h100 replication |

Superseded jobs 4139/4141/4142/4143/4145/4149, 4150–4153, and 4158–4163
were cancelled. Do not recover or quote them.

## Commands

Validate the migrated matrix without submitting anything:

```bash
python -m rlp.campaigns.rlp_20260811 validate
```

Inspect current managed state and bounded logs:

```bash
sky jobs queue -o json | jq -r '.[] | select(.job_name | startswith("rlp11-")) | [.job_id,.job_name,.status,.job_duration] | @tsv' | sort -n
sky jobs logs JOB_ID --no-follow --tail 200 -r
sky jobs logs JOB_ID --controller --no-follow --tail 200 -r
```

## Completion procedure

1. Require every expected actor seed and report draw.
2. Select TwoRoom hyperparameters independently at h25 and h100 on draws 50/51.
3. Report only draws 42/43/44.
4. For OGBench, report the diagonal: h25 actor at h25 and h100 actor at h100.
5. Keep superseded OGBench rh=1 results out of headline rows.
6. Update only the corresponding cells in `RESULTS.md`; keep exactly three tables.
