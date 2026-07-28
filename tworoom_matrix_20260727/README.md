# tworoom_matrix_20260727 — TwoRoom planner × cost matrix

Campaign of record for the TwoRoom comparison of **LIPv4** (learned amortised planner)
against sampling/gradient planners on the native latent cost and on a learned TD
quasimetric cost, across the three author-released world models.

## Read these first

| file | contents |
|---|---|
| `RESULTS_tworoom_matrix.md` | results, findings, caveats, and the defects that had to be fixed |
| `HYPERPARAMS.md` | every hyperparameter: env, dataset, all three WM architectures, planners, TD, LIPv4, caches, protocol, software versions |
| `MANIFEST.md` | sha256 + size of every checkpoint |

## Headline

All three bases reproduce the published TwoRoom values:

| base | ours (std × h25) | paper |
|---|---|---|
| PLDM | 96.7 | 97 |
| LeWM | 89.3 | 87 |
| DINO-WM | 100.0 | 100 |

LIPv4 vs the strongest native-latent planner, and the compute it takes:

**Status: COMPLETE** — all cells finished 2026-07-28. All 26 artifacts verified to load.

| | rollouts / plan step | LeWM h25/h50 | PLDM h25/h50 | DINO h25/h50 |
|---|---|---|---|---|
| Latent + MPPI † | 9,000 | 65.3 / 47.3 | 71.3 / 55.3 | 95.3 / 96.7 |
| Latent + Adam | 3,000 | 92.0 / 67.3 | 92.0 / 72.0 | 96.7 / 95.3 |
| Latent + CEM | 9,000 | 89.3 / 54.7 | 96.7 / 77.3 | **100.0** / 98.0 |
| TD + CEM | 9,000 | **100.0** / 99.8 | **98.7 / 99.6** | 99.8 / **100.0** |
| **LIPv4** | **8** | **100.0 / 100.0** | 97.1 / 98.9 | 99.3 / 99.3 |

† MPPI at repository defaults; temperature never tuned — a lower bound.

The h50 column is where methods separate: the native latent cost degrades by −34.6
(LeWM), −19.4 (PLDM) and −2.0 (DINO-WM) when the goal distance doubles — ordered by
latent capacity — while the learned-cost and amortised arms do not degrade at all.

## Layout

```
checkpoints/         converted world models (deterministic output of the converter)
  lejepa_tworoom/    LeWM   — ViT-tiny, 192-d CLS latent
  pldm_tworoom/      PLDM   — identical architecture, different objective
  dinowm_tworoom/    DINO-WM — frozen DINOv2-S, 77,224-d flat patch-token latent
metrics/             TD quasimetric critics (the learned cost), 3 seeds per base
actors/              LIPv4 planners
  trm_canon_lejepa_v4_s{0,1,2}.pt          LeWM, K=8, amax 2.2
  trm_canon_pldm_v4_s{0,1,2}.pt            PLDM, K=8, amax 2.2
  trm_canon_dinowm_r200000_v4k4_s0.pt      DINO-WM, K=4
  dinolip_k8b8_s0.pt                       DINO-WM, K=8 (batch 8)
  sweep_pldm_a28k8_s{0,1,2}.pt             PLDM, amax 2.8 (sweep winner)
results/             per-cell CSVs and driver logs (raw, unaggregated)
code/                snapshot of the exact scripts and configs that ran
```

## Verifying the archive still reproduces

```bash
STABLEWM_HOME=$(pwd) python3 verify_reproduction.py
```

Loads all three world models and compares their latents on a fixed synthetic input
against `ref_latents.npz`, captured on the campaign machine (Linux / H100 /
torch 2.4.1 / transformers 4.57.6). Needs no dataset and no GPU; runs in seconds.
Observed drift on macOS / M2 Max / torch 2.12.1 / transformers 5.12.1:
**1.6e-4, 6.6e-5, 4.2e-4** — float32 accumulation-order noise, means equal to six
decimals.

> **This check earned its keep.** `lejepa` and `pldm` originally would not load at
> all under transformers 5.x: the stored state_dicts carry the 4.x ViT layout
> (`encoder.encoder.layer.N.attention.attention.query`) while 5.x builds
> `encoder.layers.N.attention.q_proj`, so every encoder key was missing.
> `load_pretrained` now detects the mismatch and remaps in either direction
> (`stable_worldmodel/wm/utils.py::_vit_layout_compat`, snapshot in
> `code/stable_worldmodel/wm_utils.py`). DINOv2-backed models are unaffected.

## Reproducing

1. Fetch the dataset: HF `quentinll/lewm-tworooms` → `tworoom.tar.zst` → `tworoom.h5`
   (920,809 frames / 10,000 episodes). `code/scripts_plan/tworoom_pod_setup.sh` does this.
2. Convert the author archives:
   `code/scripts_plan/convert_tworoom_bases.py --only lejepa,pldm,dinowm`
   (verify against `MANIFEST.md`).
3. Phase 1 — caches, TD ×3 seeds, LIPv4 ×3 seeds:
   `tworoom_phase1_matrix.sh <base> <gpu>`; DINO needs `MAX_ROWS=200000 LIP_BATCH=16`
   and `ITERS=4` (see `HYPERPARAMS.md` §7).
4. Evaluate: `tworoom_matrix.sh <base> <gpu> 8 "anchor full"` (or `cemonly` for DINO).
   Run `anchor` first — it spends 3 evals checking against 87/97/100 before committing
   to the full card.

## Known limitations

- **No held-out split.** TD critics and LIPv4 actors train on the same 10,000 episodes
  the evaluation draws its tasks from, so these are in-distribution fit rather than
  generalisation estimates. `eval.ep_range=lo:hi` exists for a held-out card; DINO's
  200k-row cache touches only episodes 0–2,170, so episodes 8,000–9,999 are already
  clean for it.
- **MPPI is untuned** — no plan-time config shipped with the repository; ours mirrors CEM
  with the default temperature, never swept.
- **DINO-WM coverage:** its three latent arms are complete (CEM, Adam, MPPI, all n=3),
  but its *learned-cost* arms are CEM-only — TD+Adam and TD+MPPI were not run, since
  DINO cells cost ~34 min each and CEM dominated both other optimisers on the two
  smaller-latent bases. LIPv4 has 1 actor seed where the twins have 3. Its Adam arm ran
  at `solver.batch_size=1` (~50 GB; 200 retained candidates need 78.5 GB and exhaust an
  80 GB card) — env-chunking only, so the algorithm is unchanged, but the RNG
  realisation differs from the twins' batch-10 runs.
- **Statistical resolution:** n=50 binary trials ⇒ SE ≈ 2.8 points, so differences ≤2
  points between near-ceiling arms are unresolved. Several arms sit at 99–100.
