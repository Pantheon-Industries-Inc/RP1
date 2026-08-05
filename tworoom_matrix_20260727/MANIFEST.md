# Artifact manifest — TwoRoom matrix campaign (2026-07-27/28)

Code provenance: branch `eval-sweep`, commit `29886246e4386d0e81cdc3ef851371636a065034`.
Produced on 4xH100 80GB / 224 CPU / 2TB RAM; torch 2.4.1+cu124, transformers 4.57.6.
**All artifacts verified to load** (26/26) with self-consistent metadata.

Reproduction needs: this directory + the public dataset
(HF `quentinll/lewm-tworooms`). Latent caches are NOT stored — they are
regenerable from the stored world models in ~3 min (LeWM/PLDM) or ~4 min (DINO).

## World models (deterministic output of convert_tworoom_bases.py)

| file | size | sha256 (first 32) |
|---|---|---|
| `checkpoints/dinowm_tworoom/config.json` | 0.0 MB | `d210b3e29c36f4d934a2c9336fedfe60` |
| `checkpoints/dinowm_tworoom/weights.pt` | 168.9 MB | `61a167cee3663ba415871d6d9f0eb277` |
| `checkpoints/lejepa_tworoom/config.json` | 0.0 MB | `d94eb07205d346ddf17ebd6be53c28c2` |
| `checkpoints/lejepa_tworoom/weights.pt` | 68.9 MB | `d534461eb7fc0b69426c1f0138b30156` |
| `checkpoints/pldm_tworoom/config.json` | 0.0 MB | `cd4492a84fd42a6e47fc30d59c03accc` |
| `checkpoints/pldm_tworoom/weights.pt` | 68.9 MB | `47cf3731d9777ce4b43536f411d35d30` |

## TD critics (learned cost) and LIPv4 tandem critics

| file | size | sha256 (first 32) |
|---|---|---|
| `metrics/dinolip_k8b8_s0_value.pt` | 75.8 MB | `70ab2ad4c1c1c1ce9259b7478baacc68` |
| `metrics/sweep_pldm_a22k12_s0_value.pt` | 0.6 MB | `b11772659d62c0ca38ec12db14f9d63c` |
| `metrics/sweep_pldm_a28k8_s0_value.pt` | 0.6 MB | `6106b81c9279c33b4dd2c04be6bad391` |
| `metrics/sweep_pldm_a28k8_s1_value.pt` | 0.6 MB | `4f0484205befaae1e10b4cdf50cc36a8` |
| `metrics/sweep_pldm_a28k8_s2_value.pt` | 0.6 MB | `ca05f63ca47da9ff938dfc2b42672b1b` |
| `metrics/sweep_pldm_a35k12_s0_value.pt` | 0.6 MB | `049893840d97aec62aaeb2f0dbf58c16` |
| `metrics/sweep_pldm_a35k8_s0_value.pt` | 0.6 MB | `7d7708fc33b945723863d18df2c9880b` |
| `metrics/td_canon_dinowm_r200000_e0.1_n50_s0.pt` | 75.8 MB | `b1b5b3295dd91da3d44f55a6564fa76f` |
| `metrics/td_canon_dinowm_r200000_e0.1_n50_s1.pt` | 75.8 MB | `7ce9ee967eb0881bb9dc718f1bc24817` |
| `metrics/td_canon_dinowm_r200000_e0.1_n50_s2.pt` | 75.8 MB | `ed8bf591c64da48dfd7f6a846930ab03` |
| `metrics/td_canon_lejepa_e0.1_n50_s0.pt` | 0.6 MB | `419800f572292dd334b2b3d89b572210` |
| `metrics/td_canon_lejepa_e0.1_n50_s1.pt` | 0.6 MB | `4bb643f1f234c8e2e3bd50e3bf00ed27` |
| `metrics/td_canon_lejepa_e0.1_n50_s2.pt` | 0.6 MB | `35cc64db98595461dbfa6616afea5abe` |
| `metrics/td_canon_pldm_e0.1_n50_s0.pt` | 0.6 MB | `c2ee4605b4c6d9e334bee3283b0e62b2` |
| `metrics/td_canon_pldm_e0.1_n50_s1.pt` | 0.6 MB | `299429504350a2b375fcd10601f13f01` |
| `metrics/td_canon_pldm_e0.1_n50_s2.pt` | 0.6 MB | `63be54a2fda26562c032869d2756002f` |
| `metrics/trm_canon_dinowm_r200000_v4k4_s0_value.pt` | 75.8 MB | `a1dca7c40b21dd9d3e0a83d259641f58` |
| `metrics/trm_canon_lejepa_v4_s0_value.pt` | 0.6 MB | `1e928cd9affd23d91759d6dfd1f7476b` |
| `metrics/trm_canon_lejepa_v4_s1_value.pt` | 0.6 MB | `a33598ab03f4aca50ca2a93b514c1192` |
| `metrics/trm_canon_lejepa_v4_s2_value.pt` | 0.6 MB | `8c6b44e80ec337070b7efffca07d49d4` |
| `metrics/trm_canon_pldm_v4_s0_value.pt` | 0.6 MB | `9dee68138d2f2acdab5f1ea1307a6324` |
| `metrics/trm_canon_pldm_v4_s1_value.pt` | 0.6 MB | `1c076221bdd4a82bf56de3eb2b164a07` |
| `metrics/trm_canon_pldm_v4_s2_value.pt` | 0.6 MB | `04f292cf006121c874d4f496315dfbf1` |

## LIPv4 planners

| file | size | sha256 (first 32) |
|---|---|---|
| `actors/dinolip_k8b8_s0.pt` | 1.3 MB | `37873417e683ed9e41073e4a147e8a55` |
| `actors/sweep_pldm_a22k12_s0.pt` | 1.3 MB | `44259da9710e57410717ff42bc9b62e4` |
| `actors/sweep_pldm_a28k8_s0.pt` | 1.3 MB | `cc830b23a69ac05fa558df3bd291efd9` |
| `actors/sweep_pldm_a28k8_s1.pt` | 1.3 MB | `937bb7ebc27a1fc5268401add44eb088` |
| `actors/sweep_pldm_a28k8_s2.pt` | 1.3 MB | `9d781a2bdc341e57d549955f6c22b428` |
| `actors/sweep_pldm_a35k12_s0.pt` | 1.3 MB | `f62801846a11ddd9b9efab7feb70c430` |
| `actors/sweep_pldm_a35k8_s0.pt` | 1.3 MB | `14085819f0b86da2cb7ca798d857e9cd` |
| `actors/trm_canon_dinowm_r200000_v4k4_s0.pt` | 1.3 MB | `24c4206a5bbe79f84273cada375e70b1` |
| `actors/trm_canon_lejepa_v4_s0.pt` | 1.3 MB | `323a513134e00273f2ab1dec233cd47c` |
| `actors/trm_canon_lejepa_v4_s1.pt` | 1.3 MB | `24d6b073ead4047a11a0e97434a28910` |
| `actors/trm_canon_lejepa_v4_s2.pt` | 1.3 MB | `ba122481a786fd41b3d0071ec68d13ca` |
| `actors/trm_canon_pldm_v4_s0.pt` | 1.3 MB | `573cd341a2e29005450d60a9b8883ce8` |
| `actors/trm_canon_pldm_v4_s1.pt` | 1.3 MB | `8e732e0f2c4bed0c4c31ea57404f1719` |
| `actors/trm_canon_pldm_v4_s2.pt` | 1.3 MB | `189d33eba24a6cb882d5f109a4d5a6de` |


## PWM reactive policy (added 2026-07-29)

| file | size | md5 (first 32) |
|---|---|---|
| `actors/pwm_lejepa_s0.pt` | 3.4M | `f9f9d4a23c3ef42bd5867d6affdccf96` |
| `actors/pwm_lejepa_s1.pt` | 3.4M | `6d76789dd06989dca8b13f19d3fdafea` |
| `actors/pwm_lejepa_s2.pt` | 3.4M | `e77e56e8d95fce763759e6516d7f4006` |
| `actors/pwm_pldm_s0.pt` | 3.4M | `646159e86cff9d6cbd8f05cd6b160c8f` |
| `actors/pwm_pldm_s1.pt` | 3.4M | `d985179160e0441622b01a1157d4ebcc` |
| `actors/pwm_pldm_s2.pt` | 3.4M | `a740395ea774b321e158679378cb79bf` |
| `actors/pwm_dinowm_r200000_s0.pt` | 154M | `3a96020e20f61c343a45fb11208ab535` |
| `metrics/pwm_lejepa_s0_value.pt` | 584K | `90427a55cd1f26c44306759e3c14b30f` |
| `metrics/pwm_lejepa_s1_value.pt` | 584K | `970a9a131b2a0d2ab563e2d9c6aa98a6` |
| `metrics/pwm_lejepa_s2_value.pt` | 584K | `d85074ab98e68b01f990e96e70405479` |
| `metrics/pwm_pldm_s0_value.pt` | 584K | `0afa89cd5eccb0a0480c9dd14bce3984` |
| `metrics/pwm_pldm_s1_value.pt` | 584K | `60dce5dded833c12acb3e9604b19fc37` |
| `metrics/pwm_pldm_s2_value.pt` | 584K | `768149a0dfdbc85cab46c3d8a8c241ca` |

DINO PWM co-trained critic (~150 MB) intentionally not archived — regenerates
deterministically from `metrics/td_canon_dinowm_r200000_e0.1_n50_s0.pt` + seed 0
via `code/tworoom_pwm.sh train dinowm <gpu> 0`. LIP-K truncation checkpoints not
archived — 2-line `ck["iters"]` patch (HYPERPARAMS §4.4) regenerates them from the
archived K=8 actors.

## Post-campaign journals & drivers (added 2026-07-29)

| file | contents |
|---|---|
| `results/it10_driver.log` | 72 cells, CEM at paper budget (§9.3) |
| `results/lipk_driver.log` | 72 cells, LIPv4 K∈{4,2} truncation (§10.2) |
| `results/pwm_driver.log` | PWM train + 84 eval cells, proto & rh1 (§10.5) |
| `code/it10_rerun.sh`, `code/lipk_sweep.sh`, `code/tworoom_pwm.sh` | drivers |
| `logs_outputs_20260729.tar.gz` | all 238 per-cell eval/train logs |
