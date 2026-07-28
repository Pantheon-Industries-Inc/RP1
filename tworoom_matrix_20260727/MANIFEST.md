# Artifact manifest — TwoRoom matrix campaign (2026-07-27)

Code provenance: branch `eval-sweep`, commit `6cb926c798a094ebbaa166ca26304fbf7b1f3329`.
Produced on 4xH100 80GB / 224 CPU / 2TB RAM; torch 2.4.1+cu124, transformers 4.57.6.

## World models (converted from the author-released archives)

Deterministic output of `code/scripts_plan/convert_tworoom_bases.py` applied to
`stable-worldmodel/checkpoints/tworoom/pretrained/{lewm,pldm,dinowm}.tar.zst`.
Encoder parity vs the reference implementation: 2.4-2.8e-05.

| file | size | sha256 |
|---|---|---|
| `checkpoints/dinowm_tworoom/weights.pt` | 176M | `61a167cee3663ba415871d6d9f0eb277...` |
| `checkpoints/lejepa_tworoom/weights.pt` |  80M | `d534461eb7fc0b69426c1f0138b30156...` |
| `checkpoints/pldm_tworoom/weights.pt` |  80M | `47cf3731d9777ce4b43536f411d35d30...` |
| `checkpoints/dinowm_tworoom/config.json` | 4.0K | `d210b3e29c36f4d934a2c9336fedfe60...` |
| `checkpoints/lejepa_tworoom/config.json` | 4.0K | `d94eb07205d346ddf17ebd6be53c28c2...` |
| `checkpoints/pldm_tworoom/config.json` | 4.0K | `cd4492a84fd42a6e47fc30d59c03accc...` |

## TD critics (learned cost; 3 initialization seeds per base)

| file | size | sha256 |
|---|---|---|
| `metrics/td_canon_dinowm_r200000_e0.1_n50_s0.pt` |  80M | `b1b5b3295dd91da3d44f55a6564fa76f...` |
| `metrics/td_canon_dinowm_r200000_e0.1_n50_s1.pt` |  80M | `7ce9ee967eb0881bb9dc718f1bc24817...` |
| `metrics/td_canon_dinowm_r200000_e0.1_n50_s2.pt` |  80M | `ed8bf591c64da48dfd7f6a846930ab03...` |
| `metrics/td_canon_lejepa_e0.1_n50_s0.pt` | 584K | `419800f572292dd334b2b3d89b572210...` |
| `metrics/td_canon_lejepa_e0.1_n50_s1.pt` | 584K | `4bb643f1f234c8e2e3bd50e3bf00ed27...` |
| `metrics/td_canon_lejepa_e0.1_n50_s2.pt` | 584K | `35cc64db98595461dbfa6616afea5abe...` |
| `metrics/td_canon_pldm_e0.1_n50_s0.pt` | 584K | `c2ee4605b4c6d9e334bee3283b0e62b2...` |
| `metrics/td_canon_pldm_e0.1_n50_s1.pt` | 584K | `299429504350a2b375fcd10601f13f01...` |
| `metrics/td_canon_pldm_e0.1_n50_s2.pt` | 584K | `63be54a2fda26562c032869d2756002f...` |

## LIPv4 actors

| file | size | sha256 |
|---|---|---|
| `actors/dinolip_k8b8_s0.pt` | 1.3M | `37873417e683ed9e41073e4a147e8a55...` |
| `actors/sweep_pldm_a28k8_s0.pt` | 1.3M | `cc830b23a69ac05fa558df3bd291efd9...` |
| `actors/sweep_pldm_a28k8_s1.pt` | 1.3M | `937bb7ebc27a1fc5268401add44eb088...` |
| `actors/sweep_pldm_a28k8_s2.pt` | 1.3M | `9d781a2bdc341e57d549955f6c22b428...` |
| `actors/trm_canon_dinowm_r200000_v4k4_s0.pt` | 1.3M | `24c4206a5bbe79f84273cada375e70b1...` |
| `actors/trm_canon_lejepa_v4_s0.pt` | 1.3M | `323a513134e00273f2ab1dec233cd47c...` |
| `actors/trm_canon_lejepa_v4_s1.pt` | 1.3M | `24d6b073ead4047a11a0e97434a28910...` |
| `actors/trm_canon_lejepa_v4_s2.pt` | 1.3M | `ba122481a786fd41b3d0071ec68d13ca...` |
| `actors/trm_canon_pldm_v4_s0.pt` | 1.3M | `573cd341a2e29005450d60a9b8883ce8...` |
| `actors/trm_canon_pldm_v4_s1.pt` | 1.3M | `8e732e0f2c4bed0c4c31ea57404f1719...` |
| `actors/trm_canon_pldm_v4_s2.pt` | 1.3M | `189d33eba24a6cb882d5f109a4d5a6de...` |
