| TwoRoom — success % (h25 / h100) | WM rollouts / decision | Hyperparameters | LeWM | PLDM | DINO-WM |
|---|---:|---|---:|---:|---:|
| latent + MPPI | 9,000 | MPPI: 300 samples × 30 iterations, temperature 0.5 | 53.3 / pending | 54.7 / pending | 95.3 / — |
| latent + Adam | 3,000 | AdamW: 100 samples × 30 steps, LR 0.1 | 76.0 / pending | 77.3 / pending | 96.7 / — |
| latent + CEM | 9,000 | CEM: 300 samples × 30 iterations | 77.3 / pending | 82.0 / pending | 100.0 / — |
| value + MPPI | 9,000 | value: γ=1.0, expectile=0.1, n-step=50, 6k steps, batch=1024; MPPI: 300×30, temperature=0.5 | 62.7 / pending | 68.0 / pending | 68.0 / — |
| value + Adam | 3,000 | value: γ=1.0, expectile=0.1, n-step=50, 6k steps, batch=1024; AdamW: 100×30, LR=0.1 | 74.7 / pending | 85.3 / pending | 80.0 / — |
| value + CEM | 9,000 | value: γ=1.0, expectile=0.1, n-step=50, 6k steps, batch=1024; CEM: 300×30 | 83.3 / pending | 76.7 / pending | 82.0 / — |
| RLP terminal, previous h25 | 17 | LeWM: amax=2.2; PLDM: amax=2.8; mean-weight=0.1, actor-LR=3e-4, replay=0, expand=0, 8k steps, batch=128, K=8, H=5 | 100.0 / pending | 97.1 / pending | 99.6 / — |
| **RLP tel-exact, horizon-matched rerun** | **17** | separate h25/h100 actors; max-delta=5/20 fs5; LeWM amax=1.8/2.2/2.6; PLDM amax=2.4/2.8/3.2; actor-LR=1e-4/3e-4, mean-weight=0.1, replay=0, expand=0, critic frozen, 6k steps, batch=128, K=8, H=5, rh=1 | **pending / pending** | **pending / pending** | — |

| Reacher — held-at-end success % (0.1 / 0.05 rad) | WM rollouts / decision | Hyperparameters | LeWM | PLDM | DINO-WM |
|---|---:|---|---:|---:|---:|
| latent-window + MPPI | 9,000 | 3-frame latent L2; MPPI: 300×30, temperature=0.5, rh=5 | 39.3 / 13.7 | 37.0 / 10.7 | — |
| latent-window + Adam | 3,000 | 3-frame latent L2; AdamW: 100×30, LR=0.1, rh=5 | 56.3 / 22.3 | 56.7 / 23.0 | — |
| latent-window + CEM | 9,000 | 3-frame latent L2; CEM: 300×30, rh=5 | 86.3 / 53.7 | 83.3 / 47.3 | — |
| value + MPPI | 9,000 | value: 3-frame window, γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=1024; MPPI: 300×30, temperature=0.5, rh=5 | 25.7 / 9.3 | 21.0 / 7.0 | — |
| value + Adam | 3,000 | value: 3-frame window, γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=1024; AdamW: 100×30, LR=0.1, rh=5 | 45.3 / 15.3 | 40.3 / 10.0 | — |
| value + CEM | 9,000 | value: 3-frame window, γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=1024; CEM: 300×30, rh=5 | 55.0 / 26.7 | 47.3 / 16.7 | — |
| latent-window + CEM, deadline-50 | 9,000 | 3-frame latent L2; CEM: 300×30, rh=1, score predicted chunk landing at step 50 | invalid — rerun required | invalid — rerun required | — |
| value + CEM, deadline-50 | 9,000 | value: 3-frame window, γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=1024; CEM: 300×30, rh=1, score predicted chunk landing at step 50 | invalid — rerun required | invalid — rerun required | — |
| RLP terminal, previous campaign | 17 | LeWM: amax=2.4, mean-weight=0.3, actor-LR=1e-4; PLDM: amax=1.4, mean-weight=0.1, actor-LR=3e-4; 6k steps, batch=256, K=8, H=5 | 98.8 / 68.0 | 91.0 / 50.3 | — |
| RLP terminal, current control | 17 | LeWM: amax=2.4, mean-weight=0.3, actor-LR=1e-4; PLDM: amax=1.4, mean-weight=0.1, actor-LR=3e-4; common: replay=0.5, expand=0, critic frozen from step 0, 6k steps, batch=256, K=8, H=5, rh=5; report draws 42–47 | 93.1 / 51.9 | 74.4 / 33.8 | — |
| **RLP tel-exact, selected** | **17** | LeWM: amax=2.8, mean-weight=0.3, actor-LR=1e-4; PLDM: amax=1.2, mean-weight=0.1, actor-LR=3e-4; common: replay=0.5, expand=0, critic frozen from step 0, 6k steps, batch=256, K=8, H=5, rh=5; selected only on held-at-end 0.1-rad draws 50/51, report draws 42–47 | **84.3 / 42.3** | **83.1 / 37.4** | — |
| published paper | — | — | 98.2 / 66.3 | 94.2 / — | — |

| Cube / OGBench — success % (h25 / h100) | WM rollouts / decision | Hyperparameters | LeWM | PLDM | DINO-WM |
|---|---:|---|---:|---:|---:|
| latent + MPPI | 9,000 | MPPI: 300 samples × 30 iterations, temperature 0.5 | 58.7 / 46.0 | 58.0 / 46.0 | 74.7 / — |
| latent + Adam | 3,000 | AdamW: 100 samples × 30 steps, LR=0.1 | 72.7 / 56.7 | 58.7 / 51.3 | 69.3 / — |
| latent + CEM | 9,000 | CEM: 300 samples × 30 iterations | 75.3 / 53.3 | 62.7 / 54.7 | 82.0 / — |
| value + MPPI | 9,000 | twins value: γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=1024; DINO value: dense 75,264-D critic, γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=256; MPPI: 300×30, temperature=0.5 | 60.7 / 47.3 | 64.7 / 50.0 | 66.0 / — |
| value + Adam | 3,000 | twins value: γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=1024; DINO value: dense 75,264-D critic, γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=256; AdamW: 100×30, LR=0.1 | 70.0 / 59.3 | 63.3 / 50.7 | 64.0 / — |
| value + CEM | 9,000 | twins value: γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=1024; DINO value: dense 75,264-D critic, γ=0.98, expectile=0.03, n-step=50, 12k steps, batch=256; CEM: 300×30 | 73.3 / 62.0 | 66.0 / 52.7 | 68.0 / — |
| RLP terminal, previous campaign | 17 | LeWM: amax=1.6; PLDM: amax=4.5; mean-weight=0.1, actor-LR=3e-4→3e-5, replay=0.5, expand=1, critic live for first 3k/6k steps, batch=256, K=8, H=5, rh=5 | 89.1 / 82.4 | 82.9 / 77.1 | pending / — |
| **RLP terminal, corrected replication** | **17** | separate h25/h100 actors with max-delta=10/20 fs5; otherwise exact previous recipe: LeWM amax=1.6, PLDM amax=4.5, mean-weight=0.1, actor-LR=3e-4→3e-5, replay=0.5, expand=1, critic live for first 3k/6k steps, batch=256, K=8, H=5, rh=5 | **pending / pending** | **pending / pending** | — |
| RLP + Dyna, previous campaign | 17 | LeWM: amax=1.6; PLDM: amax=4.5; mean-weight=0.1, actor-LR=3e-4→3e-5, replay=0.5, expand=1, 6k steps, batch=256, K=8, H=5 | 94.4 / — | 91.3 / — | — |
