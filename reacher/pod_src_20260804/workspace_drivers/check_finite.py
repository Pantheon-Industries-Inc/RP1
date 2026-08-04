"""Verify the trained value blobs are finite despite the late `td_loss nan` print.

Campaign notes call that print a torch-2.4 cosmetic with clean weights, but this
pod runs torch 2.5.1, so the old diagnosis is not automatically transferable --
check rather than assume. A non-finite teacher would silently corrupt every
actor trained against it.
"""

import torch

for tag in ["alr1e4_s0", "alr1e4_s1", "alr1e4_s2", "centre_s0", "steps16k_s2"]:
    p = f"/workspace/metrics/lip4_wa_{tag}_value.pt"
    try:
        b = torch.load(p, map_location="cpu", weights_only=False)
        sd = b["state_dict"]
        bad = [k for k, v in sd.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
        n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
        ld = b["latent_dim"]
        print(f"{tag:<14} latent_dim={ld:<5} params={n:<9} non-finite tensors={len(bad)}"
              + (f"  -> {bad[:3]}" if bad else "  OK"))
    except Exception as e:  # noqa: BLE001
        print(f"{tag:<14} {type(e).__name__}: {e}")

# and the actors themselves
print()
for tag in ["alr1e4_s0", "centre_s0"]:
    p = f"/workspace/actors/lip4_wa_{tag}.pt"
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        bad = [k for k, v in sd.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
        print(f"actor {tag:<12} non-finite tensors={len(bad)}" + ("  OK" if not bad else f" -> {bad[:3]}"))
    except Exception as e:  # noqa: BLE001
        print(f"actor {tag:<12} {type(e).__name__}: {e}")
