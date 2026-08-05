mods = ["torch","torchvision","numpy","loguru","hydra","omegaconf","lance","h5py","hdf5plugin",
        "einops","tqdm","PIL","imageio","gymnasium","dm_control","mujoco","transformers",
        "pyarrow","scipy","matplotlib","wandb","lightning","timm"]
ok, miss = [], []
for m in mods:
    try:
        mod = __import__(m)
        ok.append(m + "=" + str(getattr(mod, "__version__", "?")))
    except Exception:
        miss.append(m)
print("PRESENT:", ", ".join(ok))
print()
print("MISSING:", ", ".join(miss) if miss else "(none)")
