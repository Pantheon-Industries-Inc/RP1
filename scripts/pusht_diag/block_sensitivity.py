"""E9: is the PushT critic an agent-position distance in disguise?
Env: DIAG, ACTOR, WM, H5, OUT, DRAW. For each of the first NT eval tasks:
  sweep the BLOCK position (agent fixed at start, block angle fixed) and the
  AGENT position (block fixed at start) on a GxG grid; report V's dynamic
  range for each sweep, plus the decomposition
  V(start) / V(agent@goal, block@start) / V(agent@start, block@goal) / V(goal config).
"""
from __future__ import annotations
import io, json, os
from pathlib import Path
import h5py, numpy as np, torch
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import gymnasium as gym, stable_worldmodel  # noqa
from PIL import Image
DIAG, ACTOR, WMD, H5, OUT = (Path(os.environ[k]) for k in ("DIAG", "ACTOR", "WM", "H5", "OUT"))
DRAW = int(os.environ.get("DRAW", "42")); G = int(os.environ.get("GRID", "24")); NT = int(os.environ.get("NT", "15"))
OUT.mkdir(parents=True, exist_ok=True); dev = "cuda" if torch.cuda.is_available() else "cpu"
def to_frame(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)): return np.asarray(Image.open(io.BytesIO(bytes(x))).convert("RGB"))
    x = np.asarray(x); return x.transpose(1, 2, 0) if (x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3) else x
with h5py.File(H5, "r") as h:
    epi = np.asarray(h["episode_idx"][:]).reshape(-1); stp = np.asarray(h["step_idx"][:]).reshape(-1)
    lo, hi, off, n = 16000, 18685, 25, 50
    ep_ids = np.unique(epi); ep_ids = ep_ids[(ep_ids >= lo) & (ep_ids < hi)]
    maxlen = {int(e): int(stp[epi == e].max()) for e in ep_ids}
    keep = np.isin(epi, ep_ids); mx = np.zeros(len(epi), dtype=np.int64); mx[keep] = np.array([maxlen[int(e)] for e in epi[keep]])
    valid = np.nonzero(keep & (stp <= mx - off))[0]
    rows = np.sort(np.random.default_rng(DRAW).choice(valid, size=n, replace=False))
    states = np.asarray(h["state"][:], dtype=np.float64); s_start = states[rows]; s_goal = states[rows + off]
    goal_frames = [to_frame(h["pixels"][int(r) + off]) for r in rows]
from rlp.core.value import load_metric
from rlp.core.world_model import load_pretrained
import stable_pretraining as spt
from torchvision.transforms import v2 as T
wm = load_pretrained(str(WMD)).to(dev).eval(); wm.requires_grad_(False)
ck = torch.load(ACTOR / "train_checkpoints" / "planner.pt", map_location="cpu", weights_only=False); W = int(ck.get("window_frames") or 1)
crit = {"ac": load_metric(str(ACTOR / "train_checkpoints" / "value_ac"), device=dev), "td": load_metric(str(ACTOR / "train_checkpoints" / "value_td"), device=dev)}
stats = spt.data.dataset_stats.ImageNet
tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=stats["mean"], std=stats["std"]), T.Resize(224)])
@torch.no_grad()
def encode(frames, bs=256):
    zs = []
    for i in range(0, len(frames), bs):
        px = torch.stack([tf(f) for f in frames[i:i + bs]]).to(dev).unsqueeze(1); zs.append(wm.encode({"pixels": px})["emb"][:, 0].float())
    return torch.cat(zs)
@torch.no_grad()
def V(nm, z, zg): return crit[nm](z.repeat(1, W), zg.repeat(1, W)).reshape(-1).cpu().numpy()
env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224).unwrapped; env.reset(seed=0)
xs = np.linspace(40, 472, G)
def render(st): env._set_state(st[:5]); return env.render()
zg_all = encode(goal_frames)
res = {"per_task": [], "W": W}
for e in range(NT):
    s0, g = s_start[e], s_goal[e]
    agent_sweep = []; block_sweep = []
    for y in xs:
        for x in xs:
            a = s0.copy(); a[0], a[1] = x, y; agent_sweep.append(render(a))
            b = s0.copy(); b[2], b[3] = x, y; block_sweep.append(render(b))
    cfgs = {"start": s0.copy(), "agent@goal_block@start": np.r_[g[0], g[1], s0[2], s0[3], s0[4]], "agent@start_block@goal": np.r_[s0[0], s0[1], g[2], g[3], g[4]], "goal_config": g.copy()}
    zA, zB = encode(agent_sweep), encode(block_sweep); zC = encode([render(v) for v in cfgs.values()])
    zg = zg_all[e:e + 1]
    # task geometry: a block that is already within tolerance of its goal pose makes a
    # low V(agent@goal, block@start) CORRECT, not an agent shortcut
    row = {"task": e, "block_move_px": float(np.linalg.norm(g[2:4] - s0[2:4])),
           "block_rot_deg": float(np.degrees(np.abs((g[4] - s0[4] + np.pi) % (2 * np.pi) - np.pi))),
           "agent_move_px": float(np.linalg.norm(g[:2] - s0[:2]))}
    for nm in ("ac", "td"):
        vA, vB = V(nm, zA, zg.expand(len(zA), -1)), V(nm, zB, zg.expand(len(zB), -1)); vC = V(nm, zC, zg.expand(len(zC), -1))
        row[nm] = {"agent_sweep_range": float(vA.max() - vA.min()), "agent_sweep_std": float(vA.std()), "block_sweep_range": float(vB.max() - vB.min()), "block_sweep_std": float(vB.std()),
                   **{f"V_{k}": float(v) for k, v in zip(cfgs, vC)}}
    lat = (zB - zg).norm(dim=-1).cpu().numpy(); latA = (zA - zg).norm(dim=-1).cpu().numpy()
    row["latent"] = {"agent_sweep_range": float(latA.max() - latA.min()), "block_sweep_range": float(lat.max() - lat.min())}
    res["per_task"].append(row)
    print(f"[e9] task {e}: ac agent-range {row['ac']['agent_sweep_range']:.1f} block-range {row['ac']['block_sweep_range']:.1f} | V start {row['ac']['V_start']:.1f} agent@goal {row['ac']['V_agent@goal_block@start']:.1f} block@goal {row['ac']['V_agent@start_block@goal']:.1f} goal {row['ac']['V_goal_config']:.1f} | td agent {row['td']['agent_sweep_range']:.1f} block {row['td']['block_sweep_range']:.1f}", flush=True)
agg = {nm: {k: float(np.mean([r[nm][k] for r in res["per_task"]])) for k in res["per_task"][0][nm]} for nm in ("ac", "td")}
agg["latent"] = {k: float(np.mean([r["latent"][k] for r in res["per_task"]])) for k in ("agent_sweep_range", "block_sweep_range")}
res["mean"] = agg
(OUT / "block_sensitivity.json").write_text(json.dumps(res, indent=1))
print("[e9] MEAN " + json.dumps(agg), flush=True)
# one figure: agent-xy vs block-xy heatmaps for the first 3 tasks (co-trained critic)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 2, figsize=(8, 11))
for e in range(3):
    s0, g = s_start[e], s_goal[e]; A = []; B = []
    for y in xs:
        for x in xs:
            a = s0.copy(); a[0], a[1] = x, y; A.append(render(a)); b = s0.copy(); b[2], b[3] = x, y; B.append(render(b))
    zg = zg_all[e:e + 1]; vA = V("ac", encode(A), zg.expand(G * G, -1)).reshape(G, G); vB = V("ac", encode(B), zg.expand(G * G, -1)).reshape(G, G)
    for a_, arr, ttl, mk in ((ax[e, 0], vA, "V vs AGENT xy (block fixed at start)", (s0[0], s0[1], g[0], g[1])), (ax[e, 1], vB, "V vs BLOCK xy (agent fixed at start)", (s0[2], s0[3], g[2], g[3]))):
        im = a_.imshow(arr, extent=[40, 472, 472, 40], cmap="viridis"); plt.colorbar(im, ax=a_, fraction=0.046)
        a_.plot(mk[0], mk[1], "wo", mec="k"); a_.plot(mk[2], mk[3], "w*", ms=11, mec="k"); a_.set_title(f"task {e}: {ttl}", fontsize=8); a_.set_xticks([]); a_.set_yticks([])
fig.suptitle("PushT co-trained critic: o = start, * = goal-frame position of the swept object", fontsize=9); fig.tight_layout(); fig.savefig(OUT / "e9_agent_vs_block.png", dpi=110)
print("[e9] DONE", flush=True)
