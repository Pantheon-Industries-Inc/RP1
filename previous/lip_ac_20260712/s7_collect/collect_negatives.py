import numpy as np, h5py, sys, os, time
try: import hdf5plugin
except ImportError: pass
import gymnasium as gym
import stable_worldmodel

MODE = sys.argv[1]            # fail | drop
WID  = int(sys.argv[2])       # worker id
NW   = int(sys.argv[3])       # num workers
TARGET = int(sys.argv[4])     # clips this worker must bank
H5 = "/workspace/datasets/lewm_cube_full/cube_single_expert.h5"
OUT = "/workspace/datasets/lewm_cube_negatives/shard_%s_%02d.h5" % (MODE, WID)
LOG = open("/workspace/s7_collect/log_%s_%02d.txt" % (MODE, WID), "w", buffering=1)

h = h5py.File(H5, "r")
N, T = 10000, 201
con = h["proprio_gripper_contact"][:].reshape(N, T)
blkz = h["privileged_block_0_pos"][:].reshape(N, T, 3)[:, :, 2]
ep_off_all = h["ep_offset"][:]
# eval episodes (seeds 42/43/44) excluded from mining
ep_idx_col = h["ep_idx"][:]; step_col = h["step_idx"][:]; ep_len_col = h["ep_len"][:]
maxstart = ep_len_col[ep_idx_col] - 26
valid = np.nonzero(step_col <= maxstart)[0]
excl = set()
for seed in (42, 43, 44):
    g = np.random.default_rng(seed)
    rows = np.sort(valid[g.choice(len(valid) - 1, size=50, replace=False)])
    excl |= set(int(e) for e in ep_idx_col[rows])
c = (con[:, :, 0] > 0).astype(np.int8) if con.ndim == 3 else (con > 0).astype(np.int8)
events = []
for e in range(N):
    if e in excl: continue
    ce = c[e]
    for t in range(3, T - 40):
        if ce[t] == 1 and ce[t-1] == 0 and (blkz[e, t:t+15].max() - blkz[e, t]) >= 0.03:
            events.append((e, t))
            break                      # one event per episode = max diversity
events = [ev for i, ev in enumerate(events) if i % NW == WID]
rng = np.random.default_rng(1000 + WID)
rng.shuffle(events)
LOG.write("worker %d/%d mode=%s events_avail=%d target=%d excl_eps=%d\n" % (WID, NW, MODE, len(events), TARGET, len(excl)))

env = gym.make("swm/OGBCube-v0", env_type="single", ob_type="pixels", height=224, width=224, permute_blocks=False).unwrapped
CLIP = 65
def collect_one(e, t_grasp):
    off = int(ep_off_all[e])
    qpos = h["qpos"][off:off+T]; qvel = h["qvel"][off:off+T]; act = h["action"][off:off+T]
    t0 = max(int(t_grasp - rng.integers(10, 18)), 1)
    if np.isnan(act[t0:t0+CLIP]).any(): return None
    z0 = blkz[e, t0]
    kind = ""
    if MODE == "fail":
        r = rng.random()
        bias = rng.uniform(0.15, 0.45) * (1 if rng.random() < 0.5 else -1)
        bias2 = rng.uniform(0.15, 0.45) * (1 if rng.random() < 0.5 else -1)
        shift = int(rng.integers(2, 6)) * (1 if rng.random() < 0.5 else -1)
        if r < 0.4: kind = "xy"
        elif r < 0.6: kind = "zhigh"
        elif r < 0.8: kind = "timing"
        else: kind = "combo"
    else:
        kind = "drop"
        h_rel = rng.uniform(0.045, 0.15)
        keep_arm = rng.random() < 0.5
    env.reset(seed=int(1_000_000 + e))
    env.set_state(qpos[t0], qvel[t0])
    P, A, Q, V, B, C_, O = [], [], [], [], [], [], []
    released = False
    for i in range(CLIP):
        a = act[t0+i].astype(np.float64).copy()
        if MODE == "fail":
            if kind in ("xy", "combo"): a[0] += bias; a[1] += bias2
            if kind == "zhigh": a[2] += rng.uniform(0.2, 0.4)
            if kind in ("timing", "combo"):
                j = t0 + i + shift
                a[4] = act[min(max(j, 0), T-1)][4] if not np.isnan(act[min(max(j, 0), T-1)][4]) else a[4]
        else:
            zc = (B[-1][2] - z0) if B else 0.0
            if released or zc > h_rel:
                released = True
                a[4] = -1.0
                if not keep_arm: a[0]=a[1]=a[2]=a[3]=0.0
        a = np.clip(a, -1.0, 1.0)
        o, r_, te, tr, inf = env.step(a)
        P.append(o.astype(np.uint8)); A.append(a.astype(np.float32))
        Q.append(np.asarray(inf["prev_qpos"], dtype=np.float64)); V.append(np.asarray(inf["prev_qvel"], dtype=np.float64))
        B.append(np.asarray(inf["privileged/block_0_pos"], dtype=np.float64))
        C_.append(float(np.asarray(inf["proprio/gripper_contact"]).ravel()[0]))
        O.append(np.asarray(inf["proprio/gripper_opening"], dtype=np.float64).ravel())
    Bz = np.array([b[2] for b in B]) - z0
    if MODE == "fail":
        ok = Bz.max() < 0.03
        label = ("fail_" + kind) if ok else "success_perturbed"
    else:
        ok = released and Bz.max() >= 0.045 and Bz[-1] < 0.03
        label = "drop" if ok else "drop_reject"
    return dict(P=np.stack(P), A=np.stack(A), Q=np.stack(Q), V=np.stack(V),
                B=np.stack(B), C=np.array(C_), O=np.stack(O), label=label, ok=ok, e=e, t0=t0, kind=kind)

banked, extras, tried = [], [], 0
t_start = time.time()
for (e, t) in events:
    if len(banked) >= TARGET: break
    tried += 1
    try: r = collect_one(e, t)
    except Exception as ex:
        LOG.write("ERR ep %d: %r\n" % (e, ex)); continue
    if r is None: continue
    (banked if r["ok"] else extras).append(r)
    if tried % 20 == 0:
        LOG.write("tried=%d banked=%d extras=%d rate=%.1fs/clip\n" % (tried, len(banked), len(extras), (time.time()-t_start)/max(tried,1)))
keep_extras = extras[: max(len(banked)//5, 10)]
allc = banked + keep_extras
LOG.write("DONE banked=%d extras_kept=%d tried=%d\n" % (len(banked), len(keep_extras), tried))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with h5py.File(OUT, "w") as f:
    n_ep = len(allc); steps = sum(len(x["P"]) for x in allc)
    f.create_dataset("pixels", (steps,224,224,3), dtype="uint8", compression="gzip", compression_opts=2, chunks=(1,224,224,3))
    for nm, shp, dt in [("action",(steps,5),"f4"),("qpos",(steps,21),"f8"),("qvel",(steps,20),"f8"),
                        ("privileged_block_0_pos",(steps,3),"f8"),("proprio_gripper_contact",(steps,1),"f8"),
                        ("proprio_gripper_opening",(steps,1),"f8"),("ep_idx",(steps,),"i4"),("step_idx",(steps,),"i4")]:
        f.create_dataset(nm, shp, dtype=dt)
    f.create_dataset("ep_len", (n_ep,), dtype="i4"); f.create_dataset("ep_offset", (n_ep,), dtype="i8")
    f.create_dataset("label", (n_ep,), dtype=h5py.string_dtype())
    f.create_dataset("src_ep", (n_ep,), dtype="i4"); f.create_dataset("src_t0", (n_ep,), dtype="i4")
    row = 0
    for i, x in enumerate(allc):
        L = len(x["P"])
        f["pixels"][row:row+L] = x["P"]; f["action"][row:row+L] = x["A"]
        f["qpos"][row:row+L] = x["Q"]; f["qvel"][row:row+L] = x["V"]
        f["privileged_block_0_pos"][row:row+L] = x["B"]
        f["proprio_gripper_contact"][row:row+L] = x["C"].reshape(-1,1)
        f["proprio_gripper_opening"][row:row+L] = x["O"][:, :1]
        f["ep_idx"][row:row+L] = i; f["step_idx"][row:row+L] = np.arange(L)
        f["ep_len"][i] = L; f["ep_offset"][i] = row
        f["label"][i] = x["label"]; f["src_ep"][i] = x["e"]; f["src_t0"][i] = x["t0"]
        row += L
LOG.write("WROTE %s eps=%d steps=%d\n" % (OUT, n_ep, row))
