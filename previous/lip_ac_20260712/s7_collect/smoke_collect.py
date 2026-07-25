import numpy as np, h5py
try: import hdf5plugin
except ImportError: pass
import gymnasium as gym
import stable_worldmodel  # registers envs
h = h5py.File("/workspace/datasets/lewm_cube_full/cube_single_expert.h5","r")
N, T = 10000, 201
EP = 638
off = int(h["ep_offset"][EP])
qpos = h["qpos"][off:off+T]; qvel = h["qvel"][off:off+T]
act  = h["action"][off:off+T]
blk  = h["privileged_block_0_pos"][off:off+T]
con  = h["proprio_gripper_contact"][off:off+T]
pix0 = h["pixels"][off+20]
c = (con[:,0] > 0).astype(int)
acqs = [t for t in range(1, T-20) if c[t]==1 and c[t-1]==0 and blk[t:t+15,2].max()-blk[t,2] >= 0.03]
t_grasp = acqs[0]; t0 = max(t_grasp-15, 1)
print("episode", EP, "grasp acquisition at t=", t_grasp, "replay from t0=", t0)
env = gym.make("swm/OGBCube-v0", env_type="single", ob_type="pixels", height=224, width=224, permute_blocks=False).unwrapped
obs, info = env.reset(seed=0)
print("info keys sample:", sorted(k for k in info.keys())[:8])
def get_blk(info): return np.asarray(info["privileged/block_0_pos"])
def replay(perturb, steps=45):
    env.reset(seed=0)
    env.set_state(qpos[t0], qvel[t0])
    trace, contact = [], []
    o = None
    for i in range(steps):
        a = act[t0+i].astype(np.float64).copy()
        if np.isnan(a).any(): break
        a = perturb(i, a, trace)
        o, r, te, tr, inf = env.step(a)
        trace.append(get_blk(inf)); contact.append(float(np.asarray(inf["proprio/gripper_contact"]).ravel()[0]))
    return np.array(trace), np.array(contact), o
# 1) fidelity: unperturbed replay
tr, co, o = replay(lambda i,a,t: a)
ref = blk[t0+1:t0+1+len(tr)]
print("FIDELITY: steps=%d  max|blk_sim - blk_data|=%.4f  final dz sim=%.3f data=%.3f" % (len(tr), np.abs(tr-ref[:len(tr)]).max(), tr[-1,2]-blk[t0,2], ref[len(tr)-1,2]-blk[t0,2]))
print("render check: obs shape", None if o is None else o.shape, " pixel-MSE vs dataset t0+20 frame:", float(((o.astype(np.float32)-pix0.astype(np.float32))**2).mean()) if o is not None else "NA")
# 2) grasp-fail recipe: xy bias during approach
def fail_pert(i, a, t):
    a2 = a.copy(); a2[0] += 0.35; a2[1] -= 0.35; return a2
tr2, co2, _ = replay(fail_pert)
print("FAIL-RECIPE: max lift=%.3f (want < 0.03)  contact frames=%d" % (tr2[:,2].max()-blk[t0,2], int((co2>0).sum())))
# 3) drop recipe: faithful until airborne, then force open
def drop_pert(i, a, t):
    if len(t) and (t[-1][2] - blk[t0,2]) > 0.05:
        a2 = a.copy(); a2[4] = -1.0; return a2
    return a
tr3, co3, _ = replay(drop_pert, steps=60)
zr = tr3[:,2] - blk[t0,2]
print("DROP-RECIPE: max lift=%.3f  final height=%.3f (want: rose >0.05 then fell <0.03)" % (zr.max(), zr[-1]))
