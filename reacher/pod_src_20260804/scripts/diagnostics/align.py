import hdf5plugin, h5py, numpy as np
import stable_worldmodel as swm
with h5py.File("/workspace/datasets_canon/lewm-reacher/reacher.h5","r") as f:
    cq = f["qpos"][:8]; ca = f["action"][:8]
ds = swm.data.load_dataset("dmc/reacher_random.lance")
oq = np.asarray(ds.get_col_data("qpos"))[:8]
oa = np.asarray(ds.get_col_data("action"))[:8]
print("canon qpos[0:6]\n", cq[:6])
print("ours  qpos[0:6]\n", oq[:6])
print("canon action[0:6]\n", ca[:6])
print("ours  action[0:6]\n", oa[:6])
# our qpos[k] == canon qpos[k+1]; is our action[k] == canon action[k+1] (consistent)
# or == canon action[k] (misaligned by one)?
print("max|ours_qpos[0:5] - canon_qpos[1:6]| =", np.abs(oq[:5]-cq[1:6]).max())
print("max|ours_act[0:5]  - canon_act[1:6] | =", np.abs(oa[:5]-ca[1:6]).max(), "  <- 0 => pairing CONSISTENT")
print("max|ours_act[0:5]  - canon_act[0:5] | =", np.abs(oa[:5]-ca[:5]).max(), "  <- 0 => OFF BY ONE")
