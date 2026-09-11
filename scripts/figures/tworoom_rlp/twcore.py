"""Core harness: load TwoRoom WM + LIP actor + value, render, refine, probe."""
from __future__ import annotations
import os, sys, math
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import torch
import gymnasium as gym
import stable_worldmodel  # registers swm/* envs  # noqa

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

BASES = {
    "lejepa": dict(
        wm="assets/core/world_model/lejepa_tworoom",
        planner="assets/core/planner/tworoom/lejepa/terminal_a1.8_m0.1_l1e-4",
    ),
    "pldm": dict(
        wm="assets/core/world_model/pldm_tworoom",
        planner="assets/core/planner/tworoom/pldm/terminal_a1.8_m0.0_l1e-3",
    ),
}

def make_env():
    env = gym.make("swm/TwoRoom-v1")
    env.reset(seed=0)
    return env.unwrapped

def frame(u, pos):
    """Render (3,224,224) uint8 with agent at pos (x,y). No target drawn."""
    import torch as T
    f = u._render_frame(T.as_tensor([float(pos[0]), float(pos[1])], dtype=T.float32))
    return f  # uint8 (3,H,W)

def norm_batch(frames_u8: torch.Tensor) -> torch.Tensor:
    x = frames_u8.float()/255.0
    return (x - IMAGENET_MEAN)/IMAGENET_STD

def load_stack(base_key, seed, device="cpu"):
    from rlp.core.world_model import load_pretrained
    from rlp.core.value import load_metric
    from rlp.core.planner import PlannerNet
    import torch as T
    cfg = BASES[base_key]
    wm = load_pretrained(cfg["wm"]).to(device).eval()
    wm.requires_grad_(False)
    if hasattr(wm, "interpolate_pos_encoding"):
        wm.interpolate_pos_encoding = True
    pdir = f"{cfg['planner']}/s{seed}"
    ck = T.load(f"{pdir}/planner.pt", map_location=device, weights_only=False)
    actor = PlannerNet(ck["z_dim"], horizon=ck["horizon"], a_dim=ck["a_dim"],
                       amax=ck["amax"], feed=ck.get("feed","none"),
                       use_zg=ck.get("use_zg",False), use_gate=ck.get("use_gate",False),
                       use_z0=ck.get("use_z0",False), use_grad=ck.get("use_grad",True)).to(device)
    actor.load_state_dict(ck["sd"]); actor.eval()
    # value: sibling dir named by recorded stem
    import glob
    vdirs = glob.glob(f"{pdir}/g_*_value")
    assert vdirs, f"no value dir in {pdir}"
    value = load_metric(vdirs[0], device=device).eval()
    meta = dict(z_dim=ck["z_dim"], horizon=ck["horizon"], a_dim=ck["a_dim"],
                amax=ck["amax"], iters=ck["iters"], objective=ck.get("temporal_objective","terminal"))
    return wm, actor, value, meta

@torch.no_grad()
def encode_pos(wm, u, positions, device="cpu", bs=64):
    """positions: (N,2) -> latent emb last frame (N,192)."""
    outs=[]
    positions=np.asarray(positions, dtype=np.float32).reshape(-1,2)
    for i in range(0, len(positions), bs):
        chunk = positions[i:i+bs]
        fr = torch.stack([frame(u,p) for p in chunk])  # (b,3,H,W)
        px = norm_batch(fr).to(device)                 # (b,3,H,W)
        px = px.unsqueeze(1)                           # (b,1,3,H,W) single-frame "history"
        emb = wm.encode({"pixels": px})["emb"]         # (b,T,D)
        outs.append(emb[:, -1].float().cpu())
    return torch.cat(outs, 0)

# ---------------- geometry / geodesic ----------------
def occupancy(u):
    """(H,W) bool: True = blocked (wall/border, door cut out)."""
    wall_mask, door_mask = u._wall_and_door_masks()
    return wall_mask.cpu().numpy().astype(bool)

def free_mask(u):
    return ~occupancy(u)

def geodesic_grid(u, goal_xy, step=4):
    """BFS shortest-path distance (px) from every free cell to goal, respecting walls.
    Returns (grid_x, grid_y, D) on a subsampled lattice; D=nan in walls/unreachable."""
    from collections import deque
    occ = occupancy(u)              # (H,W) True=blocked, indexed [y,x]
    H, W = occ.shape
    gx, gy = int(round(goal_xy[0])), int(round(goal_xy[1]))
    # nearest free cell to goal
    if occ[gy, gx]:
        ys, xs = np.where(~occ)
        i = np.argmin((xs-gx)**2 + (ys-gy)**2); gx, gy = int(xs[i]), int(ys[i])
    INF = np.inf
    dist = np.full((H, W), INF, np.float64)
    dist[gy, gx] = 0.0
    dq = deque([(gx, gy)])
    # 8-connected, euclidean step cost
    nbrs = [(-1,0,1),(1,0,1),(0,-1,1),(0,1,1),(-1,-1,1.4142),(1,-1,1.4142),(-1,1,1.4142),(1,1,1.4142)]
    while dq:
        cx, cy = dq.popleft()
        base = dist[cy, cx]
        for dx, dy, c in nbrs:
            nx, ny = cx+dx, cy+dy
            if 0<=nx<W and 0<=ny<H and not occ[ny,nx] and dist[ny,nx] > base + c:
                dist[ny,nx] = base + c
                dq.append((nx,ny))
    # relax a few passes (BFS with weights isn't exact Dijkstra; iterate)
    for _ in range(3):
        changed=False
        ys,xs=np.where(np.isfinite(dist))
        order=np.argsort(dist[ys,xs])
        for idx in order:
            cx,cy=xs[idx],ys[idx]; base=dist[cy,cx]
            for dx,dy,c in nbrs:
                nx,ny=cx+dx,cy+dy
                if 0<=nx<W and 0<=ny<H and not occ[ny,nx] and dist[ny,nx]>base+c:
                    dist[ny,nx]=base+c; changed=True
        if not changed: break
    dist[~np.isfinite(dist)] = np.nan
    return dist  # (H,W) px, nan in walls

# ---------------- latent<->position probe ----------------
def fit_probe(wm, u, device="cpu", n=26):
    """Ridge probe latent(192)->(x,y). Returns predict fn + R2."""
    bs = np.linspace(u.BORDER_SIZE+6, u.IMG_SIZE-u.BORDER_SIZE-6, n)
    XX, YY = np.meshgrid(bs, bs)
    pos = np.stack([XX.ravel(), YY.ravel()],1)
    occ = occupancy(u)
    keep = ~occ[np.clip(pos[:,1].astype(int),0,223), np.clip(pos[:,0].astype(int),0,223)]
    pos = pos[keep]
    Z = encode_pos(wm, u, pos, device=device).numpy()
    P = pos.astype(np.float64)
    Zc = Z - Z.mean(0, keepdims=True); Pc = P - P.mean(0, keepdims=True)
    lam = 1e-1
    W_ = np.linalg.solve(Zc.T@Zc + lam*np.eye(Z.shape[1]), Zc.T@Pc)  # (192,2)
    zmean = Z.mean(0); pmean = P.mean(0)
    pred = (Z - zmean)@W_ + pmean
    ss_res = ((pred-P)**2).sum(0); ss_tot = ((P-P.mean(0))**2).sum(0)
    r2 = 1 - ss_res/ss_tot
    def predict(Zq):
        Zq = np.asarray(Zq)
        return (Zq - zmean)@W_ + pmean
    return predict, r2, pos, Z

# ---------------- refinement ----------------
@torch.no_grad()
def _encode_hist_goal(wm, u, start_xy, goal_xy, device="cpu"):
    zs = encode_pos(wm, u, [start_xy], device=device).to(device)   # (1,D)
    zg = encode_pos(wm, u, [goal_xy], device=device).to(device)    # (1,D)
    z_hist = zs.unsqueeze(1).expand(-1,3,-1).contiguous()          # (1,3,D)
    return z_hist, zg

def refine(wm, actor, value, meta, u, start_xy, goal_xy, device="cpu"):
    """Run lip4 refinement K=iters. Return list of imagined latent trajs per iter (K+1,H,D)
    including iter0 (A=0) and the plan A per iter."""
    from rlp.core.rollout import rollout_traj
    from rlp.core.temporal import trajectory_value
    H, adim, amax, K = meta["horizon"], meta["a_dim"], meta["amax"], meta["iters"]
    z_hist, zg = _encode_hist_goal(wm, u, start_xy, goal_xy, device=device)
    a_hist = torch.zeros(1,2,adim, device=device)
    z0 = z_hist[:,-1]
    A = torch.zeros(1,H,adim, device=device)
    trajs=[]; plans=[]; values=[]
    def imagined(Acur):
        return rollout_traj(wm, z_hist, a_hist, Acur)   # (1,H,D)
    # record iter0 state (A=0)
    tr0 = imagined(A); trajs.append(tr0.detach()); plans.append(A.detach().clone())
    values.append(float(trajectory_value(value, tr0, zg, z0, meta["objective"]).item()))
    for k in range(K):
        with torch.enable_grad():
            A_in = A.detach().requires_grad_(True)
            traj = rollout_traj(wm, z_hist, a_hist, A_in)
            score = trajectory_value(value, traj, zg, z0, meta["objective"])
            (gA,) = torch.autograd.grad(score.sum(), A_in)
        with torch.no_grad():
            traj_f = traj.detach()
            E = trajectory_value(value, traj_f, zg, z0, meta["objective"])
            A = actor(A, gA, E, z0, zg, traj_f, k=k)
        tr = imagined(A)
        trajs.append(tr.detach()); plans.append(A.detach().clone())
        values.append(float(trajectory_value(value, tr, zg, z0, meta["objective"]).item()))
    return dict(trajs=trajs, plans=plans, values=values, z_hist=z_hist.detach(),
                zg=zg.detach(), z0=z0.detach())

# ---------------- protocol-correct instances (goal = state @ offset) ----------------
def expert_instance(seed, offset=25, device="cpu"):
    """Reproduce the eval's (origin, goal) construction: roll the ExpertPolicy
    from env.reset(seed) and take origin=state[0], goal=state[offset].
    Returns origin, goal, full expert path (T,2), actual offset used."""
    import gymnasium as gym, numpy as np
    from stable_worldmodel.envs.two_room import ExpertPolicy
    env = gym.make("swm/TwoRoom-v1")
    env.reset(seed=seed); u = env.unwrapped
    pol = ExpertPolicy(action_noise=0.0)          # deterministic route for a clean goal
    pol.set_env(env)
    states=[u.agent_position.numpy().copy()]
    for t in range(offset+40):
        info={"state": u.agent_position.numpy()[None], "goal_state": u.target_position.numpy()[None]}
        try:
            a = pol.get_action(info)
        except Exception:
            a = np.zeros((1,2), np.float32)
        a = np.asarray(a, np.float32).reshape(-1)[:2]
        obs, r, term, trunc, info2 = env.step(a)
        states.append(u.agent_position.numpy().copy())
        if term or trunc: break
    states=np.array(states)
    off=min(offset, len(states)-1)
    return states[0].copy(), states[off].copy(), states, off

def expert_instance_seed(draw_seed, offset=25):
    """Same as expert_instance but keyed by an explicit draw seed."""
    return expert_instance(draw_seed, offset=offset)

# ---------------- action stats + true-dynamics execution ----------------
def collect_action_stats(n_expert_ep=60, n_random_ep=60, seed=7):
    """Fit action StandardScaler the way the eval does (expert noise 2.0 + random)."""
    import gymnasium as gym, numpy as np
    from stable_worldmodel.envs.two_room import ExpertPolicy
    rng=np.random.default_rng(seed)
    acts=[]
    env=gym.make("swm/TwoRoom-v1")
    pol=ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05); pol.set_env(env)
    for ep in range(n_expert_ep):
        env.reset(seed=int(rng.integers(0,1_000_000))); u=env.unwrapped
        for t in range(60):
            info={"state":u.agent_position.numpy()[None],"goal_state":u.target_position.numpy()[None]}
            a=np.asarray(pol.get_action(info),np.float32).reshape(-1)[:2]
            acts.append(a.copy()); obs,r,term,trunc,_=env.step(a)
            if term or trunc: break
    for ep in range(n_random_ep):
        env.reset(seed=int(rng.integers(0,1_000_000))); u=env.unwrapped
        for t in range(40):
            a=rng.uniform(-1,1,size=2).astype(np.float32)
            acts.append(a.copy()); obs,r,term,trunc,_=env.step(a)
            if term or trunc: break
    A=np.array(acts,np.float32)
    return A.mean(0), A.std(0)+1e-8, len(A)

def execute_plan(u, origin, goal, A_z, amean, astd, action_block=5):
    """Un-z-score plan and roll it through TRUE env dynamics (with collisions).
    A_z: (H, a_dim) z-scored plan. Returns executed path (T+1, 2) respecting walls."""
    import torch as T, numpy as np
    # set true start/goal on the env
    u.agent_position = T.as_tensor(np.asarray(origin,np.float32))
    u.target_position = T.as_tensor(np.asarray(goal,np.float32))
    A = np.asarray(A_z, np.float32).reshape(-1, 2)          # (H*block, 2) primitive actions
    A = A*astd[None,:] + amean[None,:]                      # un-z-score
    A = np.clip(A, -1.0, 1.0)
    path=[np.asarray(origin,np.float32).copy()]
    for a in A:
        u.step(a.astype(np.float32))
        path.append(u.agent_position.numpy().copy())
    return np.array(path)

# ---------------- Adam / CEM refinement on the SAME value function ----------------
def refine_adam(wm, value, meta, u, start_xy, goal_xy, n_steps=30, lr=0.1, device="cpu"):
    """AdamW gradient descent on V(z_T(A), z_g); single plan from A0=0. Captures A per step."""
    from rlp.core.rollout import rollout_traj
    from rlp.core.temporal import trajectory_value
    H, adim, amax = meta["horizon"], meta["a_dim"], meta["amax"]
    z_hist, zg = _encode_hist_goal(wm, u, start_xy, goal_xy, device=device)
    a_hist = torch.zeros(1,2,adim, device=device); z0=z_hist[:,-1]
    A = torch.zeros(1,H,adim, device=device, requires_grad=True)
    opt = torch.optim.AdamW([A], lr=lr)
    def val(Ad):
        with torch.no_grad():
            tr=rollout_traj(wm,z_hist,a_hist,Ad); return float(trajectory_value(value,tr,zg,z0,meta["objective"]).item())
    plans=[A.detach().clone()]; values=[val(A.detach())]
    for i in range(n_steps):
        opt.zero_grad()
        traj = rollout_traj(wm, z_hist, a_hist, A)
        score = trajectory_value(value, traj, zg, z0, meta["objective"]).sum()
        score.backward(); opt.step()
        with torch.no_grad(): A.clamp_(-amax, amax)
        plans.append(A.detach().clone()); values.append(val(A.detach()))
    return dict(plans=plans, values=values)

def refine_cem(wm, value, meta, u, start_xy, goal_xy, n_steps=30, num_samples=300,
               topk=30, var_scale=1.0, seed=0, device="cpu"):
    """CEM on V(z_T(A), z_g) (matches swm convention). Captures elite-mean per iter."""
    from rlp.core.rollout import rollout_traj
    from rlp.core.temporal import trajectory_value
    H, adim = meta["horizon"], meta["a_dim"]
    z_hist, zg = _encode_hist_goal(wm, u, start_xy, goal_xy, device=device)
    a_hist = torch.zeros(1,2,adim, device=device); z0=z_hist[:,-1]
    g=torch.Generator(device=device).manual_seed(seed)
    mean=torch.zeros(H,adim,device=device); std=var_scale*torch.ones(H,adim,device=device)
    def val(m):
        with torch.no_grad():
            tr=rollout_traj(wm,z_hist,a_hist,m.unsqueeze(0)); return float(trajectory_value(value,tr,zg,z0,meta["objective"]).item())
    plans=[mean.clone().unsqueeze(0)]; values=[val(mean)]
    zhN=z_hist.expand(num_samples,-1,-1); ahN=a_hist.expand(num_samples,-1,-1); zgN=zg.expand(num_samples,-1)
    for i in range(n_steps):
        cand = torch.randn(num_samples,H,adim,generator=g,device=device)*std.unsqueeze(0)+mean.unsqueeze(0)
        cand[0]=mean
        with torch.no_grad():
            traj=rollout_traj(wm,zhN,ahN,cand)
            costs=trajectory_value(value,traj,zgN,zhN[:,-1],meta["objective"])   # (N,)
        elite=cand[torch.argsort(costs)[:topk]]
        mean=elite.mean(0); std=elite.std(0)
        plans.append(mean.clone().unsqueeze(0)); values.append(val(mean))
    return dict(plans=plans, values=values)
