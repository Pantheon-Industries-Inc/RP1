"""Add SWM_RECORD_PATH on-policy recording to World._evaluate_from_dataset.

Anchored string replacements on stable_worldmodel/world/world.py; fails loudly
if an anchor is missing/ambiguous. Idempotent: skips if already patched.
Recording mirrors World.collect()'s conventions (per-env buffers, length-1
time-dim squeeze, action rotation, lance writer) so recorded datasets match
the mixture collector's format. Episodes shorter than 25 steps are dropped
(too short for a training window / one plan horizon).
"""
from pathlib import Path

p = Path("/workspace/code/stable-worldmodel/stable_worldmodel/world/world.py")
t = p.read_text()

if "SWM_RECORD_PATH" in t:
    print("already patched — nothing to do")
    raise SystemExit(0)

a1 = (
    "        goal_snapshot = {k: self.infos[k].copy() for k in goal_state}\n"
    "\n"
    "        results = {"
)
n1 = """        goal_snapshot = {k: self.infos[k].copy() for k in goal_state}

        # SWM_RECORD_PATH: optionally record (pixels, action, qpos, qvel) per
        # step to a lance dataset -- on-policy collection through the eval
        # path (goal-conditioned, unlike collect()). Mirrors collect()'s
        # buffering, squeeze and action-rotation conventions.
        import os as _os
        _rec_path = _os.environ.get('SWM_RECORD_PATH')
        _rec_bufs = None
        if _rec_path:
            _rec_cols = ('pixels', 'action', 'qpos', 'qvel')
            _rec_bufs = [defaultdict(list) for _ in range(n)]
            _rec_done = np.zeros(n, dtype=bool)

        results = {"""

a2 = (
    "            world.infos.update(deepcopy(goal_snapshot))\n"
    "            results['episode_successes'] |= world.terminateds"
)
n2 = """            world.infos.update(deepcopy(goal_snapshot))
            if _rec_bufs is not None:
                for _col in _rec_cols:
                    if _col not in world.infos:
                        continue
                    _d = world.infos[_col]
                    if not isinstance(_d, (np.ndarray, torch.Tensor)):
                        continue
                    if _d.ndim > 1 and _d.shape[1] == 1:
                        _d = (_d.squeeze(1) if isinstance(_d, torch.Tensor)
                              else np.squeeze(_d, axis=1))
                    for _i in range(n):
                        if _rec_done[_i]:
                            continue
                        _v = _d[_i]
                        _v = (_v.detach().cpu().numpy()
                              if isinstance(_v, torch.Tensor) else _v.copy())
                        _rec_bufs[_i][_col].append(_v)
                _rec_done[:] = _rec_done | world.terminateds | world.truncateds
            results['episode_successes'] |= world.terminateds"""

a3 = (
    "        self._run(max_steps=eval_budget, mode=mode, on_step=on_step)\n"
    "\n"
    "        results['success_rate'] = ("
)
n3 = """        self._run(max_steps=eval_budget, mode=mode, on_step=on_step)

        if _rec_bufs is not None:
            from stable_worldmodel.data.format import get_format as _get_format

            _MIN_LEN = 25  # drop episodes shorter than one plan horizon
            _stats = {'kept': 0, 'dropped': 0}

            def _rec_iter():
                for _i in range(n):
                    _ep = {k: list(v) for k, v in _rec_bufs[_i].items()}
                    if not _ep or len(_ep.get('action', ())) < _MIN_LEN:
                        _stats['dropped'] += 1
                        continue
                    _ep['action'].append(_ep['action'].pop(0))
                    _stats['kept'] += 1
                    yield _ep

            with _get_format('lance').open_writer(_rec_path) as _w:
                _w.write_episodes(_rec_iter())
            print('[record] kept=%(kept)d dropped=%(dropped)d' % _stats
                  + ' -> ' + _rec_path, flush=True)

        results['success_rate'] = ("""

for name, a in (("a1", a1), ("a2", a2), ("a3", a3)):
    c = t.count(a)
    print(name, "count:", c)
    assert c == 1, f"{name} anchor not unique/missing"

t = t.replace(a1, n1).replace(a2, n2).replace(a3, n3)
p.write_text(t)
print("world.py patched OK")
