"""Make dataset-driven eval resets deterministic.

RECONSTRUCTED 2026-07-30 after losing volume 3cv5zzezm9 (verbatim from the
copy that ran on the original pod).

Root cause, measured on 2026-07-28: `_evaluate_from_dataset` calls
`self.reset(seed=init_state.get('seed'))`, the canonical reacher h5 has no
`seed` column, so every eval resets with seed=None. With seed=None the
dm_control task RNG is left unseeded and each sub-env draws its own target-ball
position -- a salient red geom, mean 0.20 m from the dataset episode's ball
(arm reach is 0.24 m) -- differently on EVERY invocation. The same command at
the same cfg.seed returned held-at-end 68/70/76/80/82 over five runs, with the
task draw proven identical (5/50 episodes flipped outcome).

Fix: when the dataset provides no seed, derive one per env from the drawn
(episode, start) pair. The ball becomes a deterministic property of the TASK,
not of the run: the same episode/start always renders the same scene, across
runs, seeds, and arms. Distribution is unchanged (still a random ball per task,
exactly as the authors' protocol has it) -- only run-to-run jitter is removed.
Verified 74.0 x3 bit-repeatable after the fix.
"""

P = "/workspace/swm_cem/stable_worldmodel/world/world.py"
s = open(P).read()
assert "deterministic property of the TASK" not in s, "already patched"

OLD = """        self.reset(seed=init_state.get('seed'))

        if callables:"""
NEW = """        _seeds = init_state.get('seed')
        if _seeds is None:
            # No seed column in the dataset => seed=None => each sub-env draws
            # task randomness (reacher: the target-ball position, a salient red
            # geom in every frame) from an UNSEEDED RNG, so identical commands
            # returned different success rates (measured 68-82 held-at-end over
            # five invocations of one cell). Derive the seed from the drawn
            # (episode, start) pair instead: scene randomness becomes a
            # deterministic property of the TASK, stable across runs and arms,
            # while the across-task distribution stays exactly as before.
            _seeds = [
                int((1_000_003 * int(e) + int(st)) % 2_147_483_647)
                for e, st in zip(episodes_idx, start_steps)
            ]
        self.reset(seed=_seeds)

        if callables:"""
assert s.count(OLD) == 1, f"anchor x{s.count(OLD)}"
s = s.replace(OLD, NEW)
open(P, "w").write(s)
print("patched world.py: deterministic per-task reset seeds")
