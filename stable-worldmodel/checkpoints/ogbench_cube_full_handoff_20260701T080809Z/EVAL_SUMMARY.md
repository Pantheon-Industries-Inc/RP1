# OGBench Cube Planning Eval Summary

- Updated: 2026-07-02T02:55:00Z
- Completed: 9/12
- Solvers: `cem`, `adam` (`adam` is the repo's `GradientSolver` / GD-style planner)
- LeWM checkpoint step: `40000`

| Variant | Phase | Solver | Status | Success Rate | Eval Time (s) |
| --- | --- | --- | --- | ---: | ---: |
| single | dino | cem | done | 84.0 | 2726.181816339493 |
| single | dino | adam | done | 62.0 | 2747.3841321468353 |
| single | lewm | cem | done | 72.0 | 172.36901783943176 |
| single | lewm | adam | done | 62.0 | 212.01403045654297 |
| double | dino | cem | done | 70.0 | 2966.730549812317 |
| double | dino | adam | pending |  |  |
| double | lewm | cem | pending |  |  |
| double | lewm | adam | pending |  |  |
| triple | dino | cem | done | 76.0 | 2887.0365102291107 |
| triple | dino | adam | done | 64.0 | 2666.900864839554 |
| triple | lewm | cem | done | 64.0 | 180.5164875984192 |
| triple | lewm | adam | done | 54.0 | 222.59298539161682 |
