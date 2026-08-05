# OGBench Cube Offline Trajectory Examples

Generated from the converted OGBench visual cube HDF5 files in:

```text
/workspace/datasets/ogbench_swm/
```

Files:

- `ogbench_1cube_offline_trajectory.mp4`: sample from `visual-cube-single-play-v0`.
- `ogbench_2cube_offline_trajectory.mp4`: sample from `visual-cube-double-play-v0`.
- `ogbench_3cube_offline_trajectory.mp4`: sample from `visual-cube-triple-play-v0`.
- `ogbench_cube_1v2v3_side_by_side.mp4`: side-by-side comparison of the same three clips.
- `ogbench_cube_contact_sheet.png`: static frame grid for quick SSH/Codex inspection.
- `ogbench_*_offline_trajectory.gif`: animated GIF previews that do not need an MP4 player.
- `ogbench_cube_1v2v3_side_by_side.gif`: animated side-by-side preview.
- `index.html`: local browser gallery with PNG, GIF, and MP4 views.

For SSH port forwarding from a laptop, run this in the repo:

```bash
cd /workspace/src/Pantheon-SWM/stable-worldmodel/outputs/ogbench_cube_offline_trajectory_examples
python -m http.server 8765
```

Then connect with:

```bash
ssh -L 8765:localhost:8765 <host>
```

and open `http://localhost:8765/index.html` locally.

Training is intentionally paused; these are only offline dataset examples.
