# Humanoid MuJoCo Examples

Collection of small MuJoCo simulations with consistent video-generation utilities. Each script writes an MP4 to `videos/` (created automatically) and plots key signals.

## Requirements
- Python 3.10+
- Packages: `mujoco`, `numpy`, `opencv-python`, `matplotlib`
- Offscreen rendering: each XML already sets `<visual><global offwidth="1280" offheight="720"/></visual>` to match 1280×720 video.

Install deps (example):
```bash
pip install mujoco numpy opencv-python matplotlib
```

## Scripts
- `reaction_wheel.py`
  - PD control of a reaction wheel rod.
  - Generates `videos/reaction_wheel.mp4` and plots torque/angle.
  - Optional `camera_settings` dict (lookat, distance, azimuth, elevation) overrides the XML camera `cam1`.

- `cartpole.py`
  - PD control of a cart-pole.
  - Generates `videos/cartpole.mp4` and plots force/angle.
  - Uses XML camera `fixed` (offscreen size set in XML).

- `double_revolute.py`
  - PD control of a 2-DOF arm (two hinge joints).
  - Generates `videos/double_revolute.mp4` and plots joint angles vs targets.
  - Accepts optional `camera_settings` (lookat, distance, azimuth, elevation); default named camera `fixed` if none provided.

- `double_pendulum.py`
  - Passive double pendulum with selectable integrator (defaults RK4 in `generate_video`).
  - Generates `videos/double_pendulum.mp4` and plots angles, velocities, total energy.
  - Supports optional `camera_settings` (lookat, distance, azimuth, elevation); default named camera `fixed` if none provided.

- `linear_inverted_pendulum.py`
  - Generates multiple videos with different force modes (`free_fall`, `radial`, `mg`, `normal`) using camera `fixed`.

- `load_unitree.py`, `model_loader.py`, `LIP_play_videos_grid.py`, `linear_inverted_pendulum.py`, etc.
  - Utilities/demos for loading MJCFs and combining outputs (see source for specifics).

## Usage
Run any script from the repo root, e.g.:
```bash
/usr/bin/python3 reaction_wheel.py
/usr/bin/python3 cartpole.py
/usr/bin/python3 double_revolute.py
/usr/bin/python3 double_pendulum.py
/usr/bin/python3 linear_inverted_pendulum.py
```
Videos will appear under `videos/`. Adjust `camera_settings` in the script if you want custom views:
```python
camera_settings={
    "lookat": [0.0, 0.0, 0.5],
    "distance": 2.5,
    "azimuth": 90,
    "elevation": -20,
}
```

## Notes
- If you change video resolution, also update the XML `<global offwidth/offheight>` to avoid framebuffer errors.
- Live viewer paths remain available by toggling the `GENERATE_VIDEO` flag inside each script.
