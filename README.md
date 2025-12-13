# MuJoCo Simulation Package

This directory contains Python scripts for controlling MuJoCo simulations and associated XML model files.

## Files

### `double_revolute.py`
A Python script that loads a double revolute joint model and applies a PD controller to reach a target configuration.

- **Controller**: PD Control (`Kp=50`, `Kd=25`).
- **Target**: `[pi/2, -pi/2]` (approx `[1.57, -1.57]` rad).
- **Output**: 
  - Generates an MP4 video in the `videos/` directory by default.
  - Displays a Matplotlib plot of joint trajectories vs targets after execution.
- **Usage**:
  ```bash
  python3 double_revolute.py
  ```
  *Note: You can toggle the `GENERATE_VIDEO` boolean inside the script to switch between headless video rendering and the interactive passive viewer.*

### `xmls/`
Directory containing MuJoCo MJCF (XML) model files.

- **`kondo_scene_flat_ground.xml`**: 
  - Scene setup for a Kondo humanoid robot on flat ground.
  - Includes `kondo_6.xml`.
  - Defines keyframes for various poses: `stand`, `stand_2`, `squat`, and `stand_hands_folded`.

## Requirements

The following Python libraries are required:
- `mujoco`
- `numpy`
- `opencv-python` (cv2)
- `matplotlib`

## Setup

The scripts are currently configured with a specific shebang:
`#!/home/nandhith/humanoid_venv/bin/python3`

Ensure this virtual environment exists, or run the script explicitly with your preferred Python interpreter:
`python3 double_revolute.py`
