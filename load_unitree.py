import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the scene which includes the robot and a floor
model = mujoco.MjModel.from_xml_path("unitree_g1/scene_mjx.xml")
data = mujoco.MjData(model)

# --- Use the stable "home" keyframe as the target ---
# The "home" keyframe is a pre-defined, stable standing pose.
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
if key_id == -1:
    raise ValueError("Keyframe 'home' not found in the model. Check scene_mjx.xml.")
standing_target_qpos = model.key_qpos[key_id][7:]

# Launch the passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set the camera to the predefined "track" camera in the XML
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    start_time = time.time()
    while viewer.is_running() and time.time() - start_time < 30:
        step_start = time.time()

        # Set the control signal to the target joint positions from the "home" keyframe.
        data.ctrl[:] = standing_target_qpos

        # Step the simulation
        mujoco.mj_step(model, data)
        viewer.sync()

        # Maintain real-time pacing
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))