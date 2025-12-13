#!/home/nandhith/humanoid_venv/bin/python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt
try:
    from .mujoco_ros_utils import MujocoRosConnector
except ImportError:
    from mujoco_ros_utils import MujocoRosConnector

 
def generate_video(model, filename, force_mode, duration=2.0, rHz=60):
    """Generates a video of the simulation with a specific force model."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Video and rendering parameters
    width, height = 1280, 720
    cam_name = "fixed"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, rHz, (width, height))
    scene_option = mujoco.MjvOption()
    rFrtime = 1.0 / rHz

    # Get the mass of the slider body ('link2')
    slider_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link2')
    slider_mass = model.body_mass[slider_body_id]
    gravity = np.linalg.norm(model.opt.gravity)
    
    print(f"Generating video '{filename}' with force mode: '{force_mode}'...")
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        while data.time < duration:
            simstart = data.time
            while (data.time - simstart) < rFrtime:
                # --- Force Calculation based on mode ---
                force = 0.0
                if force_mode == 'mg':
                    force = slider_mass * gravity
                elif force_mode == 'radial':
                    theta = data.qpos[0]
                    slider_pos = data.qpos[1]
                    thetadot = data.qvel[0]
                    r = 0.5 + slider_pos
                    force = slider_mass * (gravity * np.cos(theta) - r * thetadot**2)
                elif force_mode == 'normal':
                    theta = data.qpos[0]
                    # Add a small epsilon to avoid division by zero when cos(theta) is 0
                    force = (slider_mass * gravity) / (np.cos(theta) + 1e-6)
                
                # Apply force to the slider actuator (actuator index 1)
                data.ctrl[1] = force
                mujoco.mj_step(model, data)
            
            # Render and write frame
            renderer.update_scene(data, camera=cam_name, scene_option=scene_option)
            frame = renderer.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    video_writer.release()
    print(f"Finished: {filename}")

def main():
    """Generates all four requested videos."""
    model = mujoco.MjModel.from_xml_path("xmls/linear_inverted_pendulum.xml")

    # Define the video generation tasks
    tasks = {
        "free_fall.mp4": "free_fall",
        "radial_dynamics.mp4": "radial",
        "constant_force_mg.mp4": "mg",
        "normal_force.mp4": "normal"
    }

    for filename, mode in tasks.items():
        generate_video(model, filename, mode)

if __name__ == "__main__":
   main()
   print("Simulation finished.")
