import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt

# --- PD Controller Parameters ---
Kp = 50
Kd = 25

# --- Target Configuration ---
target_qpos = np.array([1.570796326794896, -1.570796326794896])

 # Load the model and data
model = mujoco.MjModel.from_xml_path("xmls/double_revolute.xml")# <-- your XML file
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

# Lists to store joint data
qpos_history = []
time_history = []

def my_controller(m: mujoco.MjModel, d: mujoco.MjData):
    error_pos = target_qpos - d.qpos[:2]
    error_vel = -d.qvel[:2]
    torque = Kp * error_pos + Kd * error_vel
    d.ctrl[:2] = torque


def generate_video(model, data, filename, duration=10.0, width=1280, height=720, rHz=60, cam_name="fixed", camera_settings=None):
    """Generates a video of the double revolute simulation."""
    mujoco.mj_resetDataKeyframe(model, data, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, rHz, (width, height))
    scene_option = mujoco.MjvOption()
    rFrtime = 1.0 / rHz

    time_history = []
    q1_history = []
    q2_history = []

    print(f"Generating video '{filename}'...")
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        cam_obj = None
        if camera_settings:
            cam_obj = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam_obj)
            if 'lookat' in camera_settings:
                cam_obj.lookat[:] = np.array(camera_settings['lookat'])
            if 'distance' in camera_settings:
                cam_obj.distance = float(camera_settings['distance'])
            if 'azimuth' in camera_settings:
                cam_obj.azimuth = float(camera_settings['azimuth'])
            if 'elevation' in camera_settings:
                cam_obj.elevation = float(camera_settings['elevation'])

        while data.time < duration:
            simstart = data.time
            while (data.time - simstart) < rFrtime:
                mujoco.mj_step(model, data)
                time_history.append(data.time)
                q1_history.append(float(data.qpos[0]))
                q2_history.append(float(data.qpos[1]))

            renderer.update_scene(data, camera=(cam_obj if cam_obj is not None else cam_name), scene_option=scene_option)
            frame = renderer.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Finished: {filename}")
    return time_history, q1_history, q2_history


def plot_results(time_history, q1_history, q2_history):
    plt.figure(figsize=(10, 5))
    plt.plot(time_history, q1_history, label='q1')
    plt.plot(time_history, q2_history, label='q2')
    plt.plot(time_history, np.ones_like(time_history) * target_qpos[0], 'r--', label='target q1')
    plt.plot(time_history, np.ones_like(time_history) * target_qpos[1], 'g--', label='target q2')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Angle [rad]')
    plt.title('Joint Angles vs Target')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    mujoco.set_mjcb_control(my_controller)
    GENERATE_VIDEO = False

    if GENERATE_VIDEO:
        os.makedirs("videos", exist_ok=True)
        time_history, q1_history, q2_history = generate_video(
            model, data, "videos/double_revolute.mp4", duration=10.0,
            camera_settings={
                "lookat": [0.0, 0.0, 0.5],
                "distance": 2.5,
                "azimuth": 90,
                "elevation": -20,
            }
        )
    else:
        time_history = []
        q1_history = []
        q2_history = []
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < 20:
                step_start = time.time()
                mujoco.mj_step(model, data)
                time_history.append(data.time)
                q1_history.append(float(data.qpos[0]))
                q2_history.append(float(data.qpos[1]))
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    plot_results(time_history, q1_history, q2_history)


if __name__ == "__main__":
    main()
    print("Simulation finished.")
