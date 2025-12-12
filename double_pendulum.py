#Imports
import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt

#Loading Model, Loading Data, and setting initial generalized coords using keyframe defined in XML
model = mujoco.MjModel.from_xml_path("xmls/double_pendulum.xml") # <-- your XML file
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0) # zeroth keyframe

#Video Function Definition
def generate_video(model, data, filename, duration=10.0, width=1280, height=720, rHz=60, integrator=1, cam_name="fixed", camera_settings=None):
    """Generate video of the passive double pendulum swing."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    model.opt.integrator = integrator

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, rHz, (width, height))
    scene_option = mujoco.MjvOption()
    rFrtime = 1.0 / rHz

    time_history = []
    q1_history = []
    q2_history = []
    energy_history = []

    with mujoco.Renderer(model, width=width, height=height) as renderer:
        cam_obj = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam_obj)
        if camera_settings:
            if 'lookat' in camera_settings:
                cam_obj.lookat[:] = np.array(camera_settings['lookat'])
            if 'distance' in camera_settings:
                cam_obj.distance = float(camera_settings['distance'])
            if 'azimuth' in camera_settings:
                cam_obj.azimuth = float(camera_settings['azimuth'])
            if 'elevation' in camera_settings:
                cam_obj.elevation = float(camera_settings['elevation'])

        print(f"Generating video '{filename}'...")
        while data.time < duration:
            simstart = data.time
            while (data.time - simstart) < rFrtime:
                mujoco.mj_step(model, data)
                time_history.append(data.time)
                q1_history.append(float(data.qpos[0]))
                q2_history.append(float(data.qpos[1]))
                energy_history.append(float(np.sum(data.energy)))

            renderer.update_scene(data, camera=(cam_obj if camera_settings else cam_name), scene_option=scene_option)
            frame = renderer.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Finished: {filename}")
    return time_history, q1_history, q2_history, energy_history

#Plotting Function Definition
def plot_results(time_history, q1_history, q2_history, energy_history):
    plt.figure(figsize=(10, 7))
    plt.subplot(3, 1, 1)
    plt.plot(time_history, q1_history, label='q1')
    plt.plot(time_history, q2_history, label='q2')
    plt.ylabel('Joint Angle [rad]')
    plt.title('Double Pendulum Angles and Energy')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_history, np.gradient(q1_history), label='dq1/dt')
    plt.plot(time_history, np.gradient(q2_history), label='dq2/dt')
    plt.ylabel('Joint Velocity [rad/s]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_history, energy_history, label='Total Energy')
    plt.xlabel('Time [s]')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    GENERATE_VIDEO = False

    if GENERATE_VIDEO:
        os.makedirs("videos", exist_ok=True)
        time_history, q1_history, q2_history, energy_history = generate_video(
            model, data, "videos/double_pendulum.mp4", duration=10.0, integrator=1,
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
        energy_history = []
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < 10:
                step_start = time.time()
                mujoco.mj_step(model, data)
                time_history.append(data.time)
                q1_history.append(float(data.qpos[0]))
                q2_history.append(float(data.qpos[1]))
                energy_history.append(float(np.sum(data.energy)))
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    plot_results(time_history, q1_history, q2_history, energy_history)


if __name__ == "__main__":
    main()
    print("Simulation finished.")
