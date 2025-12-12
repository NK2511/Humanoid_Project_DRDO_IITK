#Imports
import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt

#Controller Gains
Kp_cart = 70
Kd_cart = 20
Kp_pole = 220
Kd_pole = 20
target_qpos = np.array([0, 0])


#Loading Model, Loading Data, and setting initial generalized coords using keyframe defined in XML
model = mujoco.MjModel.from_xml_path("xmls/cartpole.xml")   # <-- your XML file
data  = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0) # zeroth keyframe

#Controller Function Definition
#Takes Model and Data as input arguments outputs control torques to be applied
def my_controller(m:mujoco.MjModel, d:mujoco.MjData):
        x     = data.qpos[0]
        xdot  = data.qvel[0]
        theta = data.qpos[1]
        thetadot = data.qvel[1]

        force = (Kp_cart * (0 - x)
            + Kd_cart * (0 - xdot)
            + Kp_pole * (0 - theta)
            + Kd_pole * (0 - thetadot)
        )#torque/force calculation
        data.ctrl[0] = -force #applying torque/forces

#Video Function Definition
def generate_video(model, data, filename, duration=2.0, width=1280, height=720, rHz=60, cam_name="fixed"):
    """Generates a video of the simulation."""
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Video and rendering parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, rHz, (width, height))
    scene_option = mujoco.MjvOption()
    rFrtime = 1.0 / rHz

    # Lists to store data for plotting
    time_history = []
    torque_history = []
    rod_angle_history = []
    
    print(f"Generating video '{filename}'...")
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        while data.time < duration:
            simstart = data.time
            while (data.time - simstart) < rFrtime:
                mujoco.mj_step(model, data)
                
                # Log data 
                time_history.append(data.time)
                torque_history.append(data.ctrl[0])
                rod_angle_history.append(data.qpos[0])
            
            # Render and write frame
            renderer.update_scene(data, camera=cam_name, scene_option=scene_option)
            frame = renderer.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    video_writer.release()
    print(f"Finished: {filename}")
    return time_history, torque_history, rod_angle_history

#Plotting Function Definition
def plot_results(time_history, torque_history, rod_angle_history):
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # Subplot 1: Torque
    axs[0].plot(time_history, torque_history, label='Reaction Wheel Torque')
    axs[0].set_ylabel('Torque [Nm]')
    axs[0].set_title('Reaction Wheel Torque and Rod Angle vs. Time')
    axs[0].legend()
    axs[0].grid(True)
    # Subplot 2: Rod Angle
    axs[1].plot(time_history, rod_angle_history, label='Rod Angle', color='orange')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Angle [rad]')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def main(): 
    # Lists to store data for plotting
    time_history = []
    torque_history = []
    rod_angle_history = []

    # Set controller callback, this function will be called at each simulation step to compute control inputs
    mujoco.set_mjcb_control(my_controller)

    # Toggle between video generation and live viewer
    GENERATE_VIDEO = False 

    if GENERATE_VIDEO:
        os.makedirs("videos", exist_ok=True)
        time_history, torque_history, rod_angle_history = generate_video(model, data, "videos/cartpole.mp4", duration=10.0)
    else:
        #Launching Viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < 10:
                step_start = time.time()
                mujoco.mj_step(model, data) #Step Simulation
                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))# used to view motion in real-time
                viewer.sync()

              # Log data
                time_history.append(data.time)
                torque_history.append(data.ctrl[0])
                rod_angle_history.append(data.qpos[0])
    
    plot_results(time_history, torque_history, rod_angle_history)



if __name__ == "__main__":
    main()
    print("Simulation finished.")