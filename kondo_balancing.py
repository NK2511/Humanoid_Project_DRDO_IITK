#!/home/nandhith/humanoid_venv/bin/python3
import mujoco
import mujoco.viewer
import numpy as np
from time import sleep
import cv2
import os
import matplotlib.pyplot as plt
try:
    from .mujoco_ros_utils import MujocoRosConnector
except ImportError:
    from mujoco_ros_utils import MujocoRosConnector

# Load the kondo model
model_path = 'xmls/kondo_scene_flat_ground.xml'
model = mujoco.MjModel.from_xml_path(model_path)

# make data
data = mujoco.MjData(model)

# Get the actuator ID for the neck yaw joint
head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'NY')

# set the standing keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)

nact = model.nu
# define kp kd tau_sat
Kp_list = np.ones(nact) * 5    # 22 DOF, Kp = 5 for all
Kd_list = np.ones(nact) * 0.1  # Kd = 0.1 for all

Tsat = 1.2                        # Saturation torque
tau_sat_list = np.ones(nact) * Tsat # Saturation torque for all joints

q_des_list = np.zeros(nact)    # Desired joint angles
dq_des_list = np.zeros(nact)   # Desired joint velocities

# Global pause flag for simulation
ppause = False

def keyboard_func(keycode):
    """
    Keyboard callback to toggle simulation pause.
    Spacebar toggles the pause state.
    """
    global ppause
    if keycode == ord(' '):
        ppause = not ppause

# define control callback function
def satFunc(x, abs_bound):
    """
    Saturate the input x to the absolute bound abs_bound

    @param x: input value
    @param abs_bound: absolute bound
    @return: saturated value
    """
    # Smooth saturation function
    n = 5  # Smoothness parameter
    num = x
    den = np.power(1 + np.power(x / abs_bound, 2 * n), 1 / (2 * n))
    return num / den
    
def controller(m:mujoco.MjModel, d:mujoco.MjData):
    """
    Controller for the robot.

    @param m: MuJoCo model
    @param d: MuJoCo data
    """
    # joint torques vector
    jt = np.zeros(22)

    # Calculate the joint torques
    global Kp_list, Kd_list, q_des_list, dq_des_list, tau_sat_list
    for i in range(22):
        jt[i] = Kp_list[i] * (q_des_list[i] - d.qpos[7+i]) + Kd_list[i] * (dq_des_list[i] - d.qvel[6+i])

    # Apply sinusoidal torque to the head joint
    jt[head_id] = 0.5 * np.sin(3 * d.time)

    # Saturate the joint torques
    d.ctrl = satFunc(jt, tau_sat_list)

# define main and setup the viewer

def main():

    # set the control callback
    mujoco.set_mjcb_control(controller)

    # Set the joint angles accordingly
    global q_des_list, dq_des_list
    q_des_list = data.qpos[7:].copy()
    dq_des_list = data.qvel[6:].copy()

    global ppause
    idx_geom = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_func) as viewer:

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # Simulation loop
        while viewer.is_running():
            
            simstart = data.time

            # Step for 1/60 sec
            if not ppause:

                while data.time - simstart < 1 / 60:                    
                    # Control callback is set already, just step the simulation
                    mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Print the location of body frame of body waist
            # print(f"Body waist position: {data.xpos[model.body('Waist').id]}")
            
            # Print the location of the body frame of head
            # print(f"Body head position: {data.xpos[model.body('Head').id]}")
            
            # Difference between the two positions
            # diff = data.xpos[model.body('Head').id] - data.xpos[model.body('Waist').id]
            # print(f"Difference between head and waist: {diff}")
            
            # print(data.qpos[0:7])
            min, max = model.jnt_range[3]

            print(min, max)

            viewer.sync()
            sleep(0.013)
        
        viewer.close()


if __name__ == "__main__":
    main()