#Imports
import mujoco
import mujoco.viewer
import time

#Loading Model, Loading Data, and setting initial generalized coords using keyframe defined in XML
model = mujoco.MjModel.from_xml_path("xmls/linear_inverted_pendulum.xml")  #Path to your XML file
data  = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0) # zeroth keyframe

def main():
    #Launching Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)#Step Simulation
            time.sleep(2*max(0, model.opt.timestep - (time.time() - step_start)))# used to view motion in real-time
            viewer.sync()

if __name__ == "__main__":
    main()
    print("Simulation finished.")