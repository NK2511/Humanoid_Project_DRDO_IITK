"""
/home/nandhith/humanoid_python/mujoco_ros_utils.py
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading

class MujocoRosConnector:
    """
    A helper class to publish MuJoCo data to ROS 2 topics easily.
    It runs ROS in a background thread so it doesn't block your simulation loop.
    """
    def __init__(self, node_name='mujoco_ros_connector'):
        # Initialize ROS 2 context if not already active
        if not rclpy.ok():
            rclpy.init()
        
        self.node = Node(node_name)
        self.publishers = {}
        
        # Spin in a separate thread to handle ROS callbacks without blocking the sim
        self._thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._thread.start()

    def publish(self, topic_name, data):
        """
        Publishes data to a ROS 2 topic.
        Auto-creates the publisher if it doesn't exist.
        
        Args:
            topic_name (str): The name of the ROS topic (e.g., 'qpos').
            data (list, np.ndarray, float): The data to publish.
        """
        # Create publisher if new topic
        if topic_name not in self.publishers:
            self.publishers[topic_name] = self.node.create_publisher(Float64MultiArray, topic_name, 10)
        
        # Format message
        msg = Float64MultiArray()
        if isinstance(data, np.ndarray):
            msg.data = data.flatten().tolist()
        elif isinstance(data, list):
            msg.data = data
        else:
            # Assume scalar
            msg.data = [float(data)]
            
        self.publishers[topic_name].publish(msg)

    def shutdown(self):
        """Cleanly shuts down the ROS node."""
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
