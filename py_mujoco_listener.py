#!/home/nandhith/humanoid_venv/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import os

class MujocoListener(Node):
    def __init__(self):
        super().__init__('mujoco_listener')

        self.create_subscription(Float64MultiArray, 'qpos', self.qpos_callback, 10)
        self.create_subscription(Float64MultiArray, 'qvel', self.qvel_callback, 10)

    def qpos_callback(self, msg):
        self.get_logger().info(f"Received qpos: {msg.data}")

    def qvel_callback(self, msg):
        self.get_logger().info(f"Received qvel: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = MujocoListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

