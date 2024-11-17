#!/usr/bin/env python3
import rospy
import os
import sys
import numpy as np
import threading
from threading import Lock
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp")))

import DR_init
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *
from DR_common import *

from std_msgs.msg import String
from dsr_msgs.msg import TorqueRTStream, ServoJRTStream, ServoLRTStream
import psutil

# Global variables
g_stRTState = None
mtx = Lock()
first_get = False

# Constants class remains unchanged
class Constants:
    J_m = np.array([
        [0.0004956, 0, 0, 0, 0, 0],
        [0, 0.0004956, 0, 0, 0, 0],
        [0, 0, 0.0001839, 0, 0, 0],
        [0, 0, 0, 0.00009901, 0, 0],
        [0, 0, 0, 0, 0.00009901, 0],
        [0, 0, 0, 0, 0, 0.00009901]
    ])

    Gear_Ratio = np.array([
        [100, 0, 0, 0, 0, 0],
        [0, 100, 0, 0, 0, 0],
        [0, 0, 100, 0, 0, 0],
        [0, 0, 0, 80, 0, 0],
        [0, 0, 0, 0, 80, 0],
        [0, 0, 0, 0, 0, 80]
    ])

    M_d = np.array([
        [2.5, 0, 0, 0, 0, 0],
        [0, 2.5, 0, 0, 0, 0],
        [0, 0, 2.5, 0, 0, 0],
        [0, 0, 0, 2.5, 0, 0],
        [0, 0, 0, 0, 2.5, 0],
        [0, 0, 0, 0, 0, 2.5]
    ])

    D_d = np.array([
        [15, 0, 0, 0, 0, 0],
        [0, 15, 0, 0, 0, 0],
        [0, 0, 15, 0, 0, 0],
        [0, 0, 0, 15, 0, 0],
        [0, 0, 0, 0, 15, 0],
        [0, 0, 0, 0, 0, 15]
    ])

    K_d = np.array([
        [100, 0, 0, 0, 0, 0],
        [0, 100, 0, 0, 0, 0],
        [0, 0, 100, 0, 0, 0],
        [0, 0, 0, 100, 0, 0],
        [0, 0, 0, 0, 100, 0],
        [0, 0, 0, 0, 0, 100]
    ])

    K_o = np.array([
        [0.1, 0, 0, 0, 0, 0],
        [0, 0.1, 0, 0, 0, 0],
        [0, 0, 0.1, 0, 0, 0],
        [0, 0, 0, 0.1, 0, 0],
        [0, 0, 0, 0, 0.1, 0],
        [0, 0, 0, 0, 0, 0.1]
    ])

class RT_STATE:
    def __init__(self):
        # Initialize all state variables
        self.time_stamp = 0.0
        self.actual_joint_position = np.zeros(6)
        self.actual_joint_velocity = np.zeros(6)
        self.actual_joint_position_abs = np.zeros(6)
        self.actual_joint_velocity_abs = np.zeros(6)
        self.actual_tcp_position = np.zeros(6)
        self.actual_tcp_velocity = np.zeros(6)
        self.actual_flange_position = np.zeros(6)
        self.actual_flange_velocity = np.zeros(6)
        self.actual_motor_torque = np.zeros(6)
        self.actual_joint_torque = np.zeros(6)
        self.raw_joint_torque = np.zeros(6)
        self.raw_force_torque = np.zeros(6)
        self.external_joint_torque = np.zeros(6)
        self.external_tcp_force = np.zeros(6)
        self.target_joint_position = np.zeros(6)
        self.target_joint_velocity = np.zeros(6)
        self.target_joint_acceleration = np.zeros(6)
        self.target_motor_torque = np.zeros(6)
        self.target_tcp_position = np.zeros(6)
        self.target_tcp_velocity = np.zeros(6)
        self.gravity_torque = np.zeros(6)
        self.joint_temperature = np.zeros(6)
        self.goal_joint_position = np.zeros(6)
        self.goal_tcp_position = np.zeros(6)
        
        # Initialize matrices
        self.mass_matrix = np.zeros((6, 6))
        self.coriolis_matrix = np.zeros((6, 6))
        self.jacobian_matrix = np.zeros((6, 6))

class RealtimeAPINode:
    def __init__(self):
        # Initialize single ROS node
        rospy.init_node('realtime_api')
        
        # Initialize DRFL interface
        self.drfl = None  # Initialize your DRFL interface here
        
        # Initialize control vectors
        self.trq = np.zeros(6)
        self.trq_d = np.zeros(6)
        
        # Set up RT monitoring
        self.setup_monitoring()
        
        # Initialize timers
        self.read_timer = rospy.Timer(rospy.Duration(0.000333), self.read_data_rt_api)  # ~3000Hz
        self.torque_timer = rospy.Timer(rospy.Duration(0.001), self.torque_rt_api)  # 1000Hz

    def setup_monitoring(self):
        try:
            if self.drfl:
                self.drfl.set_on_rt_monitoring_data(self.on_rt_monitoring_data)
        except Exception as e:
            rospy.logerr(f"Failed to set up RT monitoring: {e}")

    @staticmethod
    def on_rt_monitoring_data(data):
        global g_stRTState
        if g_stRTState is None:
            g_stRTState = RT_STATE()

        g_stRTState.time_stamp = data.time_stamp
        
        # Update vectors
        for i in range(6):
            g_stRTState.actual_joint_position[i] = data.actual_joint_position[i]
            g_stRTState.actual_joint_velocity[i] = data.actual_joint_velocity[i]
            g_stRTState.gravity_torque[i] = data.gravity_torque[i]

        # Update matrices
        g_stRTState.mass_matrix = np.array(data.mass_matrix).reshape(6, 6)
        g_stRTState.coriolis_matrix = np.array(data.coriolis_matrix).reshape(6, 6)
        g_stRTState.jacobian_matrix = np.array(data.jacobian_matrix).reshape(6, 6)

        rospy.loginfo(f"time_stamp: {data.time_stamp}")

    def read_data_rt_api(self, event):
        global g_stRTState, first_get
        try:
            data = self.drfl.read_data_rt() if self.drfl else None
            if data:
                with mtx:
                    g_stRTState.time_stamp = data.time_stamp
                    # Update state vectors and matrices
                    for i in range(6):
                        g_stRTState.actual_joint_position_abs[i] = data.actual_joint_position_abs[i]
                        g_stRTState.actual_joint_velocity_abs[i] = data.actual_joint_velocity_abs[i]
                        g_stRTState.gravity_torque[i] = data.gravity_torque[i]
                    
                    if not first_get:
                        first_get = True
                        rospy.loginfo("Data updated")
        except Exception as e:
            rospy.logerr(f"Failed to read RT data: {e}")

    def torque_rt_api(self, event):
        global first_get
        if not first_get:
            rospy.loginfo("Data not updated yet")
            return

        self.trq = self.gravity_compensation()
        # self.trq = self.external_force_resist()  # Alternative control strategy
        
        with mtx:
            self.trq_d = self.trq.copy()

        if self.drfl:
            self.drfl.torque_rt(self.trq_d.tolist(), 0)
            rospy.loginfo(f"trq_d: {self.trq_d}")

    def gravity_compensation(self):
        """Implement gravity compensation control strategy"""
        with mtx:
            return g_stRTState.gravity_torque.copy()

    def external_force_resist(self):
        """Implement external force resistance control strategy"""
        with mtx:
            return g_stRTState.actual_joint_torque.copy()

def set_cpu_affinity():
    """Set CPU affinity for the current process"""
    try:
        # Set affinity to CPUs 2 and 3 (bitmask 0b1100)
        p = psutil.Process()
        p.cpu_affinity([2, 3])
        
        rospy.loginfo("Pinned CPUs:")
        for cpu in p.cpu_affinity():
            rospy.loginfo(f"  CPU{cpu}")
        return True
    except Exception as e:
        rospy.logerr(f"Couldn't set CPU affinity. Error: {e}")
        return False

def main():
    try:
        # Set CPU affinity
        if not set_cpu_affinity():
            return

        # Initialize global state
        global g_stRTState
        g_stRTState = RT_STATE()

        # Create single node instance
        node = RealtimeAPINode()

        # RT Initialize (if DRFL is available)
        if node.drfl:
            assert node.drfl.connect_rt_control("192.168.137.100", 12345)
            node.drfl.set_rt_control_output("v1.0", 0.001, 4)
            node.drfl.start_rt_control()

        try:
            rospy.spin()
        finally:
            # RT Shutdown
            if node.drfl:
                node.drfl.stop_rt_control()
                node.drfl.disconnect_rt_control()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")

if __name__ == '__main__':
    main()