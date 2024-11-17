#!/usr/bin/env python3
import rospy
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp")))

import DR_init
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *
from DR_common import *

from std_msgs.msg import String
from dsr_msgs.msg import TorqueRTStream, ServoJRTStream, ServoLRTStream
from dsr_msgs.srv import ReadDataRT, ReadDataRTRequest
import threading
import numpy as np
from threading import Lock

# Global variables
g_stRTState = None
mtx = Lock()
first_get = False

class RT_STATE:
    def __init__(self):
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
        self.coriolis_matrix = np.zeros((6, 6))
        self.mass_matrix = np.zeros((6, 6))
        self.jacobian_matrix = np.zeros((6, 6))

class RealtimeControlNode:
    def __init__(self):
        # Initialize the single ROS node
        rospy.init_node('realtime_control')
        
        # Initialize RT client
        self.client_ = rospy.ServiceProxy('/dsr01/realtime/read_data_rt', ReadDataRT)
        
        # Initialize publishers
        self.torque_publisher_ = rospy.Publisher('/dsr01/torque_rt_stream', TorqueRTStream, queue_size=10)
        self.servoj_publisher_ = rospy.Publisher('/dsr01/servoj_rt_stream', ServoJRTStream, queue_size=10)
        self.servol_publisher_ = rospy.Publisher('/dsr01/servol_rt_stream', ServoLRTStream, queue_size=10)
        
        # Initialize control variables
        self.trq_g = np.zeros(6)
        self.trq_d = np.zeros(6)
        self.pos_d = np.zeros(6)
        self.vel_d = np.zeros(6)
        self.acc_d = np.zeros(6)
        self.time_d = 0.0
        
        # Start threads
        self.client_thread_ = threading.Thread(target=self.read_data_rt_client)
        self.client_thread_.start()
        
        # Initialize timers for publishers
        self.timer_torque_ = rospy.Timer(rospy.Duration.from_sec(0.001), self.torque_rt_stream_publisher)
        self.timer_servoj_ = rospy.Timer(rospy.Duration.from_sec(0.001), self.servoj_rt_stream_publisher)
        self.timer_servol_ = rospy.Timer(rospy.Duration.from_sec(0.001), self.servol_rt_stream_publisher)

    def read_data_rt_client(self):
        global g_stRTState, first_get
        rate = rospy.Rate(3000)  # 3000Hz
        
        while not rospy.is_shutdown():
            try:
                rate.sleep()
                rospy.wait_for_service('/dsr01/realtime/read_data_rt', timeout=1.0)
                
                request = ReadDataRTRequest()
                response = self.client_(request)
                
                if not first_get:
                    first_get = True
                
                with mtx:
                    g_stRTState.time_stamp = response.data.time_stamp
                    for i in range(6):
                        g_stRTState.actual_joint_position[i] = response.data.actual_joint_position[i]
                        g_stRTState.actual_joint_velocity[i] = response.data.actual_joint_velocity[i]
                        g_stRTState.gravity_torque[i] = response.data.gravity_torque[i]
                    
                    # Update matrices
                    for i in range(6):
                        for j in range(6):
                            g_stRTState.coriolis_matrix[i][j] = response.data.coriolis_matrix[i].data[j]
                            g_stRTState.mass_matrix[i][j] = response.data.mass_matrix[i].data[j]
                            g_stRTState.jacobian_matrix[i][j] = response.data.jacobian_matrix[i].data[j]

            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
            except rospy.ROSException as e:
                rospy.logwarn("Waiting for the server to be up...")

    def torque_rt_stream_publisher(self, event):
        global first_get, g_stRTState
        if not first_get:
            return

        with mtx:
            for i in range(6):
                self.trq_g[i] = g_stRTState.gravity_torque[i]
        
        self.trq_d = self.trq_g.copy()

        message = TorqueRTStream()
        message.tor = self.trq_d.tolist()
        message.time = 0.0
        
        self.torque_publisher_.publish(message)
        rospy.loginfo(f"trq_d: {self.trq_d}")

    def servoj_rt_stream_publisher(self, event):
        global first_get
        if not first_get:
            return

        message = ServoJRTStream()
        message.pos = self.pos_d.tolist()
        message.vel = self.vel_d.tolist()
        message.acc = self.acc_d.tolist()
        message.time = self.time_d

        self.servoj_publisher_.publish(message)
        rospy.loginfo("ServoJRTStream Published")

    def servol_rt_stream_publisher(self, event):
        global first_get
        if not first_get:
            return

        message = ServoLRTStream()
        message.pos = self.pos_d.tolist()
        message.vel = self.vel_d.tolist()
        message.acc = self.acc_d.tolist()
        message.time = self.time_d

        self.servol_publisher_.publish(message)
        rospy.loginfo("ServolRTStream Published")

    def __del__(self):
        if hasattr(self, 'client_thread_') and self.client_thread_.is_alive():
            self.client_thread_.join()
            rospy.loginfo("client_thread_ joined")
        rospy.loginfo("RealtimeControl node shut down")

def set_cpu_affinity():
    """Set CPU affinity for the current process"""
    try:
        import psutil
        # Set affinity to CPUs 2 and 3 (equivalent to bitmask 0b1100)
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

        # Create single node that handles all functionality
        node = RealtimeControlNode()

        # Spin node
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()