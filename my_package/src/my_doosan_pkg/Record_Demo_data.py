#!/usr/bin/env python3

import rospy
import os
from math import *
import numpy as np
import time
import threading  # Threads are a way to run multiple tasks concurrently within a single process. By using threads, you can perform multiple operations simultaneously, which can be useful for tasks like handling asynchronous events, running background tasks.
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../common/imp"))) # get import path : DSR_ROBOT.py 

import DR_init  # at doosan-robot/common/imp/
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *  # at doosan-robot/common/imp/
from DR_common import *  # at doosan-robot/common/imp/

import pyrealsense2 as rs
import cv2 as cv
from cv2 import aruco

# Importing messages and services 
from dsr_msgs.msg import RobotStop, RobotState  # at doosan-robot/dsr_msgs/msg/
from dsr_msgs.srv import *
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from scipy.spatial.transform import Rotation


def call_back_func(msg):
    T_matrix = np.zeros((6,4,4))
    # Process the TF messages
    i = 0
    T_e_0 = np.eye(4)
    Homo_matrix = np.eye(4)
    for transform in msg.transforms:
        # Extract relevant information from the message
        frame_id = transform.header.frame_id
        child_frame_id = transform.child_frame_id
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Print the information
        t = np.array([translation.x, translation.y, translation.z])
        r = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
        Homo_matrix[:3,:3] = r.as_matrix()
        Homo_matrix[:3,-1] = t.T
        T_e_0 = np.dot(T_e_0,Homo_matrix)

        T_matrix[i,:,:] = Homo_matrix
        i += 1

def shutdown():
    print("shutdown time!")
    print("shutdown time!")

    # '/my_node' is publishing data using publisher named 'my_publisher' to the topic '/dsr01a0509/stop'
    my_publisher.publish(stop_mode=STOP_TYPE_QUICK)
    return 

def call_back_func_1(msg):
    pos_list = [round(i,4) for i in list(msg.position)]
    # print(f"Joint_angles in radian: {pos_list}")

def call_back_func_2(msg):
    pos_list = [round(i,4) for i in list(msg.current_posj)]
    print(f"Joint_angles degrees: {pos_list}")

if __name__ == "__main__":
    rospy.init_node('my_node')  # creating a node
    rospy.on_shutdown(shutdown)  # A function named 'shutdown' to be called when the node is shutdown.

    rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')  # Wait until the service becomes available
    """
    This line creates a service proxy named set_robot_mode_proxy for calling the service /dsr01a0509/system/set_robot_mode. 
    SetRobotMode is the service type. This service is used to set the mode of the robot system, such as changing between 
    manual and automatic modes.

    service proxy: set_robot_mode_proxy
     
    Node: /dsr01a0509
    service: /dsr01a0509/system/set_robot_mode
    type: dsr_msgs/SetRobotMode
    Args: robot_mode

    FILE --> SetRobotMode.srv   (ROS service defined in srv file, it contain a request msg and response msg)
    #_________________________________
    # set_robot_mode
    # Change the robot-mode
    # 0 : ROBOT_MODE_MANUAL  (robot LED lights up blue) --> use it for recording demonstration
    # 1 : ROBOT_MODE_AUTONOMOUS  (robot LED lights up in white)
    # 2 : ROBOT_MODE_MEASURE
    # drfl.SetRobotMode()
    #________________________________
    int8 robot_mode # <Robot_Mode>
    ---
    bool success
    """
    set_robot_mode_proxy  = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
    set_robot_mode_proxy(ROBOT_MODE_MANUAL)  # Calls the service proxy and pass the args:robot_mode, to set the robot mode to ROBOT_MODE_MANUAL.

    # Creates a publisher on the topic '/dsr01a0509/stop' to publish RobotStop messages with a queue size of 10.         
    my_publisher = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)  

    # Create subscriber 
    """
    there are two topics which can be subscribed to get joint data and velocity,
    (1) /dsr01a0509/joint_states  -->  gives joint angles as position in radian
    (2) /dsr01a0509/state  -->  gives complete info of robot and joint angle as current_posj in degree
    """ 
    my_subscriber_1 = rospy.Subscriber('/dsr01a0509/joint_states', JointState, call_back_func_1)  # In radian
    my_subscriber_2 = rospy.Subscriber('/dsr01a0509/state', RobotState, call_back_func_2)  # In degrees

    my_subscriber = rospy.Subscriber('/tf', TFMessage, call_back_func)  # In degrees

    rospy.spin()  # To stop the loop and program by pressing ctr + C    