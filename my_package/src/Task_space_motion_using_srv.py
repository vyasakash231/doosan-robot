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

# Importing messages and services 
from dsr_msgs.msg import RobotStop, RobotState  # at doosan-robot/dsr_msgs/msg/
from dsr_msgs.srv import *
from sensor_msgs.msg import JointState

def shutdown():
    print("shutdown time!")
    print("shutdown time!")

    # '/my_node' is publishing data using publisher named 'my_publisher' to the topic '/dsr01a0509/stop'
    my_publisher.publish(stop_mode=STOP_TYPE_QUICK)
    return 

def call_back_func_1(msg):
    pos_list = [round(i,4) for i in list(msg.position)]
    #print(f"End-effector coord: {pos_list}")

def call_back_func_2(msg):
    pos_list = [round(i,4) for i in list(msg.current_posx)]
    print(f"End-effector coord: {pos_list}")

class Task_space_control():
    def __init__(self):
        pass

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
    # 0 : ROBOT_MODE_MANUAL
    # 1 : ROBOT_MODE_AUTONOMOUS
    # 2 : ROBOT_MODE_MEASURE
    # drfl.SetRobotMode()
    #________________________________
    int8 robot_mode # <Robot_Mode>
    ---
    bool success
    """
    set_robot_mode_proxy  = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
    set_robot_mode_proxy(ROBOT_MODE_AUTONOMOUS)  # Calls the service proxy and pass the args:robot_mode, to set the robot mode to ROBOT_MODE_AUTONOMOUS.

    # Creates a publisher on the topic '/dsr01a0509/stop' to publish RobotStop messages with a queue size of 10.         
    my_publisher = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)  

    # Create subscriber 
    """
    there are two topics which can be subscribed to get joint data and velocity,
    (1) /dsr01a0509/joint_states  -->  gives joint angles as position in radian
    (2) /dsr01a0509/state  -->  gives complete info of robot and joint angle as current_posj in degree
    # """ 
    my_subscriber_2 = rospy.Subscriber('/dsr01a0509/state', RobotState, call_back_func_2)

    # Get get current Solution_space
    current_solution_space = rospy.ServiceProxy("/dsr01a0509/aux_control/get_current_solution_space", GetCurrentSolutionSpace)

    # create ROS service_proxy
    """
    #____________________________________________________________________________________________
    # move_jointx  
    #____________________________________________________________________________________________

    float64[6] pos              # target  
    float64    vel              # set velocity: [deg/sec]
    float64    acc              # set acceleration: [deg/sec2] 
    float64    time #= 0.0      # Time [sec] 
    float64    radius #=0.0     # Radius under blending mode [mm]   
    int8       ref              # DR_BASE(0), DR_TOOL(1), DR_WORLD(2)
                                # <DR_WORLD is only available in M2.40 or later> 
    int8       mode #= 0        # MOVE_MODE_ABSOLUTE=0, MOVE_MODE_RELATIVE=1 
    int8       blendType #= 0   # BLENDING_SPEED_TYPE_DUPLICATE=0, BLENDING_SPEED_TYPE_OVERRIDE=1
    int8       sol              # SolutionSpace : 0~7
    int8       syncType #=0     # SYNC = 0, ASYNC = 1
    ---
    bool success
    """

    # rospy.wait_for_service('/dsr01a0509/motion/move_jointx')  # Wait until the service becomes available
    # # E_cc = [370, 670, 650, 0, 180, 0]
    # sol_space = get_current_posx()[1]
    # alpha, beta, gamma = get_current_posx()[0][3:]
    # E_cc = [200, 0, 800, alpha, beta, gamma]
    # move_task = rospy.ServiceProxy("/dsr01a0509/motion/move_jointx", MoveJointx)
    # move_task(E_cc, 30, 30, None, None, None, 0, 0, sol_space, 0)

    # Move in Line
    # E_cc = [370, 670, 650, 0, 180, 0]
    # E_cc=posx(367, 10, 540.5, 62.0, 180, 62.0)
    alpha, beta, gamma = get_current_posx()[0][3:]
    E_cc = [200, 0, 800, alpha, beta, gamma]
    move_in_line = rospy.ServiceProxy("/dsr01a0509/motion/move_line", MoveLine)
    move_in_line(E_cc, [30,50], [30,50], 0, 0, 0, 0, 0, 0)

    # print(f"Joint_angles: {get_current_posj()}")
    # print(f"End-effector: {get_current_posx()}") # if we pass no referance, it will take referance as base coordinate

    rospy.spin()  # To stop the loop and program by pressing ctr + C

        
