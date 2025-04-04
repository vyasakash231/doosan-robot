#!/usr/bin/env python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp"))) # get import path : DSR_ROBOT.py 
from basic_import import *

def shutdown():
    print("shutdown time!")
    print("shutdown time!")

    # '/my_node' is publishing data using publisher named 'my_publisher' to the topic '/dsr01a0509/stop'
    my_publisher.publish(stop_mode=STOP_TYPE_QUICK)
    return 

def call_back_func_1(msg):
    pos_list = [round(i,4) for i in list(msg.position)]
    #print(f"Joint_angles: {pos_list}")

def call_back_func_2(msg):
    pos_list = [round(i,4) for i in list(msg.current_posj)]
    #print(f"Joint_angles: {pos_list}")

class Task_space_control():
    def __init__(self):
        pass

if __name__ == "__main__":
    rospy.init_node('my_node')  # creating a node
    rospy.on_shutdown(shutdown)  # A function named 'shutdown' to be called when the node is shutdown.

    #rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')  # Wait until the service becomes available

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
    #my_subscriber_1 = rospy.Subscriber('/dsr01a0509/joint_states', JointState, call_back_func_1)  # In radian
    my_subscriber_2 = rospy.Subscriber('/dsr01a0509/state', RobotState, call_back_func_2)  # In degrees

    # create ROS service_proxy
    """
    #____________________________________________________________________________________________
    # move_joint  
    # The robot moves to the target joint position (pos) from the current joint position.
    #____________________________________________________________________________________________

    float64[6] pos               # target joint angle list [degree] 
    float64    vel               # set velocity: [deg/sec]
    float64    acc               # set acceleration: [deg/sec2]
    float64    time #= 0.0       # Time [sec] 
    float64    radius #=0.0      # Radius under blending mode [mm] 
    int8       mode #= 0         # MOVE_MODE_ABSOLUTE=0, MOVE_MODE_RELATIVE=1 
    int8       blendType #= 0    # BLENDING_SPEED_TYPE_DUPLICATE=0, BLENDING_SPEED_TYPE_OVERRIDE=1
    int8       syncType #=0      # SYNC = 0, ASYNC = 1
    ---
    bool success
    """
    Joint_angles = [90,0,0,0,0,0]
    rospy.wait_for_service('/dsr01a0509/motion/move_joint')  # Wait until the service becomes available
    move_joint = rospy.ServiceProxy('/dsr01a0509/motion/move_joint', MoveJoint)
    move_joint(Joint_angles, 20, 10, 0, 0, 0, 0, 0)

    print(f"Joint_angles: {get_current_posj()}") # if we pass no referance, it will take referance as base coordinate
    print(f"End-effector pos: {get_current_posx()}") # if we pass no referance, it will take referance as base coordinate

    rospy.spin()  # To stop the loop and program by pressing ctr + C    