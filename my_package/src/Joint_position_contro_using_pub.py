#!/usr/bin/env python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../common/imp"))) # get import path : DSR_ROBOT.py 
from basic_import import *
from scipy.spatial.transform import Rotation

def call_back_func(msg):
    """\tf only works when used with rviz"""
    T_matrix = np.zeros((6,4,4))
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
        Homo_matrix[:3,:3] = r.as_matrix()  #  rotation of nth frame wrt to n-1 frame
        Homo_matrix[:3,-1] = t.T   #  translation of nth frame wrt to n-1 frame
        T_e_0 = np.dot(T_e_0, Homo_matrix) 
        T_matrix[i,:,:] = T_e_0  #  rotation of nth frame wrt to base (0th) frame
        i += 1
    print(np.round(T_matrix,4),'\n')
    print("==========================================")


def shutdown():
    print("shutdown time!")
    print("shutdown time!")

    # '/my_node' is publishing data using publisher named 'my_publisher' to the topic '/dsr01a0509/stop'
    #my_publisher.publish(stop_mode=STOP_TYPE_QUICK)
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

    rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')  # Wait until the service becomes available

    set_robot_mode_proxy  = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
    set_robot_mode_proxy(ROBOT_MODE_AUTONOMOUS)  # Calls the service proxy and pass the args:robot_mode, to set the robot mode to ROBOT_MODE_AUTONOMOUS.

    # Creates a publisher on the topic '/dsr01a0509/stop' to publish RobotStop messages with a queue size of 10.         
    #my_publisher = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)  

    # Create subscriber 
    """
    there are two topics which can be subscribed to get joint data and velocity,
    (1) /dsr01a0509/joint_states  -->  gives joint angles as position in radian
    (2) /dsr01a0509/state  -->  gives complete info of robot and joint angle as current_posj in degree
    # """ 
    #my_subscriber_1 = rospy.Subscriber('/dsr01a0509/joint_states', JointState, call_back_func_1)  # In radian
    # my_subscriber_2 = rospy.Subscriber('/dsr01a0509/state', RobotState, call_back_func_2)  # In degrees

    my_subscriber = rospy.Subscriber('/tf', TFMessage, call_back_func)  # In degrees

    rate = rospy.Rate(10)  # 10 Hz

    # Creates a publisher on the topic '/dsr01a0509/stop' to publish RobotStop messages with a queue size of 10.         
    Joint_publisher = rospy.Publisher('/dsr01a0509/servoj_rt_stream', ServoJRTStream, queue_size=1) 

    # Create a Float64MultiArray message   
    msg = ServoJRTStream()
    msg.pos = [90,0,0,0,0,0]
    msg.vel = [20,0,0,0,0,0]
    msg.acc = [10,0,0,0,0,0]
    msg.time = 2.0

    while not rospy.is_shutdown():
        Joint_publisher.publish(msg)
        rate.sleep()

    # mode = get_control_mode()
    # print(mode)

    # print(f"Joint_angles: {get_current_posj()}") # if we pass no referance, it will take referance as base coordinate
    # print(f"End-effector pos: {get_current_posx()[0]}") # if we pass no referance, it will take referance as base coordinate

    rospy.spin()  # To stop the loop and program by pressing ctr + C    