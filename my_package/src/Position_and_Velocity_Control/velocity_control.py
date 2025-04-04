#! /usr/bin/python3
"""
The line at the top of this Python script is called a "shebang" line. It specifies the path to the Python interpreter that should be used to execute the script.
When you run a script in the terminal using ./script.py, the system looks for the shebang line at the top of the script to determine which interpreter to use. 
In this case, #!/usr/bin/python3 specifies that the system should use the python interpreter that is found in the user's PATH environment variable.
"""
import rospy
import os
from math import *
import numpy as np
import sys
import time
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../common/imp"))) # get import path : DSR_ROBOT.py 

from matplotlib import pyplot as Plot

import DR_init  # at doosan-robot/common/imp/
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *  # at doosan-robot/common/imp/
from DR_common import *  # at doosan-robot/common/imp/

# Importing messages and services 
from dsr_msgs.msg import RobotStop, RobotState, SpeedJRTStream, ServoJRTStream  # at doosan-robot/dsr_msgs/msg/
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
    #print(f"Joint_angles: {pos_list}")

def call_back_func_2(msg):
    pos_list = [round(i,4) for i in list(msg.current_posj)]
    joint_vel_list = [round(i,4) for i in list(msg.current_velj)]
    print(f"Joint_angles: {pos_list}")
    print(f"Joint velocity: {joint_vel_list}")

if __name__ == "__main__":
    rospy.init_node('my_node')  # creating a node
    rospy.on_shutdown(shutdown)  # A function named 'shutdown' to be called when the node is shutdown.

    rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')  # Wait until the service becomes available

    set_robot_mode_proxy  = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
    set_robot_mode_proxy(ROBOT_MODE_AUTONOMOUS)  # Calls the service proxy and pass the args:robot_mode, to set the robot mode to ROBOT_MODE_AUTONOMOUS.

    # Creates a publisher on the topic '/dsr01a0509/stop' to publish RobotStop messages with a queue size of 10.         
    my_publisher = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)  

    # Create subscriber 
    # my_subscriber_1 = rospy.Subscriber('/dsr01a0509/joint_states', JointState, call_back_func_1)  # In radian
    my_subscriber_2 = rospy.Subscriber('/dsr01a0509/state', RobotState, call_back_func_2)  # In degrees

    # connect_rt_control!!    
    drt_connect = rospy.ServiceProxy('/dsr01a0509/realtime/connect_rt_control', ConnectRTControl)
    retval = drt_connect(ip_address = "192.168.137.100", port = 12347)  
    if not retval:
       raise SystemExit('realtime connect failed')

    # set_rt_control_output
    drt_setout = rospy.ServiceProxy('/dsr01a0509/realtime/set_rt_control_output',SetRTControlOutput)
    retval = drt_setout(period = 0.01, loss = 10)  # period = 0.01 (sampling time, here 100Hz),  loss = 10 (unknown, currently unused by doosan firmware)
    if not retval:
        raise SystemExit('realtime set output failed')
    
    # Creates a publisher on the topic '/dsr01a0509/speedj_rt_stream' to publish SpeedJRTStream messages with a queue size of 1         
    speed_publisher = rospy.Publisher('/dsr01a0509/speedj_rt_stream', SpeedJRTStream, queue_size=1)  
    
    writedata=SpeedJRTStream()
    writedata.vel=[0,0,0,0,0,0]
    writedata.acc=[0,0,0,0,0,0]
    writedata.time=0

    # start_rt_control!!
    drt_start = rospy.ServiceProxy('/dsr01a0509/realtime/start_rt_control', StartRTControl)
    retval = drt_start()
    if not retval:
        raise SystemExit('realtime start control failed')

    drt_velj_limits=rospy.ServiceProxy('/dsr01a0509/realtime/set_velj_rt', SetVelJRT)    # set global joint velocity limit
    drt_accj_limits=rospy.ServiceProxy('/dsr01a0509/realtime/set_accj_rt', SetAccJRT)    # set global joint acceleration limit
    drt_velx_limits=rospy.ServiceProxy('/dsr01a0509/realtime/set_velx_rt', SetVelXRT)    # set global Task velocity limit
    drt_accx_limits=rospy.ServiceProxy('/dsr01a0509/realtime/set_accx_rt', SetAccXRT)    # set global Task acceleration limit

    drt_velj_limits([100, 100, 100, 100, 100, 100])
    drt_accj_limits([100, 100, 100, 100, 100, 100])
    drt_accx_limits(100,10)
    drt_velx_limits(200,10)

    # stop_rt_control
    stop_control = rospy.ServiceProxy('/dsr01a0509/realtime/stop_rt_control', StopRTControl)

    # disconnect_rt_control_cb
    drop_control = rospy.ServiceProxy('/dsr01a0509/realtime/disconnect_rt_control', DisconnectRTControl)
    
    # -------------main loop ------------------
    time.sleep(1)

    while not rospy.is_shutdown():
        writedata.vel = [0.5*(90 - round(get_current_posj()[0],1)),0,0,0,0,0]  # joint velocity in degree/sec

        # If we provide acc and time then it will not take velocity
        # writedata.acc = [0,0,0,0,0,0]  # joint acceleration in degree/sec2
        writedata.time = 1  # Time is necessary (otherwise it will not move)

        if round(get_current_posj()[0],1) >= 90:
            writedata.vel=[0,0,0,0,0,0]
            speed_publisher.publish(writedata)

            retval = stop_control()
            print("Stop returns: " + str(retval))

            retval = drop_control()
            print("Disconnect returns: " + str(retval))

            rospy.spin()  # To stop the loop and program by pressing ctr + C 

        # publish command
        speed_publisher.publish(writedata)
        time.sleep(0.05) # Sleep for 0.05sec   