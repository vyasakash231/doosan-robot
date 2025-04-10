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
import time
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp"))) # get import path : DSR_ROBOT.py 

from matplotlib import pyplot as Plot
from common_for_JLA import *

import DR_init  # at doosan-robot/common/imp/
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *  # at doosan-robot/common/imp/
from DR_common import *  # at doosan-robot/common/imp/

# Importing messages and services 
from dsr_msgs.msg import RobotStop, RobotState, SpeedJRTStream  # at doosan-robot/dsr_msgs/msg/
from dsr_msgs.srv import *
from sensor_msgs.msg import JointState


# Using class keyword we created our sample class called Task_space_Control
class Task_Space_Control():
    # CLASS ATTRIBUTE
    n = 6  # No of joints
    m = 3  # mth norm of a vector

    # DH Parameters
    alpha = np.array([0, -pi/2, 0, pi/2, -pi/2, pi/2])   
    a = np.array([0, 0, 0.409, 0, 0, 0])
    d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])
    le = 0

    def __init__(self):
        rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')  # Wait until the service becomes available

        set_robot_mode_proxy  = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
        set_robot_mode_proxy(ROBOT_MODE_AUTONOMOUS)  # Calls the service proxy and pass the args:robot_mode, to set the robot mode to ROBOT_MODE_AUTONOMOUS.

        # Creates a publisher on the topic '/dsr01a0509/stop' to publish RobotStop messages with a queue size of 10.
        self.my_publisher = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)  

        # Subscribe to the Kinematic pose of the manipulator
        rospy.Subscriber('/dsr01a0509/state', RobotState, self.call_back_func)  # has joint angle, ee-position in msg

        # function named 'shutdown' to be called when the node is shutdown
        rospy.on_shutdown(self.shutdown)

        # connect_rt_control!!    
        drt_connect = rospy.ServiceProxy('/dsr01a0509/realtime/connect_rt_control', ConnectRTControl)
        retval = drt_connect(ip_address = "192.168.137.100", port = 12347)  
        if not retval:
            raise SystemExit('realtime connect failed')

        # set_rt_control_output
        drt_setout = rospy.ServiceProxy('/dsr01a0509/realtime/set_rt_control_output',SetRTControlOutput)
        retval = drt_setout(period = 0.001, loss = 5)  # period = 0.001 (sampling time, here 1000Hz),  loss = 5 (unknown, currently unused by doosan firmware)
        if not retval:
            raise SystemExit('realtime set output failed')
        
        # Creates a publisher on the topic '/dsr01a0509/speedj_rt_stream' to publish SpeedJRTStream messages with a queue size of 1         
        self.speed_publisher = rospy.Publisher('/dsr01a0509/speedj_rt_stream', SpeedJRTStream, queue_size=5)

        # start_rt_control!!
        drt_start = rospy.ServiceProxy('/dsr01a0509/realtime/start_rt_control', StartRTControl)
        retval = drt_start()
        if not retval:
            raise SystemExit('realtime start control failed')
        
        drt_velj_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velj_rt', SetVelJRT)    # set global joint velocity limit
        drt_accj_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accj_rt', SetAccJRT)    # set global joint acceleration limit
        drt_velx_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velx_rt', SetVelXRT)    # set global Task velocity limit
        drt_accx_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accx_rt', SetAccXRT)    # set global Task acceleration limit

        # Set JOINT-LIMITS
        drt_velj_limits([100, 100, 100, 100, 100, 100])
        drt_accj_limits([100, 100, 100, 100, 100, 100])
        drt_accx_limits(100,10)
        drt_velx_limits(200,10)

        # stop_rt_control
        self.stop_control = rospy.ServiceProxy('/dsr01a0509/realtime/stop_rt_control', StopRTControl)

        # disconnect_rt_control_cb
        self.drop_control = rospy.ServiceProxy('/dsr01a0509/realtime/disconnect_rt_control', DisconnectRTControl)

        # Initialize the joint angles
        self.current_Px = None
        self.current_Py = None
        self.current_Pz = None
        
    def call_back_func(self, msg):
        self.current_Px = 0.001*msg.current_posx[0]  # converting position in X from mm to m
        self.current_Py = 0.001*msg.current_posx[1]  # converting position in Y from mm to m
        self.current_Pz = 0.001*msg.current_posx[2]  # converting position in Z from mm to m

    def shutdown(self):
        print("shutdown time!")
        print("shutdown time!")

        # '/my_node' is publishing data using publisher named 'my_publisher' to the topic '/dsr01a0509/stop'
        self.my_publisher.publish(stop_mode=STOP_TYPE_QUICK)
        return 

    def JLA_1(self,del_X):  # Inequality Constraint Method
        writedata = SpeedJRTStream()
        i = 0
        d_theta = np.zeros((6,1))  # defining an array to store joint velocity

        q_plot = np.radians(np.array(get_current_posj()))  # Hardware & Gazebo Joint Angle at Home Position

        joint_offset = np.array([0, -pi/2, pi/2, 0, 0, 0])  # Difference btw Hardware/Gazebo & DH convention
        
        Jc = np.eye((Task_Space_Control.n))  # Jacobian of the additional task
        epsilon = 0.5 * np.ones(np.expand_dims(joint_offset, axis=0).shape) # Activation buffer region width
        
        q_range = np.radians(np.array([[-360,360],[-95,95],[-135,135],[-360,360],[-135,135],[-360,360]]))
        
        start = time.time()
        while np.linalg.norm(del_X[:,i]) > 0.005:      
            # Calculating Weight Matrix
            We, Wc, Wv = weight_Func(Task_Space_Control.m,Task_Space_Control.n,q_range,np.expand_dims(np.radians(np.array(get_current_posj())), axis=0),epsilon)

            theta = np.radians(np.array(get_current_posj())) + joint_offset  # for DH-convention, In radians

            # Calculate Jacobain
            J = jacobian_matrix(Task_Space_Control.n,Task_Space_Control.alpha,Task_Space_Control.a,Task_Space_Control.d,theta,Task_Space_Control.le)  # Calculate J
            Je = J[0:3,:]

            # np.linalg.inv() compute inverse -> Given a square matrix J, return the matrix J_inv
            Jn = np.linalg.inv(np.transpose(Je) @ We @ Je + np.transpose(Jc) @ Wc @ Jc + Wv) @ np.transpose(Je) @ We

            # Newton Raphson Method
            del_X = np.hstack((del_X, np.array([[goal_Px - self.current_Px],[goal_Py - self.current_Py],[goal_Pz - self.current_Pz]])))

            # ADAPTIVE GAIN: formulation
            e = np.array([[goal_Px - self.current_Px],[goal_Py - self.current_Py],[goal_Pz - self.current_Pz]])
            e0 = e if i == 0 else e0
            K = 20 * (1-exp(-i*0.05)) *  exp((np.linalg.norm(e0) - np.linalg.norm(e))/np.linalg.norm(e0))  # ADAPTIVE GAIN factor

            # Store Joint Velocity for plotting
            d_theta = np.hstack((d_theta, K * np.reshape((Jn @ del_X[:,i+1]),(Task_Space_Control.n,1))))

            writedata.vel = list(K * np.reshape((Jn @ del_X[:,i+1]),(Task_Space_Control.n,)))  # joint_velocity = K * inv(J) @ [X - Xg]
            writedata.time = 0.25

            q_plot = np.vstack((q_plot, np.radians(np.array(get_current_posj())))) # In radians
            
            self.speed_publisher.publish(writedata)  # Publish Joint Velocity in degree/sec
            time.sleep(0.01) # Sleep for 0.05sec

            #print(f"error at {i}th iteration: {np.linalg.norm(del_X[:,i])}")

            i = i + 1
        
        print(f"time taken: {time.time() - start}")  # time it took to perform the manipulation
        print("Stop returns: " + str(self.stop_control()))
        print("Disconnect returns: " + str(self.drop_control()))

        (Row,Column) = del_X.shape
        Plot.figure()
        rms_values = np.sqrt(np.mean(del_X**2, axis=0)) # Compute RMS along each column
        Plot.plot(range(1,Column),del_X[0,1:Column], 'r--')
        Plot.plot(range(1,Column),del_X[1,1:Column], 'b--')
        Plot.plot(range(1,Column),del_X[2,1:Column], 'g--')
        Plot.plot(range(1,len(rms_values)),rms_values[1:], 'k-')

        Plot.xlabel('No of Iteration')
        Plot.ylabel('$\Delta$X')
        Plot.legend(['$e_{X}$','$e_{Y}$','$e_{Z}$','$RMS_{Error}$'])
        Plot.grid()
                
        Plot.figure()
        Plot.plot(range(0,i+1),d_theta[0,:], 'r-')
        Plot.plot(range(0,i+1),d_theta[1,:], 'b-')
        Plot.plot(range(0,i+1),d_theta[2,:], 'g-')
        Plot.plot(range(0,i+1),d_theta[3,:], 'y-')
        Plot.plot(range(0,i+1),d_theta[4,:], 'c-')
        Plot.plot(range(0,i+1),d_theta[5,:], 'm-')

        Plot.xlabel('No of Iteration')
        Plot.ylabel('$Joint Velocity$')
        Plot.legend(['$J_{1}$','$J_{2}$','$J_{3}$','$J_{4}$','$J_{5}$','$J_{6}$'])
        Plot.grid()

        # Now plot the data
        joints = ['theta_1','theta_2','theta_3','theta_4','theta_5','theta_6']
        Plot.figure(figsize=(20,10))
        Plot.tight_layout(pad=3.0) # give some spacing btw two subplots
        for i in range(1,Task_Space_Control.n+1):
            ax = Plot.subplot(2,3,i) # math.ceil() will round the value to upper limit
            Plot.plot(range(0,Column), np.degrees(q_plot[:,i-1]), '-.')
            Plot.ylim(-360,360)
            Plot.grid()
            Plot.ylabel(joints[i-1])
            Plot.xlabel('iteration')
        Plot.show()

        rospy.spin()  # To stop the loop and program by pressing ctr + C 

""" 
when this file is being executted no class or function run directly the only condition which run first is the 
line which has 0 indentation (except function and class) so only {if __name__ == '__main__':} is left which 
will be executted first and that will call turtle_move() class and move2goal() method. 
"""

if __name__ == "__main__":
    try:
        rospy.init_node('My_service_node')  # Initialize the ROS node

        task = Task_Space_Control()  # task = Instance / object of Task_space_Control class 

        rospy.sleep(1)  # Give buffer time for Service to activate 

        #====================================== START ACTION =============================================#

        # GOAL POSITION
        goal_Px = 0.2
        goal_Py = 0.3
        goal_Pz = 0.7

        del_X = np.ones((3,1))  # defining an array to start the loop

        task.JLA_1(del_X)  # Publish new joint angle using JLA_1

    except rospy.ROSInterruptException:
        pass 
