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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../common/imp"))) # get import path : DSR_ROBOT.py 

from matplotlib import pyplot as Plot
from common_for_JLA import *

import DR_init  # at doosan-robot/common/imp/
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *  # at doosan-robot/common/imp/
from DR_common import *  # at doosan-robot/common/imp/

# Importing messages and services 
from dsr_msgs.msg import RobotStop, RobotState  # at doosan-robot/dsr_msgs/msg/
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
        # Initialize the ROS node
        rospy.init_node('My_service_node')
        rospy.on_shutdown(self.shutdown)  # A function named 'shutdown' to be called when the node is shutdown.

        rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')  # Wait until the service becomes available

        set_robot_mode_proxy  = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
        set_robot_mode_proxy(ROBOT_MODE_AUTONOMOUS)  # Calls the service proxy and pass the args:robot_mode, to set the robot mode to ROBOT_MODE_AUTONOMOUS.

        # Creates a publisher on the topic '/dsr01a0509/stop' to publish RobotStop messages with a queue size of 10.
        self.my_publisher = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)  

        # Initialize the service client to set the pose of the manipulator in Gazebo
        rospy.wait_for_service('/dsr01a0509/motion/move_joint')  # Wait until the service becomes available
        self.move_joint = rospy.ServiceProxy('/dsr01a0509/motion/move_joint', MoveJoint)  # create service client to move joint

        rospy.wait_for_service('/dsr01a0509/motion/fkin')  # Wait until the service becomes available
        self.fkin = rospy.ServiceProxy("/dsr01a0509/motion/fkin", Fkin)  # create service client to calculate forward_kinematics

        # Initialize the joint angles
        self.current_Px = None
        self.current_Py = None
        self.current_Pz = None
        
    def call_back_func(self,q):
        current_pose = self.fkin(np.degrees(q).tolist(),DR_WORLD).conv_posx
        self.current_Px = 0.001*current_pose[0]
        self.current_Py = 0.001*current_pose[1]
        self.current_Pz = 0.001*current_pose[2]

    def shutdown(self):
        print("shutdown time!")
        print("shutdown time!")

        # '/my_node' is publishing data using publisher named 'my_publisher' to the topic '/dsr01a0509/stop'
        self.my_publisher.publish(stop_mode=STOP_TYPE_QUICK)
        return 

    def JLA_1(self,del_X):  # Inequality Constraint Method
        i = 0
        q = np.radians(np.array(get_current_posj())) #np.array([0,0,0,0,0,0]) # Hardware & Gazebo Joint Angle at Home Position

        q_plot = q

        joint_offset = np.array([0, -pi/2, pi/2, 0, 0, 0])  # Difference btw Hardware/Gazebo & DH convention
        theta = q + joint_offset  # Initial Joint position as per DH convention
        
        Jc = np.eye((Task_Space_Control.n))  # Jacobian of the additional task
        epsilon = 0.5*np.ones(np.expand_dims(q, axis=0).shape) # Activation buffer region width
        
        q_range = np.radians(np.array([[-360,360],[-95,95],[-135,135],[-360,360],[-135,135],[-360,360]]))
        
        start = time.time()
        while np.linalg.norm(del_X[:,i]) > 0.001:      
            self.call_back_func(q)  # update pose -> self.current_Px, self.current_Py, self.current_Pz 

            # Check if each element of q lies within the corresponding range in q_range
            if np.all((q_range[:, 0] <= q) & (q <= q_range[:, 1]), axis=0) == 'False':
                rospy.signal_shutdown('Joint Limit breached')

            # Calculating Weight Matrix
            """
            We(mxm) = assign priority to main tasks
            Wc(kxk) = assign priority to additional tasks 
            Wv(nxn) = assign priority to singularity tasks
            """
            We, Wc, Wv = weight_Func(Task_Space_Control.m,Task_Space_Control.n,q_range,np.expand_dims(q, axis=0),epsilon)

            # Calculate Jacobain
            J = jacobian_matrix(Task_Space_Control.n,Task_Space_Control.alpha,Task_Space_Control.a,Task_Space_Control.d,theta,Task_Space_Control.le)  # Calculate J
            Je = J[0:3,:]

            Jn = np.linalg.inv(np.transpose(Je) @ We @ Je + np.transpose(Jc) @ Wc @ Jc + Wv) @ np.transpose(Je) @ We

            # Newton Raphson Method
            del_X = np.hstack((del_X, np.array([[goal_Px - self.current_Px],[goal_Py - self.current_Py],[goal_Pz - self.current_Pz]])))

            # ADAPTIVE GAIN: formulation
            e = np.array([[goal_Px - self.current_Px],[goal_Py - self.current_Py],[goal_Pz - self.current_Pz]])
            e0 = e if i == 0 else e0
            K = 15*exp((np.linalg.norm(e0) - np.linalg.norm(e))/np.linalg.norm(e0))  # ADAPTIVE GAIN factor
            
            # Calculating Next joint Position
            theta_new = np.degrees(theta) + K * np.reshape((Jn @ del_X[:,i+1]),(Task_Space_Control.n,))  # In degrees
            q = np.radians(theta_new) - joint_offset  # for Hardware and Gazebo, # In radians
            
            q_plot = np.vstack((q_plot, q)) # In radians
    
            theta = np.radians(theta_new)  # for DH-convention, In radians

            i = i + 1

            if i % 3 == 0:
                self.move_joint(np.degrees(q), 30, 10, 0, 0, 0, 0, 0)
                print(f"error at {i}th iteration: {np.linalg.norm(del_X[:,i])}")
        
        self.move_joint(np.degrees(q), 30, 10, 0, 0, 0, 0, 0)
        print(f"error at {i}th iteration: {np.linalg.norm(del_X[:,i])}")
        
        print(f"time taken: {time.time() - start}")  # time it took to perform the manipulation

        (Row,Column) = del_X.shape
        fig_1 = Plot.figure()

        Plot.plot(range(1,Column),del_X[0,1:Column], 'r-')
        Plot.plot(range(1,Column),del_X[1,1:Column], 'b-')
        Plot.plot(range(1,Column),del_X[2,1:Column], 'g-')

        Plot.xlabel('No of Iteration')
        Plot.ylabel('$\Delta$X')
        Plot.legend(['$e_{X}$','$e_{Y}$','$e_{Z}$'])
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
        goal_Px = -0.5
        goal_Py = -0.27
        goal_Pz = 0.5

        del_X = np.ones((3,1))  # defining an array to start the loop

        task = Task_Space_Control()  # task = Instance / object of Task_space_Control class 
        rospy.sleep(0.5)  # Give buffer time for Service to activate 
        task.JLA_1(del_X)  # Publish new joint angle using JLA_1

    except rospy.ROSInterruptException:
        pass 