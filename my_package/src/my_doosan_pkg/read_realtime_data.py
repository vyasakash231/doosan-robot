#! /usr/bin/python3
"""
The line at the top of this Python script is called a "shebang" line. It specifies the path to the Python interpreter that should be used to execute the script.
When you run a script in the terminal using ./script.py, the system looks for the shebang line at the top of the script to determine which interpreter to use. 
In this case, #!/usr/bin/python3 specifies that the system should use the python interpreter that is found in the user's PATH environment variable.
"""
import rospy
import os
from math import *
import threading
from threading import Lock
import numpy as np
import time
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp"))) # get import path : DSR_ROBOT.py 

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.autolayout'] = True  # Better layout handling
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as Plot
from collections import deque

from common_for_JLA import *
from robot_RT_state import RT_STATE

import DR_init  # at doosan-robot/common/imp/
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *  # at doosan-robot/common/imp/
from DR_common import *  # at doosan-robot/common/imp/

# Importing messages and services 
from dsr_msgs.msg import RobotStop, RobotState, SpeedJRTStream  # at doosan-robot/dsr_msgs/msg/
from dsr_msgs.srv import *
from sensor_msgs.msg import JointState

# Global variables
mtx = Lock()
first_get = False


class RealTimePlot:
    def __init__(self, max_points=100):
        # Create figure with three subplots
        plt.ion()  # Interactive mode on
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.set_facecolor('white')  # White background
        plt.subplots_adjust(hspace=0.3)
        
        # Enable double buffering
        self.fig.canvas.draw()
        
        # Initialize deques for storing data
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        
        # Initialize data storage
        self.ext_joint_torques = [deque(maxlen=max_points) for _ in range(6)]
        self.raw_force_torques = [deque(maxlen=max_points) for _ in range(6)]
        self.tcp_forces = [deque(maxlen=max_points) for _ in range(6)]
        
        # Colors for the lines
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Setup plots
        self.setup_plots()
        
        # Start time for x-axis
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_interval = 0.1  # Minimum time between updates (seconds)
        
        # Show the plot
        # plt.show(block=False)
        # self.fig.canvas.flush_events()

    def setup_plots(self):
        # Common settings for all axes
        for ax, title in zip([self.ax1, self.ax2, self.ax3], 
                           ['External Joint Torque', 'Raw Force Torque', 'TCP Force']):
            ax.set_title(title, pad=10, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Time (s)', fontsize=8)
            
        self.ax1.set_ylabel('Torque (Nm)', fontsize=8)
        self.ax2.set_ylabel('Force/Torque', fontsize=8)
        self.ax3.set_ylabel('Force (N)', fontsize=8)

        # Create lines with custom colors
        self.ext_lines = [self.ax1.plot([], [], label=f'Joint {i+1}', 
                         color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.raw_lines = [self.ax2.plot([], [], label=f'Axis {i+1}', 
                         color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.tcp_lines = [self.ax3.plot([], [], label=f'Axis {i+1}', 
                         color=self.colors[i], linewidth=1.5)[0] for i in range(6)]

        # Add legends
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

    def update_data(self, external_joint_torque, raw_force_torque, external_tcp_force):
        current_time = time.time()
        
        # Limit update rate
        if current_time - self.last_update_time < self.update_interval:
            return
            
        plot_time = current_time - self.start_time
        self.times.append(plot_time)
        
        # Update data
        for i in range(6):
            self.ext_joint_torques[i].append(external_joint_torque[i])
            self.raw_force_torques[i].append(raw_force_torque[i])
            self.tcp_forces[i].append(external_tcp_force[i])

        # Convert deques to lists for plotting
        x_data = list(self.times)
        
        # Update all lines
        for i in range(6):
            self.ext_lines[i].set_data(x_data, list(self.ext_joint_torques[i]))
            self.raw_lines[i].set_data(x_data, list(self.raw_force_torques[i]))
            self.tcp_lines[i].set_data(x_data, list(self.tcp_forces[i]))

        # Update axis limits
        if len(x_data) > 0:
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_xlim(max(0, plot_time - 10), plot_time + 0.5)
                ax.relim()
                ax.autoscale_view(scaley=True)

        try:
            # Use blit for faster rendering
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.last_update_time = current_time
        except Exception as e:
            rospy.logwarn(f"Error updating plot: {e}")


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
        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()

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
        retval = drt_setout(period = 0.001, loss = 10)  # period = 0.001 (sampling time, here 1000Hz),  loss = 10 (unknown, currently unused by doosan firmware)
        if not retval:
            raise SystemExit('realtime set output failed')
        
        # Creates a publisher on the topic '/dsr01a0509/speedj_rt_stream' to publish SpeedJRTStream messages with a queue size of 1         
        self.speed_publisher = rospy.Publisher('/dsr01a0509/speedj_rt_stream', SpeedJRTStream, queue_size=5)

        # start_rt_control!!
        drt_start = rospy.ServiceProxy('/dsr01a0509/realtime/start_rt_control', StartRTControl)
        retval = drt_start()
        if not retval:
            raise SystemExit('realtime start control failed')
        
        # drt_velj_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velj_rt', SetVelJRT)    # set global joint velocity limit
        # drt_accj_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accj_rt', SetAccJRT)    # set global joint acceleration limit
        # drt_velx_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velx_rt', SetVelXRT)    # set global Task velocity limit
        # drt_accx_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accx_rt', SetAccXRT)    # set global Task acceleration limit

        # # Set JOINT-LIMITS
        # drt_velj_limits([100, 100, 100, 100, 100, 100])
        # drt_accj_limits([100, 100, 100, 100, 100, 100])
        # drt_accx_limits(100,10)
        # drt_velx_limits(200,10)

        # Initialize RT observation client
        self.client_ = rospy.ServiceProxy('/dsr01a0509/realtime/read_data_rt', ReadDataRT)

        # stop_rt_control
        self.stop_control = rospy.ServiceProxy('/dsr01a0509/realtime/stop_rt_control', StopRTControl)

        # disconnect_rt_control_cb
        self.drop_control = rospy.ServiceProxy('/dsr01a0509/realtime/disconnect_rt_control', DisconnectRTControl)

        # Initialize the joint angles
        self.current_Px = None
        self.current_Py = None
        self.current_Pz = None

        # Start threads
        self.client_thread_ = threading.Thread(target=self.read_data_rt_client)
        self.client_thread_.start()

    def read_data_rt_client(self):
        rate = rospy.Rate(10)  # Reduced rate for debugging
        
        while not rospy.is_shutdown():
            try:            
                request = ReadDataRTRequest()
                response = self.client_(request)

                print(response.data.actual_motor_torque)

                # if response and hasattr(response, 'data'):
                #     self._plot_data(response.data)

                with mtx:
                    Robot_RTState.time_stamp = response.data.time_stamp
                    for i in range(6):
                        Robot_RTState.actual_joint_position[i] = response.data.actual_joint_position[i]
                        Robot_RTState.actual_joint_velocity[i] = response.data.actual_joint_velocity[i]
                        Robot_RTState.gravity_torque[i] = response.data.gravity_torque[i]
                        # Robot_RTState.external_joint_torque[i] = response.data.external_joint_torque[i]
                        # Robot_RTState.raw_force_torque[i] = response.data.raw_force_torque[i]
                        # Robot_RTState.external_tcp_force[i] = response.data.external_tcp_force[i]
                    
                        for j in range(6):
                            Robot_RTState.coriolis_matrix[i][j] = response.data.coriolis_matrix[i].data[j]
                            Robot_RTState.mass_matrix[i][j] = response.data.mass_matrix[i].data[j]
                            Robot_RTState.jacobian_matrix[i][j] = response.data.jacobian_matrix[i].data[j]
                    
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
            except rospy.ROSException as e:
                rospy.logwarn("Waiting for the server to be up...")

            rate.sleep()

    def _plot_data(self, data):
        try:
            # Update the plot
            self.plotter.update_data(data.external_joint_torque, data.raw_force_torque, data.external_tcp_force)
        except Exception as e:
            rospy.logwarn(f"Error updating plot data: {e}")
        
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
        # Initialize global state
        global Robot_RTState
        Robot_RTState = RT_STATE()

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
