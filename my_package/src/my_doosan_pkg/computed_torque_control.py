#! /usr/bin/python3
import os
import time
import sys
import rospy
from math import *
from threading import Lock, Thread
import numpy as np
np.set_printoptions(suppress=True)

import matplotlib
matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from queue import Queue, Empty

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp")))

from common_for_JLA import *
from robot_RT_state import RT_STATE
from plot import RealTimePlot
from filters import Filters

import DR_init
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *
from DR_common import *

from dsr_msgs.msg import *
from dsr_msgs.srv import *
from sensor_msgs.msg import JointState

mtx = Lock()


class Torque_Control():
    n = 6  # No of joints

    # DH Parameters
    alpha = np.array([0, -pi/2, 0, pi/2, -pi/2, pi/2])   
    a = np.array([0, 0, 0.409, 0, 0, 0])
    d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])
    le = 0

    def __init__(self, dt):
        self.is_rt_connected = False
        self.shutdown_flag = False  # Add flag to track shutdown state
        self.Robot_RT_State = RT_STATE()

        self.dt = dt

        self.filter = Filters(dt)

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_plots_2()

        # Initialize RT control services
        self.initialize_rt_service_proxies()
        
        self.my_publisher = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)

        self.speedj_publisher = rospy.Publisher('/dsr01a0509/speedj_rt_stream', SpeedJRTStream, queue_size=10)        
        self.speedl_publisher = rospy.Publisher('/dsr01a0509/servol_rt_stream', ServoLRTStream, queue_size=10)         
        self.torque_publisher = rospy.Publisher('/dsr01a0509/torque_rt_stream', TorqueRTStream, queue_size=10)

        self.RT_observer_client = rospy.ServiceProxy('/dsr01a0509/realtime/read_data_rt', ReadDataRT)
        self.RT_writer_client = rospy.ServiceProxy('/dsr01a0509/realtime/write_data_rt', WriteDataRT)

        self.client_thread_ = Thread(target=self.read_data_rt_client)
        self.client_thread_.daemon = True  # Make thread daemon so it exits when main thread exits
        self.client_thread_.start()

        rospy.on_shutdown(self.cleanup)

    def initialize_rt_service_proxies(self):
        try:
            service_timeout = 5.0
            services = [
                ('/dsr01a0509/system/set_robot_mode', SetRobotMode),
                ('/dsr01a0509/realtime/connect_rt_control', ConnectRTControl),
                ('/dsr01a0509/realtime/set_rt_control_input', SetRTControlInput),
                ('/dsr01a0509/realtime/set_rt_control_output', SetRTControlOutput),
                ('/dsr01a0509/realtime/start_rt_control', StartRTControl),
                ('/dsr01a0509/realtime/stop_rt_control', StopRTControl),
                ('/dsr01a0509/realtime/disconnect_rt_control', DisconnectRTControl),
            ]

            # Wait for all services with timeout
            for service_name, _ in services:
                try:
                    rospy.wait_for_service(service_name, timeout=service_timeout)
                except rospy.ROSException as e:
                    rospy.logerr(f"Service {service_name} not available: {e}")
                    raise

            # Create service proxies
            self.set_robot_mode = rospy.ServiceProxy(services[0][0], services[0][1])
            self.connect_rt_control = rospy.ServiceProxy(services[1][0], services[1][1])
            self.set_rt_control_input = rospy.ServiceProxy(services[2][0], services[2][1])
            self.set_rt_control_output = rospy.ServiceProxy(services[3][0], services[3][1])
            self.start_rt_control = rospy.ServiceProxy(services[4][0], services[4][1])
            self.stop_rt_control = rospy.ServiceProxy(services[5][0], services[5][1])
            self.disconnect_rt_control = rospy.ServiceProxy(services[6][0], services[6][1])

            self.joint_vel_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velj_rt', SetVelJRT)
            self.joint_acc_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accj_rt', SetAccJRT)
            self.ee_vel_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velx_rt', SetVelXRT)
            self.ee_acc_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accx_rt', SetAccXRT)

            self.connect_to_rt_control()

        except Exception as e:
            rospy.logerr(f"Failed to initialize RT control services: {e}")
            sys.exit(1)

    def connect_to_rt_control(self):
        try:
            mode_req = SetRobotModeRequest()
            mode_req.robot_mode = ROBOT_MODE_AUTONOMOUS
            robot_mode = self.set_robot_mode(mode_req)
            if not robot_mode.success:
                raise Exception("Failed to set robot mode")
            rospy.loginfo("Robot set to autonomous mode")

            connect_req = ConnectRTControlRequest()
            connect_req.ip_address = "192.168.137.100"
            connect_req.port = 12347
            connect_response = self.connect_rt_control(connect_req)
            if not connect_response.success:
                raise Exception("Failed to connect RT control")
            
            set_output_req = SetRTControlOutputRequest()
            set_output_req.period = 0.001
            set_output_req.loss = 4
            set_output_response = self.set_rt_control_output(set_output_req)
            if not set_output_response.success:
                raise Exception("Failed to set RT control output")

            start_response = self.start_rt_control(StartRTControlRequest())
            if not start_response.success:
                raise Exception("Failed to start RT control")
                        
            self.is_rt_connected = True
            rospy.loginfo("Successfully connected to RT control")

            self.joint_vel_limits([150, 150, 150, 150, 150, 150])  # Increased from 50 deg/s
            self.joint_acc_limits([100, 100, 100, 100, 100, 100])  # Increased from 25 deg/s^2

        except Exception as e:
            rospy.logerr(f"Failed to establish RT control connection: {e}")
            self.cleanup()
            sys.exit(1)

    def cleanup(self):
        """Improved cleanup function with better error handling"""
        if self.shutdown_flag:  # Prevent multiple cleanup calls
            return
        
        self.shutdown_flag = True
        rospy.loginfo("Initiating cleanup process...")

        try:
            # Send stop command first
            stop_msg = RobotStop()
            stop_msg.stop_mode = 1  # STOP_TYPE_QUICK
            self.my_publisher.publish(stop_msg)
            rospy.sleep(0.1)  # Give time for stop command to process

            if self.is_rt_connected:
                try:
                    # Stop RT control with timeout
                    stop_future = self.stop_rt_control(StopRTControlRequest())
                    rospy.sleep(0.5)
                    
                    # Disconnect RT control
                    if not rospy.is_shutdown():  # Only try to disconnect if ROS isn't shutting down
                        self.disconnect_rt_control(DisconnectRTControlRequest())
                    
                    self.is_rt_connected = False
                    rospy.loginfo("RT control cleanup completed successfully")
                
                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn(f"Non-critical error during cleanup: {e}")
                    # Continue cleanup process despite errors
        
        except Exception as e:
            rospy.logerr(f"Critical error during cleanup: {e}")
        finally:
            rospy.loginfo("Cleanup process finished")

    def read_data_rt_client(self):
        rate = rospy.Rate(1000)
        
        while not rospy.is_shutdown() and not self.shutdown_flag:
            try:
                if not self.is_rt_connected:
                    rate.sleep()
                    continue

                request = ReadDataRTRequest()
                response = self.RT_observer_client(request)

                with mtx:
                    self.Robot_RT_State.store_data(response.data)
                    
            except (rospy.ServiceException, rospy.ROSException) as e:
                if not self.shutdown_flag:  # Only log if we're not shutting down
                    rospy.logwarn(f"Service call failed: {e}") 
            rate.sleep()

    def _plot_data(self):
        """Thread-safe plotting function with joint errors"""
        try:
            # Calculate joint errors if we have desired trajectory data
            joint_errors = None
            if hasattr(self, 'current_desired_position'):
                current_position = np.array(self.Robot_RT_State.actual_joint_position_abs)
                joint_errors = self.current_desired_position - current_position
            
            self.plotter.update_data_2(self.Robot_RT_State.actual_motor_torque, self.Robot_RT_State.external_tcp_force, self.Robot_RT_State.raw_force_torque, joint_errors)
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    def computed_torque_control(self, Kp, Kd, qd, qd_dot, qd_ddot):
        rate = rospy.Rate(1000)
        i = 0
        
        total_points = qd.shape[1]
        
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag and i < total_points:            
                with mtx:
                    # Store current desired position for plotting
                    self.current_desired_position = qd[:,i]

                    # Plot Torque
                    self._plot_data()

                    G_torque = self.Robot_RT_State.gravity_torque
                    C_matrix = self.Robot_RT_State.coriolis_matrix
                    M_matrix = self.Robot_RT_State.mass_matrix

                    q = self.Robot_RT_State.actual_joint_position_abs
                    q_dot = self.Robot_RT_State.actual_joint_velocity_abs

                    # Calculate errors
                    E = qd[:,i] - q
                    E_dot = qd_dot[:,i] - q_dot

                    # Print progress
                    if i % 500 == 0:
                        print(f"Progress: {i}/{total_points} points ({(i/total_points)*100:.1f}%)")
                        # print(f"Position error (deg): {E}")
                        print(f"Max error: {np.max(np.abs(E))}")
                        print("---------------------------------------------------------------------")

                    # Feed-back PD-control Input with reduced gains
                    u = Kp @ E[:, np.newaxis] + Kd @ E_dot[:, np.newaxis]

                    # Compute control torque
                    Torque = M_matrix @ (qd_ddot[:,[i]] + u) + C_matrix @ q_dot[:, np.newaxis] + G_torque[:, np.newaxis]

                    Torque = self.filter.low_pass_filter_torque(Torque)  # Apply low-pass filter to smooth torque
                    # Torque = self.filter.moving_average_filter(Torque)  # Apply moving average filter
                    # Torque = self.filter.smooth_torque(Torque)  # Apply second-order filter
                    
                    # Add torque limits
                    torque_limits = np.array([70, 70, 70, 70, 70, 70])
                    Torque = np.clip(Torque, -torque_limits[:, np.newaxis], torque_limits[:, np.newaxis])

                    # Send torque command
                    writedata = TorqueRTStream()
                    writedata.tor = Torque
                    writedata.time = 1.0 * self.dt
                    
                    self.torque_publisher.publish(writedata)

                rate.sleep()
                i += 1
            print(f"Control loop finished. Completed {i}/{total_points} points")
            
        except Exception as e:
            print(f"Error in control loop: {e}")
        finally:
            self.cleanup()



def generate_quintic_trajectory(q0, qf, t0, tf, dt=0.005):
    """
    Generate a quintic polynomial trajectory between two points.
    
    Args:
        q0: Initial position
        qf: Final position
        t0: Initial time
        tf: Final time
        num_points: Number of points in the trajectory
    
    Returns:
        t: Time points
        q: Position trajectory
        qd: Velocity trajectory
        qdd: Acceleration trajectory
    """
    # Time vector
    t = np.arange(t0, tf, dt)
    
    # Time parameters
    T = tf - t0
    
    # Quintic polynomial coefficients
    a0 = q0
    a1 = 0  # Initial velocity = 0
    a2 = 0  # Initial acceleration = 0
    a3 = 10 * (qf - q0) / T**3
    a4 = -15 * (qf - q0) / T**4
    a5 = 6 * (qf - q0) / T**5
    
    # Compute position, velocity, and acceleration
    q = a0 + a1*(t-t0) + a2*(t-t0)**2 + a3*(t-t0)**3 + a4*(t-t0)**4 + a5*(t-t0)**5
    qd = a1 + 2*a2*(t-t0) + 3*a3*(t-t0)**2 + 4*a4*(t-t0)**3 + 5*a5*(t-t0)**4
    qdd = 2*a2 + 6*a3*(t-t0) + 12*a4*(t-t0)**2 + 20*a5*(t-t0)**3
    return t, q, qd, qdd

def pre_process_trajectory(tf, dt):
    # Define initial and final joint angles (in degrees)
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    qf = np.array([90.0, 45.0, -45.0, 45.0, -45.0, 180.0])

    # Increased time for slower motion
    t0 = 0.0
    t = np.arange(t0, tf, dt)
    q_list = np.zeros((6, len(t)))
    q_dot_list = np.zeros((6, len(t)))
    q_ddot_list = np.zeros((6, len(t)))

    for i in range(6):
        _, q, qd, qdd = generate_quintic_trajectory(q0[i], qf[i], t0, tf, dt)
        q_list[i, :len(q)] = q
        q_dot_list[i, :len(qd)] = qd
        q_ddot_list[i, :len(qdd)] = qdd

    return t, q_list, q_dot_list, q_ddot_list


if __name__ == "__main__":
    try:
        # Initialize ROS node first
        rospy.init_node('My_service_node')
        dt = 0.002
        t, qd, qd_dot, qd_ddot = pre_process_trajectory(tf=10.0, dt=dt)
        
        # Create control object
        task = Torque_Control(dt)
        rospy.sleep(2.5)  # Give time for initialization

        Kp = np.diag([2.5, 2.5, 3.0, 3.5, 30.0, 300.0]) 
        Kd = np.diag([0.5, 0.5, 0.5, 0.5, 2.0, 20.0])
        
        # Start impedance control in a separate thread
        control_thread = Thread(target=lambda: task.computed_torque_control(Kp, Kd, qd, qd_dot, qd_ddot))
        control_thread.daemon = True
        control_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.05)  # This keeps the plot window responsive
            
    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close('all')  # Clean up plots on exit