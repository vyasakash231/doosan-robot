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

    def __init__(self):
        self.is_rt_connected = False
        self.shutdown_flag = False  # Add flag to track shutdown state
        self.Robot_RT_State = RT_STATE()

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()

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
            set_output_req.period = 0.01
            set_output_req.loss = 5
            set_output_response = self.set_rt_control_output(set_output_req)
            if not set_output_response.success:
                raise Exception("Failed to set RT control output")

            start_response = self.start_rt_control(StartRTControlRequest())
            if not start_response.success:
                raise Exception("Failed to start RT control")
                        
            self.is_rt_connected = True
            rospy.loginfo("Successfully connected to RT control")

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

                # Plot Torque
                self._plot_data(response.data)

                with mtx:
                    self.Robot_RT_State.store_data(response.data)
                    
            except (rospy.ServiceException, rospy.ROSException) as e:
                if not self.shutdown_flag:  # Only log if we're not shutting down
                    rospy.logwarn(f"Service call failed: {e}") 
            rate.sleep()

    def _plot_data(self, data):
        """Thread-safe plotting function"""
        try:
            self.plotter.update_data(data.actual_motor_torque, data.external_tcp_force)
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    def gravity_compensation(self):
        """Implements torque control with gravity compensation"""
        rate = rospy.Rate(1000)
        
        try:
            start_time = rospy.Time.now()
            while not rospy.is_shutdown() and not self.shutdown_flag:
                with mtx:
                    G_torques = self.Robot_RT_State.gravity_torque
                    t = (rospy.Time.now() - start_time).to_sec()
                    
                    writedata = TorqueRTStream()
                    writedata.tor = G_torques
                    writedata.time = 0.001
                    
                    self.torque_publisher.publish(writedata)
                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()

    def impedence_control_static(self, Kd, Dd, Md, Xd, Xd_dot):
        """Implements torque control with gravity compensation"""
        rate = rospy.Rate(1000)
        
        try:
            start_time = rospy.Time.now()
            while not rospy.is_shutdown() and not self.shutdown_flag:
                with mtx:
                    G_torque = self.Robot_RT_State.gravity_torque
                    C_matrix = self.Robot_RT_State.coriolis_matrix
                    M_matrix = self.Robot_RT_State.mass_matrix

                    q_dot = self.Robot_RT_State.actual_joint_velocity_abs
                    Xe = self.Robot_RT_State.actual_tcp_position     #  (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
                    Xe_dot = self.Robot_RT_State.actual_tcp_velocity  # w.r.t. base coordinates in [mm, deg/s]
                    J = self.Robot_RT_State.jacobian_matrix

                    C_torque = (C_matrix @ q_dot[:, np.newaxis]).reshape(-1)

                    # task space error
                    E = Xe - Xd
                    E_dot = Xe_dot - Xd_dot

                    # Classical Impedance Controller WITHOUT contact force feedback!!
                    impedence_force = Kd @ E[:, np.newaxis] + Dd @ E_dot[:, np.newaxis]  # make E, E_dot column vector from row vector
                    impedence_tau = (J.T @ impedence_force).reshape(-1)

                    torques = C_torque.tolist() + G_torque - impedence_tau.tolist()    # Cartesian PD control with gravity cancellation
                    
                    writedata = TorqueRTStream()
                    writedata.tor = G_torque
                    writedata.time = 0.01
                    
                    self.torque_publisher.publish(writedata)
                rate.sleep() 
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()

if __name__ == "__main__":
    try:
        # Initialize ROS node first
        rospy.init_node('My_service_node')
        
        # Create control object
        task = Torque_Control()
        rospy.sleep(1)  # Give time for initialization

        Kd = np.diag([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        Dd = np.diag([0.5, 0.5, 0.5, 0.25, 0.25, 0.25])
        Md = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        x1, sol = get_current_posx()   #  x1 w.r.t. DR_BASE
        Xd = np.array(x1)  # [mm, deg]
        Xd_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [mm/s, deg/s]
        
        # Start impedance control in a separate thread
        # control_thread = Thread(target=lambda: task.impedence_control_static(Kd, Dd, Md, Xd, Xd_dot))
        control_thread = Thread(target=lambda: task.gravity_compensation())
        control_thread.daemon = True
        control_thread.start()

        # print(np.round(task.Robot_RT_State.jacobian_matrix, 3), '\n')
        # theta = task.Robot_RT_State.actual_joint_position_abs
        # print("=============================================")
        # J = jacobian_matrix(task.n, task.alpha, task.a, task.d, np.radians(theta))
        # print(J)
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.1)  # This keeps the plot window responsive
            
    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close('all')  # Clean up plots on exit