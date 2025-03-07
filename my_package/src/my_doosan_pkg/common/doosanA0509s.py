#! /usr/bin/python3
import os
import time
import sys
from abc import ABC, abstractmethod
import rospy
from math import *
from threading import Lock, Thread
import numpy as np
np.set_printoptions(suppress=True)

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp")))

import DR_init
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *
from DR_common import *

from dsr_msgs.msg import *
from dsr_msgs.srv import *
from sensor_msgs.msg import JointState


class Robot(ABC):
    n = 6  # No of joints

    # DH Parameters
    alpha = np.array([0, -pi/2, 0, pi/2, -pi/2, pi/2])   
    a = np.array([0, 0, 0.409, 0, 0, 0])
    d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])
    le = 0

    def __init__(self):
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

            self.joint_vel_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velj_rt', SetVelJRT)   # The global joint velocity is set in (deg/sec)
            self.joint_acc_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accj_rt', SetAccJRT)  # The global joint acceleration is set in (deg/sec^2)
            self.ee_vel_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velx_rt', SetVelXRT)   # This function sets the velocity of the task space motion globally in (mm/sec)
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
        rate = rospy.Rate(1000)  # 1000 Hz
        
        while not rospy.is_shutdown() and not self.shutdown_flag:
            try:
                if not self.is_rt_connected:
                    rate.sleep()
                    continue

                request = ReadDataRTRequest()
                response = self.RT_observer_client(request)

                # Plot Data
                self.plot_data(response.data)
                
                # Store Real-Time data
                self.Robot_RT_State.store_data(response.data)
                    
            except (rospy.ServiceException, rospy.ROSException) as e:
                if not self.shutdown_flag:  # Only log if we're not shutting down
                    rospy.logwarn(f"Service call failed: {e}") 
            rate.sleep()

    @abstractmethod   # Force child classes to implement this method
    def plot_data(self, data):
        pass  