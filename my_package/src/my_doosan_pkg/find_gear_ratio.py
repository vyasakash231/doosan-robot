#! /usr/bin/python3
import os
import time
import sys
import rospy
from math import *
from threading import Lock, Thread
import numpy as np
np.set_printoptions(suppress=True)

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp")))

from common_for_JLA import *
from robot_RT_state import RT_STATE

import DR_init
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *
from DR_common import *

from dsr_msgs.msg import *
from dsr_msgs.srv import *
from sensor_msgs.msg import JointState

mtx = Lock()


class Gear_Ratio:
    def __init__(self):
        self.is_rt_connected = False
        self.shutdown_flag = False  # Add flag to track shutdown state
        self.Robot_RT_State = RT_STATE()

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

                with mtx:
                    self.Robot_RT_State.store_data(response.data)
                    
            except (rospy.ServiceException, rospy.ROSException) as e:
                if not self.shutdown_flag:  # Only log if we're not shutting down
                    rospy.logwarn(f"Service call failed: {e}") 
            rate.sleep()

    def get_single_joint_torque(self, joint_idx, amplitude=1.0, frequency=0.5):
        """
        Generate sinusoidal torque for a single joint while keeping others at 0
        joint_idx: Joint to test (0-5)
        amplitude: Torque amplitude in Nm
        frequency: Frequency of oscillation in Hz
        """
        torque = np.zeros(6)
        t = (rospy.Time.now() - self.start_time).to_sec()
        torque[joint_idx] = amplitude * np.sin(2 * np.pi * frequency * t)
        return torque

    def torque_control(self):
        """Implements torque control and collects gear ratio data"""
        rate = rospy.Rate(1000)
        
        # Initialize arrays to store data
        data_points = 1000  # Number of samples per joint
        gear_ratios = np.zeros((6, data_points))  # Store ratios for each joint
        current_joint = 0  # Start with first joint
        sample_count = 0
        
        self.start_time = rospy.Time.now()
        
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                with mtx:
                    # Generate torque command for current joint
                    torque = self.get_single_joint_torque(current_joint, amplitude=7.0)
                    
                    # Create and send torque command
                    writedata = TorqueRTStream()
                    writedata.tor = torque.tolist()  # Convert numpy array to list
                    writedata.time = 0.001
                    self.torque_publisher.publish(writedata)
                    
                    # Calculate and store gear ratio
                    # Avoid division by zero
                    motor_torques = np.array(self.Robot_RT_State.actual_motor_torque)
                    joint_torques = np.array(self.Robot_RT_State.actual_joint_torque)
                    
                    # Only calculate ratio when motor torque is significant
                    if abs(motor_torques[current_joint]) > 0.01:
                        gear_ratios[current_joint, sample_count] = joint_torques[current_joint] / motor_torques[current_joint]
                        sample_count += 1
                    
                    # Move to next joint after collecting enough samples
                    if sample_count >= data_points:
                        # Calculate average gear ratio for current joint
                        avg_ratio = np.median(gear_ratios[current_joint, :])
                        rospy.loginfo(f"Joint {current_joint} estimated gear ratio: {avg_ratio:.2f}")
                        
                        # Move to next joint
                        current_joint += 1
                        sample_count = 0
                        
                        # Exit if we've tested all joints
                        if current_joint >= 6:
                            break          
                    rate.sleep()
                    
            # Print final results
            final_ratios = np.median(gear_ratios, axis=1)
            rospy.loginfo("Final estimated gear ratios:")
            for joint, ratio in enumerate(final_ratios):
                rospy.loginfo(f"Joint {joint}: {ratio:.2f}")
                
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()

if __name__ == "__main__":
    try:
        # Initialize ROS node first
        rospy.init_node('my_node')
        
        # Create control object
        task = Gear_Ratio()
        rospy.sleep(1)  # Give time for initialization
        
        # Start impedance control in a separate thread
        control_thread = Thread(target=lambda: task.torque_control())
        control_thread.daemon = True
        control_thread.start()
            
    except rospy.ROSInterruptException:
        pass
