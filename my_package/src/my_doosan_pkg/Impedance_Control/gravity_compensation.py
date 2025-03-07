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

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp")))

from common_for_JLA import *
from robot_RT_state import RT_STATE
from utils import *
from plot import RealTimePlot
from doosanA0509s import Robot

import DR_init
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *
from DR_common import *

from dsr_msgs.msg import *
from dsr_msgs.srv import *
from sensor_msgs.msg import JointState


class GravityCompensation(Robot):
    def __init__(self):
        self.is_rt_connected = False
        self.shutdown_flag = False  
        self.Robot_RT_State = RT_STATE()

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_plots_1()

        super().__init__()

    def plot_data(self, data):
        try:
            self.plotter.update_data_1(data.actual_motor_torque, data.external_tcp_force, data.raw_force_torque)
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    def run_controller(self):
        rate = rospy.Rate(1000)
        
        try:
            start_time = rospy.Time.now()
            while not rospy.is_shutdown() and not self.shutdown_flag:
                G_torques = self.Robot_RT_State.gravity_torque
                t = (rospy.Time.now() - start_time).to_sec()

                print(self.Robot_RT_State.actual_flange_position, self.Robot_RT_State.actual_tcp_position)
                
                writedata = TorqueRTStream()
                writedata.tor = G_torques
                writedata.time = 0.001
                
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
        task = GravityCompensation()
        rospy.sleep(1)  # Give time for initialization
        
        # Start G control in a separate thread
        control_thread = Thread(target=lambda: task.run_controller())
        control_thread.daemon = True
        control_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.1)  # This keeps the plot window responsive
            
    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close('all')  # Clean up plots on exit










