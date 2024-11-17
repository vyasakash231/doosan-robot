#! /usr/bin/python3
import rospy
import os
from math import *
from threading import Lock, Thread
import numpy as np
import time
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp")))

from common_for_JLA import *
from robot_RT_state import RT_STATE

import DR_init
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *
from DR_common import *

from threading import Lock, Thread
from dsr_msgs.srv import ReadDataRT, ReadDataRTRequest, WriteDataRT, WriteDataRTRequest, SetRobotMode, SetRobotModeRequest

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.autolayout'] = True  # Better layout handling
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

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
        plt.show(block=False)
        self.fig.canvas.flush_events()

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

class ReadDataRtNode:
    def __init__(self):
        rospy.init_node('read_data_rt')
        self.client_ = None
        self.client_thread_ = None
        self.first_structure_printed = False
        
        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        
        self.initialize_client()

    def print_data_structure(self, data):
        if not self.first_structure_printed:
            rospy.loginfo("\n=== Data Structure Analysis ===")
            rospy.loginfo(f"Type of data: {type(data)}")
            
            attributes = dir(data)
            rospy.loginfo("\nAvailable attributes:")
            for attr in attributes:
                if not attr.startswith('_'):
                    try:
                        value = getattr(data, attr)
                        rospy.loginfo(f"  {attr}: {type(value)}")
                        if isinstance(value, (list, np.ndarray)):
                            rospy.loginfo(f"    Length: {len(value)}")
                            if len(value) > 0:
                                rospy.loginfo(f"    First element: {value[0]}")
                    except Exception as e:
                        rospy.loginfo(f"  {attr}: <error reading value: {e}>")
            
            self.first_structure_printed = True

    def _plot_data(self, data):
        try:
            # Update the plot
            self.plotter.update_data(data.external_joint_torque, data.raw_force_torque, data.external_tcp_force)
        except Exception as e:
            rospy.logwarn(f"Error updating plot data: {e}")

    def print_realtime_data(self, data):
        """Print the real-time data values"""
        rospy.loginfo("\n=== Real-time Data Values ===")
        rospy.loginfo(f"External Joint Torque: {[round(x, 3) for x in data.external_joint_torque]}")
        rospy.loginfo(f"Raw Force Torque: {[round(x, 3) for x in data.raw_force_torque]}")
        rospy.loginfo(f"TCP force w.r.t. base coordinates: {[round(x, 3) for x in data.external_tcp_force]}")

    def initialize_client(self):
        try:
            rospy.wait_for_service('/dsr01a0509/realtime/read_data_rt', timeout=5.0)
            self.client_ = rospy.ServiceProxy('/dsr01a0509/realtime/read_data_rt', ReadDataRT)
            self.client_thread_ = Thread(target=self.read_data_rt_client)
            self.client_thread_.daemon = True
            self.client_thread_.start()
        except rospy.ROSException as e:
            rospy.logerr(f"Service not available: {e}")
            sys.exit(1)

    def read_data_rt_client(self):
        global Robot_RTState
        rate = rospy.Rate(10)  # Reduced rate for debugging
        
        while not rospy.is_shutdown():
            try:
                request = ReadDataRTRequest()
                response = self.client_(request)
                
                if response and hasattr(response, 'data'):
                    self.print_data_structure(response.data)
                    # self.print_realtime_data(response.data)
                    # self._plot_data(response.data)
                    
                    with mtx:
                        if Robot_RTState is not None:
                            Robot_RTState.time_stamp = response.data.time_stamp
                            for i in range(6):
                                Robot_RTState.actual_joint_position[i] = response.data.actual_joint_position[i]
                                Robot_RTState.actual_joint_velocity[i] = response.data.actual_joint_velocity[i]
                                Robot_RTState.gravity_torque[i] = response.data.gravity_torque[i]
                                Robot_RTState.external_joint_torque[i] = response.data.external_joint_torque[i]
                                Robot_RTState.raw_force_torque[i] = response.data.raw_force_torque[i]
                                Robot_RTState.external_tcp_force[i] = response.data.external_tcp_force[i]

            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
            except Exception as e:
                rospy.logerr(f"Error in read_data_rt_client: {e}")
            
            rate.sleep()

# Global variables
Robot_RTState = None
mtx = Lock()

def main():
    try:
        global Robot_RTState
        Robot_RTState = RT_STATE()
        node = ReadDataRtNode()
        
        # Keep both the ROS node and the plot window running
        while not rospy.is_shutdown():
            plt.pause(0.1)  # Reduced update frequency
            
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

