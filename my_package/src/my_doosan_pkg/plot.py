#! /usr/bin/python3
import rospy
import time
import threading
from math import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque


class RealTimePlot:
    def __init__(self, max_points=100):        
        # Initialize deques for storing data
        self.max_points = max_points
        self.times = deque(maxlen=max_points)

        # Colors for the lines
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Start time for x-axis
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_interval = 0.1  # Minimum time between updates (seconds)

    '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

    def setup_plots_1(self):
        self.fig = plt.figure(figsize=(10, 12))
        gs = gridspec.GridSpec(3, 1)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0,0]))
        self.axs.append(self.fig.add_subplot(gs[1,0]))
        self.axs.append(self.fig.add_subplot(gs[2,0]))

        plt.subplots_adjust(hspace=0.3)

        # Initialize data storage
        self.motor_torque = [deque(maxlen=self.max_points) for _ in range(6)]
        self.tcp_forces = [deque(maxlen=self.max_points) for _ in range(6)]
        self.row_ft_data = [deque(maxlen=self.max_points) for _ in range(6)]        
        
        # Enable double buffering
        self.fig.canvas.draw()

        # Show the plot
        plt.show(block=False)
        self.fig.canvas.flush_events()

        # Common settings for all axes
        for ax, title in zip(self.axs,['Actual Motor Torque', 'TCP Force', 'Raw FTS Data']):
            ax.set_title(title, pad=10, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Time (s)', fontsize=8)
            
        self.axs[0].set_ylabel('Torque (Nm)', fontsize=8)
        self.axs[1].set_ylabel('Force (N)', fontsize=8)
        self.axs[2].set_ylabel('Force & Torque (N, Nm)', fontsize=8)

        # Create lines with custom colors
        self.motor_lines = [self.axs[0].plot([], [], label=f'Joint {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.tcp_lines = [self.axs[1].plot([], [], label=f'Axis {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.fts_lines = [self.axs[2].plot([], [], label=f'Axis {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]

        # Add legends
        for ax in self.axs:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

    def update_data_1(self, actual_motor_torque, external_tcp_force, raw_force_torque):
        current_time = time.time()
        
        # Limit update rate
        if current_time - self.last_update_time < self.update_interval:
            return
            
        plot_time = current_time - self.start_time
        self.times.append(plot_time)
        
        # Update data
        for i in range(6):
            self.motor_torque[i].append(actual_motor_torque[i])
            self.tcp_forces[i].append(external_tcp_force[i])
            self.row_ft_data[i].append(raw_force_torque[i])

        # Convert deques to lists for plotting
        x_data = list(self.times)
        
        # Update all lines
        for i in range(6):
            self.motor_lines[i].set_data(x_data, list(self.motor_torque[i]))
            self.tcp_lines[i].set_data(x_data, list(self.tcp_forces[i]))
            self.fts_lines[i].set_data(x_data, list(self.row_ft_data[i]))

        # Update axis limits
        limit = [70, 100, 150]
        if len(x_data) > 0:
            for idx, ax in enumerate(self.axs):
                ax.set_xlim(max(0, plot_time - 10), plot_time + 0.5)
                ax.set_ylim(-limit[idx], limit[idx])
                ax.relim()
                ax.autoscale_view(scaley=True)

        try:
            # Use blit for faster rendering
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.last_update_time = current_time
        except Exception as e:
            rospy.logwarn(f"Error updating plot: {e}")

    '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

    def setup_plots_2(self):
        self.fig = plt.figure(figsize=(10, 12))
        gs = gridspec.GridSpec(2, 2)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0,0]))
        self.axs.append(self.fig.add_subplot(gs[1,0]))
        self.axs.append(self.fig.add_subplot(gs[0,1]))
        self.axs.append(self.fig.add_subplot(gs[1,1]))

        plt.subplots_adjust(hspace=0.3)

        # Initialize data storage
        self.motor_torque = [deque(maxlen=self.max_points) for _ in range(6)]
        self.tcp_forces = [deque(maxlen=self.max_points) for _ in range(6)]
        self.row_ft_data = [deque(maxlen=self.max_points) for _ in range(6)]        
        self.joint_error_data = [deque(maxlen=self.max_points) for _ in range(6)]   

        # Enable double buffering
        self.fig.canvas.draw()

        # Show the plot
        plt.show(block=False)
        self.fig.canvas.flush_events()

        # Common settings for all axes
        for ax, title in zip(self.axs,['Actual Motor Torque', 'TCP Force', 'Raw FTS Data', 'Joint Error']): 
            ax.set_title(title, pad=10, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Time (s)', fontsize=8)
            
        self.axs[0].set_ylabel('Torque (Nm)', fontsize=8)
        self.axs[1].set_ylabel('Force (N)', fontsize=8)
        self.axs[2].set_ylabel('Force & Torque (N, Nm)', fontsize=8)
        self.axs[3].set_ylabel('Joint Error (deg)', fontsize=8)

        # Create lines with custom colors
        self.motor_lines = [self.axs[0].plot([], [], label=f'Joint {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.tcp_lines = [self.axs[1].plot([], [], label=f'Axis {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.fts_lines = [self.axs[2].plot([], [], label=f'Axis {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.error_lines = [self.axs[3].plot([], [], label=f'Axis {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]

        # Add legends
        for ax in self.axs:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

    def update_data_2(self, actual_motor_torque, external_tcp_force, raw_force_torque, joint_errors):
        current_time = time.time()
        
        # Limit update rate
        if current_time - self.last_update_time < self.update_interval:
            return
            
        plot_time = current_time - self.start_time
        self.times.append(plot_time)
        
        # Update data
        for i in range(6):
            self.motor_torque[i].append(actual_motor_torque[i])
            self.tcp_forces[i].append(external_tcp_force[i])
            self.row_ft_data[i].append(raw_force_torque[i])
            self.joint_error_data[i].append(joint_errors[i])

        # Convert deques to lists for plotting
        x_data = list(self.times)
        
        # Update all lines
        for i in range(6):
            self.motor_lines[i].set_data(x_data, list(self.motor_torque[i]))
            self.tcp_lines[i].set_data(x_data, list(self.tcp_forces[i]))
            self.fts_lines[i].set_data(x_data, list(self.row_ft_data[i]))
            self.error_lines[i].set_data(x_data, list(self.joint_error_data[i]))

        # Update axis limits
        limit = [70, 100, 150, 100]
        if len(x_data) > 0:
            for idx, ax in enumerate(self.axs):
                ax.set_xlim(max(0, plot_time - 10), plot_time + 0.5)
                ax.set_ylim(-limit[idx], limit[idx])
                ax.relim()
                ax.autoscale_view(scaley=True)

        try:
            # Use blit for faster rendering
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.last_update_time = current_time
        except Exception as e:
            rospy.logwarn(f"Error updating plot: {e}") 