#! /usr/bin/python3
import rospy
import time
import threading
from math import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from collections import deque


class RealTimePlot:
    def __init__(self, max_points=100):
        # Create figure with two subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 12))
        self.fig.set_facecolor('white')  # White background
        plt.subplots_adjust(hspace=0.3)
        
        # Enable double buffering
        self.fig.canvas.draw()
        
        # Initialize deques for storing data
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        
        # Initialize data storage
        self.motor_torque = [deque(maxlen=max_points) for _ in range(6)]
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
        for ax, title in zip([self.ax1, self.ax2],['Actual Motor Torque', 'TCP Force']):
            ax.set_title(title, pad=10, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Time (s)', fontsize=8)
            
        self.ax1.set_ylabel('Torque (Nm)', fontsize=8)
        self.ax2.set_ylabel('Force (N)', fontsize=8)

        # Create lines with custom colors
        self.motor_lines = [self.ax1.plot([], [], label=f'Joint {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.tcp_lines = [self.ax2.plot([], [], label=f'Axis {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]

        # Add legends
        for ax in [self.ax1, self.ax2]:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

    def update_data(self, actual_motor_torque, external_tcp_force):
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

        # Convert deques to lists for plotting
        x_data = list(self.times)
        
        # Update all lines
        for i in range(6):
            self.motor_lines[i].set_data(x_data, list(self.motor_torque[i]))
            self.tcp_lines[i].set_data(x_data, list(self.tcp_forces[i]))

        # Update axis limits
        if len(x_data) > 0:
            self.ax1.set_xlim(max(0, plot_time - 10), plot_time + 0.5)
            self.ax1.set_ylim(-70, 70)
            self.ax1.relim()
            self.ax1.autoscale_view(scaley=True)

            self.ax2.set_xlim(max(0, plot_time - 10), plot_time + 0.5)
            self.ax2.set_ylim(-100, 100)
            self.ax2.relim()
            self.ax2.autoscale_view(scaley=True)

        try:
            # Use blit for faster rendering
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.last_update_time = current_time
        except Exception as e:
            rospy.logwarn(f"Error updating plot: {e}")
