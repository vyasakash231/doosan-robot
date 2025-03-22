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
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


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

    '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

    def setup_plots_1(self):
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
        self.row_ft_data = [deque(maxlen=self.max_points) for _ in range(6)]
        self.joint_torque_estimate = [deque(maxlen=self.max_points) for _ in range(6)]        
        self.joint_torque_sensor = [deque(maxlen=self.max_points) for _ in range(6)]   
        
        # Enable double buffering
        self.fig.canvas.draw()

        # Show the plot
        plt.show(block=False)
        self.fig.canvas.flush_events()

        # Common settings for all axes
        for ax, title in zip(self.axs,['Actual Motor Torque', 'Raw FTS Data', 'Estimate Joint Torque', 'Sensor Joint Torque']):
            ax.set_title(title, pad=10, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Time (s)', fontsize=8)
            
        self.axs[0].set_ylabel('Actual Motor Torque (Nm)', fontsize=8)
        self.axs[1].set_ylabel('Raw FTS Data (N, Nm)', fontsize=8)
        self.axs[2].set_ylabel('Estimated Joint Torque by Controller (Nm)', fontsize=8)
        self.axs[3].set_ylabel('Joint Torque Sensor Data (Nm)', fontsize=8)

        # Create lines with custom colors
        self.motor_lines = [self.axs[0].plot([], [], label=f'Joint {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.fts_lines = [self.axs[1].plot([], [], label=f'Axis {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.est_joint_lines = [self.axs[2].plot([], [], label=f'Joint {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]
        self.sensor_joint_lines = [self.axs[3].plot([], [], label=f'Joint {i+1}', color=self.colors[i], linewidth=1.5)[0] for i in range(6)]

        # Add legends
        for ax in self.axs:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

    def update_data_1(self, actual_motor_torque, raw_force_torque, joint_torque_estimate, joint_torque_sensor):
        current_time = time.time()
        
        # Limit update rate
        if current_time - self.last_update_time < self.update_interval:
            return
            
        plot_time = current_time - self.start_time
        self.times.append(plot_time)
        
        # Update data
        for i in range(6):
            self.motor_torque[i].append(actual_motor_torque[i])
            self.row_ft_data[i].append(raw_force_torque[i])
            self.joint_torque_estimate[i].append(joint_torque_estimate[i])
            self.joint_torque_sensor[i].append(joint_torque_sensor[i])

        # Convert deques to lists for plotting
        x_data = list(self.times)
        
        # Update all lines
        for i in range(6):
            self.motor_lines[i].set_data(x_data, list(self.motor_torque[i]))
            self.fts_lines[i].set_data(x_data, list(self.row_ft_data[i]))
            self.est_joint_lines[i].set_data(x_data, list(self.joint_torque_estimate[i]))
            self.sensor_joint_lines[i].set_data(x_data, list(self.joint_torque_sensor[i]))

        # Update axis limits
        limit = [70, 100, 100, 100]
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

    '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

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




########################################################################################################################################################



"""https://github.com/ozhanozen/RTplot_Python_Library/tree/develop"""
# Real-time Plotting Class for Python written by Ozhan Ozen. Last update was done 09.04.2018.
# Class to be called from the code. Uses pyqtgraph library from QT, which creates a QApplication. Check
# pyqtgraph.org for reference. The initialization parameters are plotting period (ms), a safety buffer (s)
# for time-plots in case there are delays in plotting and you do not want this to disturb the plot and, finally
# anti-aliasing feature. Do not use anti-aliasing feature unless you have a super-computer. The smaller the plotting
# period, the bigger the safety buffer you should set. Of course, there is a limit
# If plotting period is too small, the pc will not be able to plot it in time and you data will be distorted in the plot.
# You would get a warning in this case. After creating a plotting object, you should use Add_New_plot functions to
# add plots to the main object, then use Start_Plotting function to start the plotting. BTW, Star_Plotting function
# stops the thread at that function until the plotting window is closed, and this plotting element has to run in main
# thread. So put everything else is other threads. Make sure that you do not forget to update the data which will be plotted,
# manually. This can be done be modifying the outer-scope accessible variables ( T_1, X_1_1, Y_1_2 ) in real time.
# For example. object.DATA["T_1"] is the time variable of the first plot whereas object.DATA["Y_1_2"] is the Y
# axis variable of the second curve of the first plot, etc.
class RTplot_Python:

    def __init__(self, UpdatePeriod, SafeSeconds, AntiAliasing):

        self.__Application = QtGui.QApplication([])
        self.__Window = pg.GraphicsWindow()
        self.__Window.setWindowTitle('Real-time Data Plotter')
        pg.setConfigOptions(antialias=AntiAliasing)

        # For creating buffer which will be used in case the real buffer do not contain enough
        # elements to plot consistent time-based data.
        self.__UPDATE_PERIOD = UpdatePeriod
        self.__DATA_BUFFER_SIZE_SAFE_PART_IN_SEC = SafeSeconds
        self.__SPEED_WARNING_ALLOWANCE = self.__DATA_BUFFER_SIZE_SAFE_PART_IN_SEC * 0.95 # Gives warning if this limit is passed. Basicall reduce the plotting period if you get this.
        self.__DATA_BUFFER_SIZE_SAFE_PART = int(self.__DATA_BUFFER_SIZE_SAFE_PART_IN_SEC * 1000 / UpdatePeriod )

        #The dictionary initializations. Will be explained when they are used. Main logic is like this: there
        # may be multiple plots (each box inside the main big window), and each plot may have multiple curves inside.
        self.__nPLOTS = 0
        self.__PLOTS = {}
        self.__DATA_BUFFER = {}
        self.__DATA_BUFFER_SIZE = {}
        self.__DATA_HISTORY = {}
        self.__CURVES = {}
        self.__nCURVES_pPLOT = {}
        self.Data = {}
        self.__FIRST_UPDATE = {}
        self.__IS_TIME = {}
        self.__IS_REVERSE = {}
        self.__TIME_TO_RESET = {}

        # Debug Messages.
        self.__DEBUG_ = False

    # Function to add a new plot. A plot may consist multiple curves (both X and Y values vs to Time,
    # two different end-effectors, etc) and a plot may be either time-series or not. Even a non-time-series may
    # include a time history (if you wanna plot the last 1 second of end-effector like a shadow following
    # real-time value). So basically, nCurves is how many curves will be in that plot. History is what will
    # be the history size (s), put 0 if you dont want a history. Bigger history take more power to plots so
    # lets be energy efficient. isTimerReverseLegend is a 3x1 boolean vector to encode settings about if one of the the plot
    # axes will be Time or not, if you wanna plot reversed (like plotting when time increase in Y axis maybe?),
    # and if want to put a legend for curves. Range is 4x1 vector which sets the ranges of the fixed axes (you cannot
    # stop/fix time unless you are Monica Belluci.) Label is a 5x1 string vector for Plot Title, XLabel, Xunit, YLabel and Yunit
    # in order. CurveNames are respective names for curves if Legend is on. Grid if a 2x1 boolean vector for
    # setting grid for axes X and Y, respectively. InvertAxes is a 2x1 boolean for inverting either X or Y axes (making
    # X axis going from positive to negative in right direction for example). PenSymbol is either "none" for standard
    # line plotting, or "o","d", etc for plotting for specific symbols. The class automatically sets different colours
    # for different curves on the same plot (for "none") and certain different symbols for a low number of curves. (If
    # you want somehow more symbols check the website and add symbol names as "stars" yourself.)
    #

    def Add_New_Plot(self, nCurves, History, isTimeReverseLegend , Range, Label,CurveNames, Grid, InvertAxes, PenSymbol):

        self.__nPLOTS = self.__nPLOTS + 1 #Increase # of plots when a new one is added. __Plots["Plot_1"] refers to Xth plot when you wanna access if
        self.__IS_TIME["Plot_" + str(self.__nPLOTS)] = isTimeReverseLegend[0]
        self.__IS_REVERSE["Plot_" + str(self.__nPLOTS)] = isTimeReverseLegend[1]
        if ((isTimeReverseLegend[0]==False) and (isTimeReverseLegend[1])==True ):
            print("ATTENTION!!! REVERSE MODE...")
        if ((isTimeReverseLegend[0]==True) and (History<=0)):
            print("ATTENTION!!! THESE SETTINGS DO NOT MAKE SENSE. SELECT HISTORY BIGGER THAN 0 FOR TIME PLOT")
        self.__nCURVES_pPLOT["Plot_" + str(self.__nPLOTS)] = nCurves
        self.__DATA_HISTORY["Plot_" + str(self.__nPLOTS)] = History
        self.__FIRST_UPDATE["Plot_" + str(self.__nPLOTS)] = True # This is used to reset data if plotting is started at a later time.
        self.__DATA_BUFFER_SIZE["Plot_" + str(self.__nPLOTS)] = int(1000 * History / self.__UPDATE_PERIOD) + self.__DATA_BUFFER_SIZE_SAFE_PART
        self.__TIME_TO_RESET["Plot_" + str(self.__nPLOTS)] = self.__DATA_BUFFER_SIZE_SAFE_PART_IN_SEC + History

        self.__PLOTS["Plot_" + str(self.__nPLOTS)] = self.__Window.addPlot()
        self.__PLOTS["Plot_" + str(self.__nPLOTS)].hideButtons() # Hide auto-scale button (A).
        self.__PLOTS["Plot_" + str(self.__nPLOTS)].showGrid(x=bool(Grid[0]), y=bool(Grid[1]))
        self.__PLOTS["Plot_" + str(self.__nPLOTS)].setTitle(title=Label[0])


        # Invert values in axes, to have an easier understandability.
        self.__PLOTS["Plot_" + str(self.__nPLOTS)].invertX(b=InvertAxes[0])
        self.__PLOTS["Plot_" + str(self.__nPLOTS)].invertY(b=InvertAxes[1])
        # Disable the stupid automatic SI prefix scaling.
        AxisXItem = self.__PLOTS["Plot_" + str(self.__nPLOTS)].getAxis("bottom")
        AxisYItem = self.__PLOTS["Plot_" + str(self.__nPLOTS)].getAxis("left")
        AxisXItem.enableAutoSIPrefix(enable=False)
        AxisYItem.enableAutoSIPrefix(enable=False)

        if (isTimeReverseLegend[2] == True):
            self.__PLOTS["Plot_" + str(self.__nPLOTS)].addLegend()

        # Generates buffer sizes for data and sets labels/ranges according to the typpe of the plot set.
        if (isTimeReverseLegend[0] == True):
            self.Data["T_" + str(self.__nPLOTS)] = 0
            for i in range(0, nCurves):
                self.Data["Y_" + str(self.__nPLOTS) + "_" + str(i + 1)] = 0
            self.__DATA_BUFFER["T" + str(self.__nPLOTS)] = np.linspace(-self.__DATA_BUFFER_SIZE["Plot_" + str(self.__nPLOTS)]*self.__UPDATE_PERIOD/1000, 0, num=self.__DATA_BUFFER_SIZE["Plot_" + str(self.__nPLOTS)], endpoint=False) # In order to show like there was a past...
            self.__DATA_BUFFER["Y" + str(self.__nPLOTS)] = np.zeros((nCurves, self.__DATA_BUFFER_SIZE["Plot_" + str(self.__nPLOTS)])) # Initialize as zero size.
            if (isTimeReverseLegend[1] == True):
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setXRange(Range[0], Range[1], padding=0)
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setLabel('bottom', Label[1], units=Label[2])
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setLabel('left', "Time", units='s')
            else:
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setYRange(Range[2], Range[3], padding=0)
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setLabel('bottom', "Time", units='s')
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setLabel('left', Label[3], units=Label[4])
        else:
            self.Data["T_" + str(self.__nPLOTS)] = 0
            for i in range(0, nCurves):
                self.Data["X_" + str(self.__nPLOTS) + "_" + str(i + 1)] = 0
                self.Data["Y_" + str(self.__nPLOTS) + "_" + str(i + 1)] = 0
            self.__DATA_BUFFER["T" + str(self.__nPLOTS)] = np.linspace(-self.__DATA_BUFFER_SIZE["Plot_" + str(self.__nPLOTS)]*self.__UPDATE_PERIOD/1000, 0, num=self.__DATA_BUFFER_SIZE["Plot_" + str(self.__nPLOTS)], endpoint=False)
            self.__DATA_BUFFER["X" + str(self.__nPLOTS)] = np.zeros((nCurves, self.__DATA_BUFFER_SIZE["Plot_" + str(self.__nPLOTS)]))
            self.__DATA_BUFFER["Y" + str(self.__nPLOTS)] = np.zeros((nCurves, self.__DATA_BUFFER_SIZE["Plot_" + str(self.__nPLOTS)]))
            if (isTimeReverseLegend[1] == True):
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setXRange(Range[0], Range[1], padding=0)
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setYRange(Range[2], Range[3], padding=0)
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setLabel('left', Label[3], units=Label[4])
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setLabel('bottom', Label[1], units=Label[2])

            else:
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setXRange(Range[0], Range[1], padding=0)
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setYRange(Range[2], Range[3], padding=0)
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setLabel('left', Label[3], units=Label[4])
                self.__PLOTS["Plot_" + str(self.__nPLOTS)].setLabel('bottom', Label[1], units=Label[2])

        # Did some color/symbol differentiation wrt to # of curves used.
        if (PenSymbol == 'none' ):
            for i in range(0, nCurves):
                if (isTimeReverseLegend[0] == True):
                    self.__CURVES["Curve_" + str(self.__nPLOTS) + "_" + str(i+1)] = self.__PLOTS["Plot_" + str(self.__nPLOTS)].plot(self.__DATA_BUFFER["T" + str(self.__nPLOTS)][:],self.__DATA_BUFFER["Y" + str(self.__nPLOTS)][i, :],pen=(i,nCurves),name=CurveNames[i])
                else:
                    self.__CURVES["Curve_" + str(self.__nPLOTS) + "_" + str(i+1)] = self.__PLOTS["Plot_" + str(self.__nPLOTS)].plot(self.__DATA_BUFFER["X" + str(self.__nPLOTS)][i, :],self.__DATA_BUFFER["Y" + str(self.__nPLOTS)][i, :],pen=(i,nCurves),name=CurveNames[i])
        else:
            for i in range(0, nCurves):
                if (i == 1):
                    PenSymbolUsed = 't'
                elif (i == 2):
                    PenSymbolUsed = 's'
                elif (i == 3):
                    PenSymbolUsed = 'h'
                else:
                    PenSymbolUsed = PenSymbol

                if (isTimeReverseLegend[0] == True):
                    self.__CURVES["Curve_" + str(self.__nPLOTS) + "_" + str(i+1)] = self.__PLOTS["Plot_" + str(self.__nPLOTS)].plot(self.__DATA_BUFFER["T" + str(self.__nPLOTS)][:],self.__DATA_BUFFER["Y" + str(self.__nPLOTS)][i, :],pen=(i,nCurves),symbol= PenSymbolUsed,name=CurveNames[i])
                else:
                    self.__CURVES["Curve_" + str(self.__nPLOTS) + "_" + str(i+1)] = self.__PLOTS["Plot_" + str(self.__nPLOTS)].plot(self.__DATA_BUFFER["X" + str(self.__nPLOTS)][i, :],self.__DATA_BUFFER["Y" + str(self.__nPLOTS)][i, :],pen=(i,nCurves),symbol= PenSymbolUsed,name=CurveNames[i])

    # This function is called periodically if Start_Plotting is called outside. Does what name suggets.
    def __Plot_Update(self):

        # First we do everything for each plot.
        for i in range(1, self.__nPLOTS + 1):
            if (self.__DATA_HISTORY["Plot_" + str(i)] == 0): # if plot does not contain any history.
                if (self.__IS_TIME["Plot_" + str(i)] == True): # if is a time-plot (you would not use this probably, does not make sense.)
                    self.__DATA_BUFFER["T" + str(i)][-1] = self.Data["T_" + str(i)] # Update new time data.
                    for k in range(1, self.__nCURVES_pPLOT["Plot_" + str(i)] + 1): # for each curve
                        self.__DATA_BUFFER["Y" + str(i)][k - 1, -1] = self.Data["Y_" + str(i) + "_" + str(k)] # Update Y axis data
                        if (self.__IS_REVERSE["Plot_" + str(i)] == True): # Plot.
                            self.__CURVES["Curve_" + str(i) + "_" + str(k)].setData(self.__DATA_BUFFER["Y" + str(i)][k - 1, -1:],self.__DATA_BUFFER["T" + str(i)][-1:])
                        else:
                            self.__CURVES["Curve_" + str(i) + "_" + str(k)].setData(self.__DATA_BUFFER["T" + str(i)][-1:], self.__DATA_BUFFER["Y" + str(i)][k - 1, -1:])
                else: # If not a time-plot, like just an end-effector in real time.
                    for k in range(1, self.__nCURVES_pPLOT["Plot_" + str(i)] + 1): # for each curve.
                        self.__DATA_BUFFER["X" + str(i)][k - 1, -1] = self.Data["X_" + str(i) + "_" + str(k)] # Update X and Y axes data.
                        self.__DATA_BUFFER["Y" + str(i)][k - 1, -1] = self.Data["Y_" + str(i) + "_" + str(k)]
                        if (self.__IS_REVERSE["Plot_" + str(i)] == True): # Plot.
                            self.__CURVES["Curve_" + str(i) + "_" + str(k)].setData(self.__DATA_BUFFER["Y" + str(i)][k - 1, -1:],self.__DATA_BUFFER["X" + str(i)][k - 1, -1:])
                        else:
                            self.__CURVES["Curve_" + str(i) + "_" + str(k)].setData(self.__DATA_BUFFER["X" + str(i)][k - 1, -1:],self.__DATA_BUFFER["Y" + str(i)][k - 1, -1:])
            else: # If plot has a history.
                if (self.__FIRST_UPDATE["Plot_" + str(i)] == True): # If the plot is run the first time, we are gonna reset the data according to the first data received (maybe robot is already running for 50s, would not wanna start time from 0...).
                    if (self.__IS_TIME["Plot_" + str(i)] == True): # If time plot.
                        self.__DATA_BUFFER["T" + str(i)] = np.linspace(-self.__DATA_BUFFER_SIZE["Plot_" + str(i)] * self.__UPDATE_PERIOD / 1000 + self.Data["T_" + str(i)], self.Data["T_" + str(i) ],num=self.__DATA_BUFFER_SIZE["Plot_" + str(i)], endpoint=False) # Show like there was a past...
                        for t in range(1, self.__nCURVES_pPLOT["Plot_" + str(i)] + 1):
                            self.__DATA_BUFFER["Y" + str(i)][t - 1, :] = np.ones(self.__DATA_BUFFER_SIZE["Plot_" + str(i)]) * self.Data["Y_" + str(i) + "_" + str(t)] # Initialize to the first value acquired. Show like the past values were like these.
                    else: # If not time plot.
                        self.__DATA_BUFFER["T" + str(i)] = np.linspace(-self.__DATA_BUFFER_SIZE["Plot_" + str(i)] * self.__UPDATE_PERIOD / 1000 + self.Data["T_" + str(i)], self.Data["T_" + str(i) ],num=self.__DATA_BUFFER_SIZE["Plot_" + str(i)], endpoint=False) # Still process time since there may be history plotting.
                        for t in range(1, self.__nCURVES_pPLOT["Plot_" + str(i)] + 1):
                            self.__DATA_BUFFER["X" + str(i)][t - 1, :] = np.ones(self.__DATA_BUFFER_SIZE["Plot_" + str(i)]) * self.Data["X_" + str(i) + "_" + str(t)] # Initialize to the first value acquired. Show like the past values were like these
                            self.__DATA_BUFFER["Y" + str(i)][t - 1, :] = np.ones(self.__DATA_BUFFER_SIZE["Plot_" + str(i)]) * self.Data["Y_" + str(i) + "_" + str(t)]
                    self.__FIRST_UPDATE["Plot_" + str(i)] = False
                    if (self.__DEBUG_ == True):
                        print("Plot_" + str(i) + " data was RESET")
                elif ((self.__DATA_BUFFER["T" + str(i)][-1] - self.__DATA_BUFFER["T" + str(i)][0] > (self.__TIME_TO_RESET["Plot_" + str(i)] + self.__SPEED_WARNING_ALLOWANCE)) and (self.__FIRST_UPDATE["Plot_" + str(i)] == False)): # This gives warning in case you cannot plot fast enough as you command and you buffer is not enough to make this un-noticable. Check class comments.
                    if (self.__DEBUG_ == True):
                        print("WARNING: THE APP IS TOO SLOW TO CATCH UP DRAWING SPEED AIM, DELAY OUT OF BOUNDARIES: " + str( self.__DATA_BUFFER["T" + str(1)][-1] - self.__DATA_BUFFER["T" + str(1)][0] - (self.__DATA_BUFFER_SIZE_SAFE_PART_IN_SEC + self.__DATA_HISTORY["Plot_" + str(i)]) ) )

                if not ((self.__IS_TIME["Plot_" + str(i)] == True) and (self.Data["T_" + str(i)] == self.__DATA_BUFFER["T" + str(i)][-1])): # Unless you did not receive any new time data.
                    self.__DATA_BUFFER["T" + str(i)][:-1] = self.__DATA_BUFFER["T" + str(i)][1:] # Update and shift buffer, for common time value and independent Y values for each curve.
                    self.__DATA_BUFFER["T" + str(i)][-1] = self.Data["T_" + str(i)]
                    for k in range(1, self.__nCURVES_pPLOT["Plot_" + str(i)] + 1):
                        if (self.__IS_TIME["Plot_" + str(i)] == True):
                            self.__DATA_BUFFER["Y" + str(i)][k - 1, :-1] = self.__DATA_BUFFER["Y" + str(i)][k - 1, 1:]
                            self.__DATA_BUFFER["Y" + str(i)][k - 1, -1] = self.Data["Y_" + str(i) + "_" + str(k)]
                        else:
                            self.__DATA_BUFFER["Y" + str(i)][k - 1, :-1] = self.__DATA_BUFFER["Y" + str(i)][k - 1, 1:]
                            self.__DATA_BUFFER["X" + str(i)][k - 1, :-1] = self.__DATA_BUFFER["X" + str(i)][k - 1, 1:]
                            self.__DATA_BUFFER["Y" + str(i)][k - 1, -1] = self.Data["Y_" + str(i) + "_" + str(k)]
                            self.__DATA_BUFFER["X" + str(i)][k - 1, -1] = self.Data["X_" + str(i) + "_" + str(k)]

                # Some times due to delays, the a fixed size buffer do not contain same amount of time history and this
                # distorts the views. What I do is to look for a index (bigger then 0 hopefully, last value is the most
                # recent) whose value will be always having the same distance (in terms of time) to the most up-to-date
                # value. I plot the values starting from this index value to the most up-to-date value, so I plot only
                # the last TIME_HISTORY[whatever, but result is in seconds] corresponding values even if not the same
                # number of data. Looks like there is no delay this way. This somehow borrows data from
                # "__DATA_BUFFER_SIZE_SAFE_PART". However it is possible that delay is too huge that this is not enough.
                # then make the plot slower and get rich and have a better pc. Check class main comments.
                MIN_DATA_TO_DRAW_INDEX = 0
                while (self.__DATA_BUFFER["T" + str(i)][MIN_DATA_TO_DRAW_INDEX] < (self.__DATA_BUFFER["T" + str(i)][-1] - self.__DATA_HISTORY["Plot_" + str(i)])):
                    if (MIN_DATA_TO_DRAW_INDEX >= self.__DATA_BUFFER_SIZE["Plot_" + str(i)] - 1):
                        MIN_DATA_TO_DRAW_INDEX = self.__DATA_BUFFER_SIZE["Plot_" + str(i)] - 1
                        break
                    MIN_DATA_TO_DRAW_INDEX = MIN_DATA_TO_DRAW_INDEX + 1
                # Plots accordingly.
                for k in range(1, self.__nCURVES_pPLOT["Plot_" + str(i)] + 1):
                    if (self.__IS_REVERSE["Plot_" + str(i)] == True ):
                        if (self.__IS_TIME["Plot_" + str(i)] == True):
                            self.__CURVES["Curve_" + str(i) + "_" + str(k)].setData( self.__DATA_BUFFER["Y" + str(i)][k - 1, MIN_DATA_TO_DRAW_INDEX: ] , self.__DATA_BUFFER["T" + str(i)][MIN_DATA_TO_DRAW_INDEX: ] )
                        else:
                            self.__CURVES["Curve_" + str(i) + "_" + str(k)].setData( self.__DATA_BUFFER["Y" + str(i)][k - 1, MIN_DATA_TO_DRAW_INDEX: ] , self.__DATA_BUFFER["X" + str(i)][k - 1, MIN_DATA_TO_DRAW_INDEX: ] )
                    else:
                        if (self.__IS_TIME["Plot_" + str(i)] == True):
                            self.__CURVES["Curve_" + str(i) + "_" + str(k)].setData( self.__DATA_BUFFER["T" + str(i)][MIN_DATA_TO_DRAW_INDEX: ] , self.__DATA_BUFFER["Y" + str(i)][k - 1, MIN_DATA_TO_DRAW_INDEX: ] )
                        else:
                            self.__CURVES["Curve_" + str(i) + "_" + str(k)].setData( self.__DATA_BUFFER["X" + str(i)][k - 1, MIN_DATA_TO_DRAW_INDEX: ] , self.__DATA_BUFFER["Y" + str(i)][k - 1, MIN_DATA_TO_DRAW_INDEX: ] )

    # Starts plotting. Lock the thread (which have to be main thread.)
    def Start_Plotting(self):
        self.__TIMER = pg.QtCore.QTimer()
        self.__TIMER.timeout.connect(self.__Plot_Update)
        self.__TIMER.start(self.__UPDATE_PERIOD)
        self.__Application.instance().exec_()

    # Adds a new row for the next plots added.
    def New_Row(self):
        self.__Window.nextRow()

    # Turns ON\OFF Debug Messages.
    def Debug_Set(self, Boolean):
        if (Boolean == True):
            self.__DEBUG_ = True
        elif(Boolean == False):
            self.__DEBUG_ = False
        else:
            print("Invalid input to the UDP_Debug_Set function")

    # Quits the application, can be used with trigger.
    def Stop_Plotting(self):
        self.__Application.quit()