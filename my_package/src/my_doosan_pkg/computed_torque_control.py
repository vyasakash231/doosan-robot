#!/usr/bin/env python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp"))) # get import path : DSR_ROBOT.py 
from basic_import import *
from .common_utils.doosanA0509s import Robot
from .common_utils.filters import Filters
from .common_utils.plot import RealTimePlot
from .common_utils.robot_RT_state import RT_STATE
# from common_for_JLA import *

class CTC(Robot):
    def __init__(self, dt):
        self.is_rt_connected = False
        self.shutdown_flag = False  # Add flag to track shutdown state
        self.Robot_RT_State = RT_STATE()

        self.dt = dt

        self.filter = Filters(dt)

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_plots_2()

        super().__init__()

        self.joint_vel_limits([150, 150, 150, 150, 150, 150])  # Increased from 50 deg/s
        self.joint_acc_limits([100, 100, 100, 100, 100, 100])  # Increased from 25 deg/s^2

    def plot_data(self):
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

    def run_controller(self, Kp, Kd, qd, qd_dot, qd_ddot):
        rate = rospy.Rate(1000)
        i = 0
        
        total_points = qd.shape[1]
        
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag and i < total_points:            
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
        task = CTC(dt)
        rospy.sleep(2.5)  # Give time for initialization

        Kp = np.diag([2.5, 2.5, 3.0, 3.5, 30.0, 300.0]) 
        Kd = np.diag([0.5, 0.5, 0.5, 0.5, 2.0, 20.0])
        
        # Start impedance control in a separate thread
        control_thread = Thread(target=lambda: task.run_controller(Kp, Kd, qd, qd_dot, qd_ddot))
        control_thread.daemon = True
        control_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.05)  # This keeps the plot window responsive
            
    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close('all')  # Clean up plots on exit