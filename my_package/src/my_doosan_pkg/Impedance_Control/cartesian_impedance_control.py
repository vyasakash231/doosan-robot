#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import *
from scipy.spatial.transform import Rotation

class CartesianImpedanceControl(Robot):
    def __init__(self):
        self.is_rt_connected = False
        self.shutdown_flag = False  
        self.Robot_RT_State = RT_STATE()

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_plots_1()

        # Set up target stiffness, damping, and filter parameters
        self.K_cartesian_target = np.eye(6)
        self.K_cartesian_target[:3, :3] *= 200.0  # Translational stiffness
        self.K_cartesian_target[3:, 3:] *= 10.0   # Rotational stiffness
        
        # Damping ratio = 1
        self.D_cartesian_target = np.eye(6)
        self.D_cartesian_target[:3, :3] *= 2.0 * np.sqrt(200.0)  # Translational damping
        self.D_cartesian_target[3:, 3:] *= 2.0 * np.sqrt(10.0)   # Rotational damping
        
        self.K_nullspace_target = 20.0
        self.filter_params = 0.3  # Parameter filter coefficient
        
        # Initialize desired position and orientation variables
        self.position_d_target = np.zeros(3)
        self.orientation_d_target = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion (x, y, z, w)
        
        # For nullspace control
        self.q_des_nullspace = np.zeros(6)
        
        # Current robot state variables
        self.tau_J_d = np.zeros(6)  # Previous desired torque
        
        # Maximum allowed torque rate change
        self.delta_tau_max = 1.0

        super().__init__()

    def start(self, position=[0.0, 0.0, 0.0]):
        """Initialize controller with current robot state"""
        # Set equilibrium point to current state
        self.position_des = self.Robot_RT_State.actual_flange_position[:3].copy()    # (x, y, z)
        _angles = self.Robot_RT_State.actual_flange_position[3:]    # (a, b, c) in degrees
        self.orientation_des = Rotation.from_euler('zyz', _angles, degrees=True).as_quat().copy()  # Convert angles from Euler ZYZ to quaternion Quaternion (x, y, z, w)
        
        self.position_des_target = self.Robot_RT_State.actual_flange_position[:3].copy() + position
        self.orientation_des_target = Rotation.from_euler('zyz', _angles, degrees=True).as_quat().copy()  # Convert angles from Euler ZYZ to quaternion Quaternion (x, y, z, w)
        
        # Set nullspace equilibrium configuration to initial joint positions
        self.q_des_nullspace = self.Robot_RT_State.actual_joint_position_abs.copy()
        
        rospy.loginfo("CartesianImpedanceController: Controller started")

    def set_compliance_parameters(self, translational_stiffness, rotational_stiffness, nullspace_stiffness):
        # Update stiffness matrix
        self.K_cartesian = np.eye(6)
        self.K_cartesian[:3, :3] *= translational_stiffness
        self.K_cartesian[3:, 3:] *= rotational_stiffness
        
        # Update damping matrix (critically damped)
        self.D_cartesian = np.eye(6)
        self.D_cartesian[:3, :3] *= 2.0 * np.sqrt(translational_stiffness)
        self.D_cartesian[3:, 3:] *= 2.0 * np.sqrt(rotational_stiffness)
        
        # Update nullspace stiffness
        self.K_nullspace = nullspace_stiffness

    def saturate_torque(self, tau, tau_J_d):
        """
        Limit both the torque rate of change and peak torque values
        for Doosan A0509 robot
        """
        # First limit rate of change as in your original function
        tau_rate_limited = np.zeros(self.n)
        for i in range(len(tau)):
            difference = tau[i] - tau_J_d[i]
            tau_rate_limited[i] = tau_J_d[i] + np.clip(difference, -self.delta_tau_max, self.delta_tau_max)
        
        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.9
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0]) # Nm

        # Clip torque values to stay within limits (both positive and negative)
        tau_saturated = np.clip(tau_rate_limited, -max_torque_limits, max_torque_limits)
        return tau_saturated

    def plot_data(self, data):
        try:
            # Instead of directly updating, queue the data for the main thread
            if hasattr(self, 'plot_data_queue'):
                self.plot_data_queue.append((data.actual_motor_torque, data.external_tcp_force, data.raw_force_torque))
            else:
                # Initialize the queue if it doesn't exist
                self.plot_data_queue = [(data.actual_motor_torque, data.external_tcp_force, data.raw_force_torque)]
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    def update_plots_from_queue(self):
        if hasattr(self, 'plot_data_queue') and self.plot_data_queue:
            # Process the oldest item in the queue
            data_tuple = self.plot_data_queue.pop(0)
            self.plotter.update_data_1(*data_tuple)

    def run_controller(self, position, K_trans, K_rot, K_null):
        # print(position)
        self.start(position)
        self.set_compliance_parameters(K_trans, K_rot, K_null)
        self.rate = rospy.Rate(1000)  # 1000 Hz control rate
        i = 0
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                G_torque = self.Robot_RT_State.gravity_torque
                C_matrix = self.Robot_RT_State.coriolis_matrix
                # M_matrix = self.Robot_RT_State.mass_matrix
                # F_ext = self.Robot_RT_State.external_tcp_force
                # F_ext = self.Robot_RT_State.raw_force_torque + [0.0, 0.0, 37.0, 0.0, 0.0, 0.0]

                q = self.Robot_RT_State.actual_joint_position_abs
                q_dot = self.Robot_RT_State.actual_joint_velocity_abs

                # actual robot flange position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
                self.current_position = self.Robot_RT_State.actual_flange_position[:3]     #  (x, y, z)
                _angles = self.Robot_RT_State.actual_flange_position[3:]    # (a, b, c) in degrees

                # Convert angles from Euler ZYZ to quaternion
                self.current_orientation = Rotation.from_euler('zyz', _angles, degrees=True).as_quat()  # Quaternion (x, y, z, w)

                # Find Jacobian matrix
                J = self.Robot_RT_State.jacobian_matrix

                # Position error
                error = np.zeros(6)
                error[:3] = self.current_position - self.position_des

                # Orientation error (using quaternions)
                # Ensure quaternion dot product is positive (shortest path)
                if np.dot(self.current_orientation, self.orientation_des) < 0.0:
                    current_orientation_adj = -self.current_orientation
                else:
                    current_orientation_adj = self.current_orientation.copy()

                # Convert quaternions to rotation matrices for error computation
                current_rotation = Rotation.from_quat([current_orientation_adj[0], current_orientation_adj[1], 
                                                    current_orientation_adj[2], current_orientation_adj[3]])
                desired_rotation = Rotation.from_quat([self.orientation_des[0], self.orientation_des[1], 
                                                    self.orientation_des[2], self.orientation_des[3]])
                
                # "difference" quaternion
                error_rotation = desired_rotation * current_rotation.inv()
                error_quat = error_rotation.as_quat()
                error[3:] = error_quat[:3]  # Extract (x, y, z) from quaternion
                
                # Transform orientation error to base frame
                current_rotation_matrix = current_rotation.as_matrix()
                error[3:] = -current_rotation_matrix @ error[3:]

                # Compute control
                # Cartesian PD control with damping ratio = 1
                tau_task = J.T @ (-self.K_cartesian @ error[:, np.newaxis] - self.D_cartesian @ (J @ q_dot[:, np.newaxis]))

                # Compute nullspace control
                jacobian_pinv = np.linalg.pinv(J.T)
                nullspace_proj = np.eye(self.n) - J.T @ jacobian_pinv
                tau_nullspace = nullspace_proj @ (self.K_nullspace * (self.q_des_nullspace - q)[:, np.newaxis] - 2.0 * np.sqrt(self.K_nullspace) * q_dot[:, np.newaxis])
                
                # Coriolis Torque -> C(q, q_dot) * q_dot
                coriolis_torque = C_matrix @ q_dot[:, np.newaxis]

                # Compute desired torque
                tau_d = tau_task + tau_nullspace + coriolis_torque + G_torque[:, np.newaxis]
                
                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d, self.tau_J_d)
                
                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()
                writedata.time = 0.5  # 100 ms
                self.torque_publisher.publish(writedata)

                self.tau_J_d = tau_d.copy()

                # Update parameters with filtering
                self.K_cartesian = (self.filter_params * self.K_cartesian_target + (1.0 - self.filter_params) * self.K_cartesian)
                self.D_cartesian = (self.filter_params * self.D_cartesian_target + (1.0 - self.filter_params) * self.D_cartesian)
                self.K_nullspace = (self.filter_params * self.K_nullspace_target + (1.0 - self.filter_params) * self.K_nullspace)

                # Update desired position and orientation with filtering
                self.position_des = (self.filter_params * self.position_des_target + (1.0 - self.filter_params) * self.position_des)
                self.orientation_des = quat_slerp(self.orientation_des, self.orientation_des_target, self.filter_params)   # Spherical linear interpolation for orientation
                
                # print(i, self.Robot_RT_State.actual_joint_position_abs)
                i += 1

                self.rate.sleep()
    
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()

if __name__ == "__main__":
    try:
        # Initialize ROS node first
        rospy.init_node('My_service_node')
        
        # Create control object
        task = CartesianImpedanceControl()
        rospy.sleep(2.5)  # Give time for initialization

        position = [50.0, 0.0, -100.0]

        # Start controller in a separate thread
        controller_thread = Thread(target=task.run_controller, args=(position, 100.0, 10.0, 30.0))  # Kc -> N/m, Dc -> Nm/rad, K_null ->  Joint space stiffness  
        controller_thread.daemon = True
        controller_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            task.update_plots_from_queue()
            plt.pause(0.01)  # This keeps the plot window responsive

        # x1, sol = get_current_posx()   #  x1 w.r.t. DR_BASE
        # Xd = np.array(x1)  # [mm, deg]

    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close('all')  # Clean up plots on exit










