#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, RealTimePlot
from scipy.spatial.transform import Rotation

class StaticImpedanceControl(Robot):
    def __init__(self):
        self.shutdown_flag = False  

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_plots_1()

        # Set up target stiffness, damping, and filter parameters
        self.K_cartesian_target = np.eye(6)
        self.K_cartesian_target[:3, :3] *= 300.0  # Translational stiffness
        self.K_cartesian_target[3:, 3:] *= 150.0   # Rotational stiffness
        
        # Damping ratio = 1
        self.D_cartesian_target = np.eye(6)
        self.D_cartesian_target[:3, :3] *= 2.0 * np.sqrt(300.0)  # Translational damping
        self.D_cartesian_target[3:, 3:] *= 2.0 * np.sqrt(150.0)   # Rotational damping

        self.filter_params = 0.3  # Parameter filter coefficient

        self.J_m = np.diag([0.0004956, 0.0004956, 0.0001839, 0.00009901, 0.00009901, 0.00009901])
        self.K_o = np.diag([0.1, 0.1, 0.2, 0.15, 0.25, 0.5])
        
        # Current robot state variables
        self.tau_J_d = np.zeros(6)  # Previous desired torque
        
        # Maximum allowed torque rate change
        self.delta_tau_max = 2.0

        # Initial estimated frictional torque
        self.tau_f = np.zeros(self.n)

        super().__init__()

    def start(self):
        # Set equilibrium point to current state
        self.position_des = self.Robot_RT_State.actual_flange_position[:3].copy()    # (x, y, z) in mm
        self.orientation_des = self.eul2quat(self.Robot_RT_State.actual_flange_position[3:].copy())  # Convert angles from Euler ZYZ (in degrees) to quaternion        
        self.q_dot_prev = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity.copy()   # convert from deg/s to rad/s
        rospy.loginfo("CartesianImpedanceController: Controller started")

    @property
    def current_velocity(self):
        EE_dot = self.Robot_RT_State.actual_flange_velocity   # (dx, dy, dz, da, db, dc)
        X_dot = np.zeros(6)
        X_dot[:3] = 0.001 * EE_dot[:3]   # convert from mm/s to m/s
        X_dot[3:] = 0.0174532925 * EE_dot[3:]  # convert from deg/s to rad/s  
        return X_dot

    @property
    def position_error(self):
        # actual robot flange position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
        current_position = self.Robot_RT_State.actual_flange_position[:3]     #  (x, y, z) in mm
        return 0.001 * (current_position - self.position_des)  # convert from mm to m
    
    @property
    def orientation_error(self):
        current_orientation = self.eul2quat(self.Robot_RT_State.actual_flange_position[3:])   # Convert angles from Euler ZYZ (in degrees) to quaternion        

        if np.dot(current_orientation, self.orientation_des) < 0.0:
            current_orientation = -current_orientation
        
        current_rotation = Rotation.from_quat(current_orientation)  # default order: [x,y,z,w]
        desired_rotation = Rotation.from_quat(self.orientation_des)  # default order: [x,y,z,w]
        
        # Compute the "difference" quaternion (q_error = q_current^-1 * q_desired)
        """https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames"""
        error_rotation = current_rotation.inv() * desired_rotation
        error_quat = error_rotation.as_quat()  # default order: [x,y,z,w]
        
        # Extract x, y, z components of error quaternion
        rot_error = error_quat[:3][:, np.newaxis]
        
        # Transform orientation error to base frame
        current_rotation_matrix = current_rotation.as_matrix()  # Assuming this returns a 3x3 rotation matrix
        rot_error = current_rotation_matrix @ rot_error
        return rot_error.reshape(-1)

    def set_compliance_parameters(self, translational_stiffness, rotational_stiffness):
        # Update stiffness matrix
        self.K_cartesian = np.eye(6)
        self.K_cartesian[:3, :3] *= translational_stiffness
        self.K_cartesian[3:, 3:] *= rotational_stiffness
        
        # Update damping matrix (critically damped)
        self.D_cartesian = np.eye(6)
        self.D_cartesian[:3, :3] *= 2.0 * np.sqrt(translational_stiffness) 
        self.D_cartesian[3:, 3:] *= 2.0 * np.sqrt(rotational_stiffness)   

    def saturate_torque(self, tau, tau_J_d):
        """
        Limit both the torque rate of change and peak torque values for Doosan A0509 robot
        """
        # # First limit rate of change as in your original function
        # tau_rate_limited = np.zeros(self.n)
        # for i in range(len(tau)):
        #     difference = tau[i] - tau_J_d[i]
        #     tau_rate_limited[i] = tau_J_d[i] + np.clip(difference, -self.delta_tau_max, self.delta_tau_max)
        # tau = tau_rate_limited.copy()
        
        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.9
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0]) # Nm

        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # Clip torque values to stay within limits (both positive and negative)
        tau_saturated = np.clip(tau, -max_torque_limits, max_torque_limits)
        return tau_saturated

    def plot_data(self, data):
        try:
            self.plotter.update_data_1(data.actual_motor_torque, data.raw_force_torque, data.actual_joint_torque, data.raw_joint_torque)
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    def calc_friction_torque(self):
        motor_torque = self.Robot_RT_State.actual_motor_torque   # in Nm
        joint_torque = self.Robot_RT_State.actual_joint_torque   # in Nm
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity   # convert from deg/s to rad/s

        term_1 = np.dot(self.K_o, (motor_torque - joint_torque - self.tau_f)) * 0.001
        term_2 = np.dot(self.K_o, np.dot(self.J_m, (self.q_dot_prev - q_dot)))
        self.tau_f = self.tau_f + term_1 + term_2

        self.q_dot_prev = q_dot.copy()

    def run_controller(self, K_trans, K_rot):
        self.start()
        self.set_compliance_parameters(K_trans, K_rot)
        tau_task = np.zeros((6,1))

        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate
        
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                # Find Jacobian matrix
                J = self.Robot_RT_State.jacobian_matrix

                # define EE-Position & Orientation error in task-space
                error = np.zeros(6)
                error[:3] = self.position_error
                error[3:] = self.orientation_error

                # EE-velocity in task-space
                current_velocity = self.current_velocity

                # Compute control
                tau_task = - J.T @ (self.K_cartesian @ error[:, np.newaxis] + self.D_cartesian @ current_velocity[:, np.newaxis])  

                # compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque 

                # estimate frictional torque in Nm
                self.calc_friction_torque()
            
                # # Compute desired torque
                # if np.linalg.norm(error[:3]) > 0.001:
                #     tau_d = tau_task + G_torque[:, np.newaxis] + self.tau_f[:, np.newaxis]
                # else:
                #     tau_d = G_torque[:, np.newaxis] + self.tau_f[:, np.newaxis]

                tau_d = tau_task + G_torque[:, np.newaxis] + self.tau_f[:, np.newaxis]

                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d, self.tau_J_d)

                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()
                writedata.time = 0.0
                self.torque_publisher.publish(writedata)

                self.tau_J_d = tau_d.copy()

                # # Calculate error magnitude (for position components only)
                # error_magnitude = np.linalg.norm(error[:3])

                # # You can tune these parameters based on your system
                # base_filter = self.filter_params
                # max_filter = 0.9  # Maximum filter value
                # error_threshold = 0.01  # Error threshold in meters
                # scale = min(1.0, error_magnitude / error_threshold)  # Normalized error, capped at 1.0

                # # Adjust filter parameter based on error
                # adaptive_filter = base_filter + (max_filter - base_filter) * scale

                # # # Update parameters with adaptive filtering
                # self.K_cartesian = (adaptive_filter * self.K_cartesian_target + (1.0 - adaptive_filter) * self.K_cartesian)
                # self.D_cartesian = (adaptive_filter * self.D_cartesian_target + (1.0 - adaptive_filter) * self.D_cartesian)

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
        task = StaticImpedanceControl()
        rospy.sleep(2.0)  # Give time for initialization

        # Start controller in a separate thread
        controller_thread = Thread(target=task.run_controller, args=(200.0, 50.0))  # translation stiff -> N/m, rotational stiffness -> Nm/rad 
        controller_thread.daemon = True
        controller_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.01)  # This keeps the plot window responsive

    except rospy.ROSInterruptException:
        pass

    finally:
        plt.close('all')  # Clean up plots on exit










