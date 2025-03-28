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
        self.shutdown_flag = False  
        self.Robot_RT_State = RT_STATE()

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_plots_1()

        # Set up target stiffness, damping, and filter parameters
        self.K_cartesian_target = np.eye(6)
        self.K_cartesian_target[:3, :3] *= 100.0  # Translational stiffness
        self.K_cartesian_target[3:, 3:] *= 100.0   # Rotational stiffness
        
        # Damping ratio = 1
        self.D_cartesian_target = np.eye(6)
        self.D_cartesian_target[:3, :3] *= 2.0 * np.sqrt(100.0)  # Translational damping
        self.D_cartesian_target[3:, 3:] *= 2.0 * np.sqrt(100.0)   # Rotational damping
        
        self.filter_params = 0.4  # Parameter filter coefficient

        self.J_m = np.diag([0.0004956, 0.0004956, 0.0001839, 0.00009901, 0.00009901, 0.00009901])
        self.K_o = np.diag([0.1, 0.1, 0.2, 0.15, 0.25, 0.4])
        
        # Initialize desired position and orientation variables
        self.position_d_target = np.zeros(3)
        self.orientation_d_target = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion (x, y, z, w)
        
        # Current robot state variables
        self.tau_J_d = np.zeros(6)  # Previous desired torque
        
        # Maximum allowed torque rate change
        self.delta_tau_max = 2.0

        # Initial estimated frictional torque
        self.tau_f = np.zeros(self.n)

        super().__init__()

    def start(self):
        """Initialize controller with current robot state"""
        # Set equilibrium point to current state
        self.position_des = 0.001 * self.Robot_RT_State.actual_flange_position[:3].copy()    # (x, y, z) convert from mm to m
        # _angles = self.Robot_RT_State.actual_flange_position[3:]    # (a, b, c) in degrees
        # self.orientation_des = Rotation.from_euler('zyz', _angles, degrees=True).as_quat().copy()  # Convert angles from Euler ZYZ to quaternion Quaternion (x, y, z, w)
        
        self.position_des_target = 0.001 * np.array([200.0, 0.0, 500.0])   # (x, y, z) convert from mm to m
        # self.orientation_des_target = Rotation.from_euler('zyz', _angles, degrees=True).as_quat().copy()   # Convert angles from Euler ZYZ to quaternion Quaternion (x, y, z, w)
        
        self.q_dot_prev = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity.copy()   # Previous joint velocity (convert from deg/s to rad/s)

        # Set nullspace equilibrium configuration to initial joint positions
        self.q_d_nullspace = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity.copy() 
        rospy.loginfo("CartesianImpedanceController: Controller started")

    def set_compliance_parameters(self, translational_stiffness, rotational_stiffness):
        # Update stiffness matrix
        self.K_cartesian = np.eye(6)
        self.K_cartesian[:3, :3] *= translational_stiffness
        self.K_cartesian[3:, 3:] *= rotational_stiffness
        
        # Update damping matrix (critically damped)
        self.D_cartesian = np.eye(6)
        self.D_cartesian[:3, :3] *= 2.0 * np.sqrt(translational_stiffness)
        self.D_cartesian[3:, 3:] *= 2.0 * np.sqrt(rotational_stiffness)

        self.K_nullspace = 10.0

    def saturate_torque(self, tau, tau_J_d):
        """
        Limit both the torque rate of change and peak torque values for Doosan A0509 robot
        """
        # First limit rate of change as in your original function
        # tau_rate_limited = np.zeros(self.n)
        # for i in range(len(tau)):
        #     difference = tau[i] - tau_J_d[i]
        #     tau_rate_limited[i] = tau_J_d[i] + np.clip(difference, -self.delta_tau_max, self.delta_tau_max)
        # tau = tau_rate_limited.copy()

        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.9
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0])  # Nm

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

    def Mx(self, Mq, J):
        Mx_inv = J @ np.linalg.inv(Mq) @ J.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-3:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv)
        return Mx

    def calc_friction_torque(self):
        motor_torque = self.Robot_RT_State.actual_motor_torque
        joint_torque = self.Robot_RT_State.actual_joint_torque
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity   # convert from deg/s to rad/s

        term_1 = np.dot(self.K_o, (motor_torque - joint_torque - self.tau_f)) * 0.001
        term_2 = np.dot(self.K_o, np.dot(self.J_m, (self.q_dot_prev - q_dot)))
        self.tau_f = self.tau_f + term_1 + term_2
        self.q_dot_prev = q_dot.copy()

    def run_controller(self, K_trans, K_rot):
        self.start()
        self.set_compliance_parameters(K_trans, K_rot)
        self.current_velocity = np.zeros(6)
        tau_task = np.zeros((6,1))

        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate

        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                # actual robot flange position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
                self.current_position = 0.001 * self.Robot_RT_State.actual_flange_position[:3]     #  (x, y, z) convert from mm to m
                current_velocity = self.Robot_RT_State.actual_flange_velocity     #  (dx, dy, dz, da, db, dc)

                # Convert angles from Euler ZYZ to quaternion
                # _angles = self.Robot_RT_State.actual_flange_position[3:]    # (a, b, c) in degrees
                # self.current_orientation = Rotation.from_euler('zyz', _angles, degrees=True).as_quat()  # Quaternion (x, y, z, w)

                self.current_velocity[:3] = 0.001 * current_velocity[:3]   # convert from mm/s to m/s
                self.current_velocity[3:] = 0.0174532925 * current_velocity[3:]  # convert from deg/s to rad/s

                # Find Jacobian matrix
                J = self.Robot_RT_State.jacobian_matrix
                Mq = self.Robot_RT_State.mass_matrix

                # Position error
                error = np.zeros(6)
                error[:3] = self.current_position - self.position_des

                print(1000*self.current_position, 1000*self.position_des_target)

                q = 0.0174532925 * self.Robot_RT_State.actual_joint_position   # convert from deg to rad
                q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity   # convert from deg/s to rad/s

                # # dynamically consistent generalized inverse
                Mx = self.Mx(Mq, J)
                J_pinv_T = Mx @ J @ np.linalg.inv(Mx)
                nullspace_proj = np.eye(6) - J.T @ J_pinv_T
                tau_nullspace = nullspace_proj @ (self.K_nullspace * (self.q_d_nullspace - q)[:, np.newaxis]) #- 1.0 * np.sqrt(self.K_nullspace) * q_dot[:, np.newaxis])

                # Compute control
                # Cartesian PD control with damping
                # tau_task = - J.T @ (self.K_cartesian @ error[:, np.newaxis] + self.D_cartesian @ self.current_velocity[:, np.newaxis])
                tau_task = - J.T @ (self.K_cartesian @ error[:, np.newaxis] + self.D_cartesian @ (J @ q_dot[:, np.newaxis]))
                        
                # compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque 

                # compute corolise torque in Nm
                # C = self.Robot_RT_State.coriolis_matrix
                # C_torque = C @ q_dot[:, np.newaxis]

                # estimate frictional torque in Nm
                self.calc_friction_torque()

                # Compute desired torque
                tau_d = tau_task + G_torque[:, np.newaxis] + tau_nullspace
                
                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d, self.tau_J_d)
                
                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()
                writedata.time = 0.0  
                self.torque_publisher.publish(writedata)

                self.tau_J_d = tau_d.copy()

                # Update parameters with filtering
                self.K_cartesian = (self.filter_params * self.K_cartesian_target + (1.0 - self.filter_params) * self.K_cartesian)
                self.D_cartesian = (self.filter_params * self.D_cartesian_target + (1.0 - self.filter_params) * self.D_cartesian)

                # Update desired position and orientation with filtering
                self.position_des = (self.filter_params * self.position_des_target + (1.0 - self.filter_params) * self.position_des)
                # self.orientation_des = quat_slerp(self.orientation_des, self.orientation_des_target, self.filter_params)   # Spherical linear interpolation for orientation
                
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
        task = CartesianImpedanceControl()
        rospy.sleep(2.0)  # Give time for initialization

        # Start controller in a separate thread
        controller_thread = Thread(target=task.run_controller, args=(5.0, 5.0))  # translation stiff -> N/m, rotational stiffness -> Nm/rad 
        controller_thread.daemon = True
        controller_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.01)  # This keeps the plot window responsive

    except rospy.ROSInterruptException:
        pass

    finally:
        plt.close('all')  # Clean up plots on exit










