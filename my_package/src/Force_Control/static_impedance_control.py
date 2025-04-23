#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, RealTimePlot, Filters
from scipy.spatial.transform import Rotation

class StaticImpedanceControl(Robot):
    def __init__(self):
        self.shutdown_flag = False  

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_impedance()

        self.Ko = np.diag([0.1, 0.1, 0.1, 0.1, 0.2, 0.4])

        # Initial estimated frictional torque
        self.tau_f = np.zeros(self.n)

        self.impedance_force = np.zeros((self.n,1))

        # Torque filters
        self.filter = Filters()

        super().__init__()

    def start(self):
        # Set equilibrium point to current state
        self.position_des = self.Robot_RT_State.actual_tcp_position[:3].copy()   # (x, y, z) in mm
        self.orientation_des = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:].copy())  # Convert angles from Euler ZYZ (in degrees) to quaternion        
        self.q_dot_prev = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs.copy()   # convert from deg/s to rad/s
        rospy.loginfo("CartesianImpedanceController: Controller started")

    @property
    def current_velocity(self):
        # EE_dot = self.Robot_RT_State.actual_tcp_velocity   # (dx, dy, dz, da, db, dc)
        # X_dot = np.zeros(6)
        # X_dot[:3] = 0.001 * EE_dot[:3]   # convert from mm/s to m/s
        # X_dot[3:] = 0.0174532925 * EE_dot[3:]  # convert from deg/s to rad/s  
        # return X_dot

        self.q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs  # convert from deg/s to rad/s
        X_dot = self.J @ self.q_dot[:,np.newaxis]
        return X_dot.reshape(-1)

    @property
    def position_error(self):
        # actual TCP position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
        current_position = self.Robot_RT_State.actual_tcp_position[:3]     #  (x, y, z) in mm
        return 0.001 * (current_position - self.position_des)  # convert from mm to m
    
    @property
    def orientation_error(self):
        current_orientation = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # Convert angles from Euler ZYZ (in degrees) to quaternion        

        # checking for flipping of quaternion (Ensure quaternion is in the same hemisphere)
        if np.dot(current_orientation, self.orientation_des) < 0.0:
            current_orientation = -current_orientation
        
        current_rotation = Rotation.from_quat(current_orientation)  # default order: [x,y,z,w]
        desired_rotation = Rotation.from_quat(self.orientation_des)  # default order: [x,y,z,w]
        
        # Compute the "difference" or quaternion_distance (q_error = q_current^-1 * q_desired)
        """https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames"""
        error_rotation = current_rotation.inv() * desired_rotation
        error_quat = error_rotation.as_quat()  # default order: [x,y,z,w]
        
        # Extract x, y, z components of error quaternion
        rot_error = error_quat[:3][:, np.newaxis]
        
        # Transform orientation error to base frame
        current_rotation_matrix = current_rotation.as_matrix()  # Assuming this returns a 3x3 rotation matrix
        rot_error = current_rotation_matrix @ rot_error
        return rot_error.reshape(-1)

    # @property
    # def orientation_error(self):
    #     """
    #     rotation vector directly encodes rotation magnitude and axis, making it more intuitive
    #     It avoids the nonlinear scaling issues that can occur with quaternion components
    #     For control purposes, the rotation vector components often provide a more useful error 
    #     signal that's proportional to the rotation needed
    #     """
    #     current_orientation = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:])
    #     if np.dot(current_orientation, self.orientation_des) < 0.0:
    #         current_orientation = -current_orientation
        
    #     current_rotation = Rotation.from_quat(current_orientation)
    #     desired_rotation = Rotation.from_quat(self.orientation_des)
        
    #     # Compute the difference quaternion
    #     error_rotation = current_rotation.inv() * desired_rotation
        
    #     # Use rotation vector instead of quaternion components
    #     rot_error = error_rotation.as_rotvec()[:, np.newaxis]
        
    #     # Transform orientation error to base frame
    #     current_rotation_matrix = current_rotation.as_matrix()
    #     rot_error = - current_rotation_matrix @ rot_error
    #     return rot_error.reshape(-1)
    
    @property
    def q_ddot(self):
        Mq = self.Robot_RT_State.mass_matrix
        C = self.Robot_RT_State.coriolis_matrix
        G = self.Robot_RT_State.gravity_torque

        if abs(np.linalg.det(Mq)) >= 1e-3:
            Mq_inv = np.linalg.inv(Mq)
        else:
            Mq_inv = np.linalg.pinv(Mq)

        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert deg/s to rad/s
        q_ddot = Mq_inv @ (tau - C @ q_dot[:, np.newaxis] - G)
        return q_ddot.reshape(-1)
    
    @property
    def current_acceleration(self):
        q = 0.0174532925 * self.Robot_RT_State.actual_joint_position   # convert deg to rad
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert deg/s to rad/s
        J = self.kinematic_mode.Jacobian(q)
        J_dot = self.kinematic_model.Jacobian_dot(q, q_dot)

        X_ddot = J_dot @ q_dot[:, np.newaxis] + J @ self.q_ddot[:, np.newaxis]
        return X_ddot.reshape(-1)

    def set_compliance_parameters(self, translational_stiffness, rotational_stiffness):
        # Update stiffness matrix
        self.K_cartesian = np.eye(6)
        self.K_cartesian[:3, :3] *= translational_stiffness
        self.K_cartesian[3:, 3:] *= rotational_stiffness
        
        # Update damping matrix (critically damped)
        self.D_cartesian = np.eye(6)
        self.D_cartesian[:3, :3] *= 1.0 * np.sqrt(translational_stiffness) 
        self.D_cartesian[3:, 3:] *= 1.0 * np.sqrt(rotational_stiffness)   

    def saturate_torque(self, tau):
        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.9
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0]) # Nm

        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # tau = self.filter.low_pass_filter_torque(tau)

        # Clip torque values to stay within limits (both positive and negative)
        tau_saturated = np.clip(tau, -max_torque_limits, max_torque_limits)
        return tau_saturated

    def plot_data(self):
        try:
            self.plotter.update_imdepdance(self.data.actual_motor_torque, 
                                           self.data.raw_force_torque, 
                                           self.data.actual_joint_torque, 
                                           self.impedance_force.reshape(-1))
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")        

    def calc_friction_torque(self):
        motor_torque = self.Robot_RT_State.actual_motor_torque   # in Nm
        joint_torque = self.Robot_RT_State.actual_joint_torque   # in Nm
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert from deg/s to rad/s

        term_1 = np.dot(self.Ko, (motor_torque - joint_torque - self.tau_f)) * 0.005
        self.tau_f = self.tau_f + term_1

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
                # q = 0.0174532925 * self.Robot_RT_State.actual_joint_position   # convert deg to rad
                # J, _, _ = self.kinematic_model.Jacobian(q)

                # define EE-Position & Orientation error in task-space
                error = np.zeros(6)
                error[:3] = self.position_error
                error[3:] = self.orientation_error

                # EE-velocity in task-space
                current_velocity = self.current_velocity

                # Compute control
                self.impedance_force = self.K_cartesian @ error[:, np.newaxis] + self.D_cartesian @ current_velocity[:, np.newaxis]
                tau_task = - J.T @ self.impedance_force

                # compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque 

                # estimate frictional torque in Nm
                self.calc_friction_torque()

                # # Compute tool compensation
                # if self.activate_tool_compensation:
                #     tau_tool = J.T @ self.tool_compensation_force
                # else:
                #     tau_tool = np.zeros((6,1))
                    
                # Compute desired Joint torque
                tau_d = tau_task + G_torque[:, np.newaxis] + self.tau_f[:, np.newaxis] 

                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d)

                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()
                writedata.time = 0.0
                self.torque_publisher.publish(writedata)  # applying calculated joint torque as motor torque

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
        controller_thread = Thread(target=task.run_controller, args=(230.0, 150.0))  # translation stiff -> N/m, rotational stiffness -> Nm/rad 
        controller_thread.daemon = True
        controller_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.01)  # This keeps the plot window responsive

    except rospy.ROSInterruptException:
        pass

    finally:
        plt.close('all')  # Clean up plots on exit










