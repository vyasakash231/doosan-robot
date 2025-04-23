#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp


class CartesianImpedanceControl(Robot):
    def __init__(self, file_name='demo'):
        self.shutdown_flag = False  
        self.file_name = file_name
        
        self.Ko = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        # Current robot state variables
        self.tau_J_d = np.zeros(6)  # Previous desired torque
        self.delta_tau_max = 1.0
        
        # Initial estimated frictional torque
        self.tau_f = np.zeros(self.n)

        self.impedance_force = np.zeros((self.n,1))

        super().__init__()

    def load_demo(self, name='demo'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.q_demo = data['q']    # shape: (N, 6), in rad
        self.position_demo = 1000 * data['traj']   # shape: (N, 3), convert from m to mm
        self.orientation_demo = data['ori']   # shape: (N, 4)
        self.linear_velocity_demo = data['vel']   # shape: (N, 3), in m/s
        self.angular_velocity_demo = data['omega']   # shape: (N, 3), in rad/s
        # self.gripper = data['grip']
        self.N = self.position_demo.shape[0]   # no of sample points

        # Apply filtering for smoother trajectories
        self.filter_trajectory()

    def filter_trajectory(self):
        """
        Apply filtering to position, orientation and velocity data to ensure smooth trajectory
        """
        # Filter parameters
        fs = 25.0  # Sample frequency (Hz)
        cutoff = 5.0  # Cutoff frequency (Hz) - adjust as needed
        order = 2  # Filter order
        
        # Create the filter
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply filter to position data (each axis separately)
        filtered_position = np.zeros_like(self.position_demo)
        for i in range(3):
            filtered_position[:, i] = filtfilt(b, a, self.position_demo[:, i])
        
        # Apply filter to linear velocity data
        filtered_linear_vel = np.zeros_like(self.linear_velocity_demo)
        for i in range(3):
            filtered_linear_vel[:, i] = filtfilt(b, a, self.linear_velocity_demo[:, i])
        
        # Apply filter to angular velocity data
        filtered_angular_vel = np.zeros_like(self.angular_velocity_demo)
        for i in range(3):
            filtered_angular_vel[:, i] = filtfilt(b, a, self.angular_velocity_demo[:, i])
        
        # Handle orientation (quaternions) - special care needed
        # Simple filtering can break quaternion properties
        filtered_orientation = np.zeros_like(self.orientation_demo)
        filtered_orientation[0] = self.orientation_demo[0]  # Keep first quaternion as is
        
        # Create rotation objects and times
        original_rotations = Rotation.from_quat(self.orientation_demo)
        times = np.arange(len(self.orientation_demo))
        
        # Create Slerp object
        slerp = Slerp(times, original_rotations)
        
        # Evaluate at the same times, but the interpolation has a smoothing effect
        smoothed_rotations = slerp(times)
        filtered_orientation = smoothed_rotations.as_quat()
        
        # Update trajectory data with filtered versions
        self.position_demo = filtered_position
        self.orientation_demo = filtered_orientation
        self.linear_velocity_demo = filtered_linear_vel
        self.angular_velocity_demo = filtered_angular_vel
        print("Trajectory filtered for smoother motion")

    def store_data(self):
        pos = 0.001 * self.Robot_RT_State.actual_tcp_position[:3]    # in m
        orient = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # quaternions
        self.record_trajectory = np.vstack((self.record_trajectory, pos))  # shape: (N, 3) 
        self.record_orientation = np.vstack((self.record_orientation, orient))  # shape: (N, 4)

        self.record_motor_torque = np.vstack((self.record_motor_torque, self.Robot_RT_State.actual_motor_torque))  # shape: (N, 6) 
        self.record_joint_torque = np.vstack((self.record_joint_torque, self.Robot_RT_State.actual_joint_torque))  # shape: (N, 6)
        self.record_imp_force = np.vstack((self.record_imp_force, self.impedance_force.reshape(-1)))  # shape: (N, 6)

    def start(self):
        """Initialize controller with current robot state"""
        # Load data
        self.load_demo(name=self.file_name)

        # define desired pose => [x, y, z] & [q1, q2, q3, q0]
        self.position_des_next = self.Robot_RT_State.actual_tcp_position[:3].copy()    # (x, y, z) in mm
        self.orientation_des_next = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:].copy())    # Convert angles from Euler ZYZ (in degrees) to quaternion        

        # for plotting
        self.record_trajectory = 0.001 * self.Robot_RT_State.actual_tcp_position[:3]   # in m
        self.record_orientation = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # quaternions

        self.record_motor_torque = self.Robot_RT_State.actual_motor_torque  # in Nm
        self.record_joint_torque = self.Robot_RT_State.actual_joint_torque  # in Nm
        self.record_imp_force = np.zeros(self.n)
        rospy.loginfo("CartesianImpedanceController: Controller started")

    @property
    def current_velocity(self):
        self.q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert from deg/s to rad/s
        X_dot = self.J @ self.q_dot[:,np.newaxis]
        return X_dot.reshape(-1)   # in [m/s, rad/s]
    
    @property
    def velocit_error(self):
        # Combine into a single 6D velocity vector
        desired_velocity = np.zeros(6)
        desired_velocity[:3] = self.desired_linear_vel
        desired_velocity[3:] = self.desired_angular_vel

        # EE-velocity in task-space
        current_velocity = self.current_velocity

        # Calculate velocity error (current - desired)
        return current_velocity - desired_velocity

    @property
    def position_error(self):
        # actual robot flange position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
        current_position = self.Robot_RT_State.actual_tcp_position[:3]   # (x, y, z) in mm
        return 0.001 * (current_position - self.position_des_next)   # convert from mm to m
    
    @property
    def orientation_error(self):
        current_orientation = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # Convert angles from Euler ZYZ (in degrees) to quaternion        

        if np.dot(current_orientation, self.orientation_des_next) < 0.0:
            current_orientation = -current_orientation
        
        current_rotation = Rotation.from_quat(current_orientation)  # default order: [x,y,z,w]
        desired_rotation = Rotation.from_quat(self.orientation_des_next)  # default order: [x,y,z,w]
        
        # Compute the "difference" quaternion (q_error = q_current^-1 * q_desired)
        """https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames"""
        error_rotation = current_rotation.inv() * desired_rotation
        error_quat = error_rotation.as_quat()  # default order: [x,y,z,w]

        # Extract the angle between quaternions
        self.error_angle = 2 * np.arccos(np.clip(abs(error_quat[3]), -1.0, 1.0))  # Using w component
        
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
        self.D_cartesian[3:, 3:] *= 0.25 * np.sqrt(rotational_stiffness)

    def saturate_torque(self, tau):
        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.95
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0])  # Nm

        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # Clip torque values to stay within limits (both positive and negative)
        tau = np.clip(tau, -max_torque_limits, max_torque_limits)
        return tau

    def plot_data(self):
        pass

    def calc_friction_torque(self):
        motor_torque = self.Robot_RT_State.actual_motor_torque   # in Nm
        joint_torque = self.Robot_RT_State.actual_joint_torque   # in Nm

        term_1 = np.dot(self.Ko, (motor_torque - joint_torque - self.tau_f)) * 0.005
        self.tau_f += term_1

    def run_controller(self, K_trans, K_rot):
        self.start()
        self.set_compliance_parameters(K_trans, K_rot)
        tau_task = np.zeros((6,1))
        error = np.zeros(6)

        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate
        
        # Calculate time step between trajectory points (in seconds)
        traj_dt = 1.0/25.0  # 10Hz = 0.1s between points

        # Track start time for trajectory indexing
        start_time = rospy.Time.now().to_sec()

        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                # Calculate elapsed time and determine trajectory index
                current_time = rospy.Time.now().to_sec()
                elapsed_time = current_time - start_time
                current_idx = int(elapsed_time / traj_dt)
                
                # Ensure index is within bounds
                if current_idx >= self.N:
                    current_idx = self.N - 1
                    if current_idx == self.N - 1 and np.linalg.norm(self.position_error)*1000 < 10.0:  # if error is less then 10mm break
                        rospy.loginfo("Trajectory complete and position error < 10mm")
                        break

                # Update desired position and orientation
                self.position_des_next = self.position_demo[current_idx,:]  # (x, y, z) in mm
                self.orientation_des_next = self.orientation_demo[current_idx,:]

                # Get desired velocity from recorded data
                self.desired_linear_vel = self.linear_velocity_demo[current_idx,:]  # Already in m/s
                self.desired_angular_vel = self.angular_velocity_demo[current_idx,:]  # Already in rad/s

                # Find Jacobian matrix
                self.J = self.Robot_RT_State.jacobian_matrix

                # define EE-Position & Orientation error in task-space
                error[:3] = self.position_error
                error[3:] = self.orientation_error

                # define EE-Velocitt error in task-space
                velocity_error = self.velocit_error

                # Cartesian PD control with damping
                self.impedance_force = self.K_cartesian @ error[:, np.newaxis] + self.D_cartesian @ velocity_error[:, np.newaxis]
                tau_task = - self.J.T @ self.impedance_force
                        
                # compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque 

                # estimate frictional torque in Nm
                self.calc_friction_torque()

                # Compute desired torque
                tau_d = tau_task + G_torque[:, np.newaxis] + self.tau_f[:, np.newaxis] 
                
                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d)
                
                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()   # target motor torque [Nm]
                writedata.time = 0.0    # target time [sec]
                self.torque_publisher.publish(writedata)

                # store actual data for plotting
                self.store_data()

                rate.sleep()
                
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()

    def save(self, name='task_performed'):
        curr_dir=os.getcwd()
        np.savez(curr_dir + '/data/' + str(name) + '.npz',
                traj=self.record_trajectory,
                ori=self.record_orientation,
                motor_torque=self.record_motor_torque,
                joint_torque=self.record_joint_torque,
                imp_force=self.record_imp_force
                #  grip=self.recorded_gripper
                )
