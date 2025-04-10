#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, RealTimePlot
from scipy.spatial.transform import Rotation

class OSC(Robot):
    def __init__(self):
        self.shutdown_flag = False  

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_plots_1()

        self.Kp = np.diag([200.0, 200.0, 200.0, 100.0, 100.0, 100.0])
        # self.Kv = np.diag([50.0, 50.0, 50.0, 25.0, 25.0, 25.0])
        self.Kv = 2 * np.sqrt(self.Kp)

        self.J_m = np.diag([0.0004956, 0.0004956, 0.0001839, 0.00009901, 0.00009901, 0.00009901])
        self.K_o = np.diag([0.1, 0.1, 0.2, 0.15, 0.25, 0.5])
        
        # Initial estimated frictional torque
        self.tau_f = np.zeros(self.n)

        super().__init__()

        time.sleep(0.5)

        # Set Global Joint Velocity Limits
        self.max_vel = [50, 20]
        request = SetVelXRTRequest()
        request.trans = self.max_vel[0]  #  in mm/s
        request.rotation = self.max_vel[1]  # in deg/s
        self.ee_vel_limits(request)

        # Set Global Task Velocity Limits
        request = SetVelJRTRequest()
        request.vel = [60, 60, 60, 60, 60, 60]  # in deg/s
        self.joint_vel_limits(request)

    def plot_data(self):
        try:
            self.plotter.update_data_1(self.data.actual_motor_torque, self.data.raw_force_torque, self.data.actual_joint_torque, self.data.raw_joint_torque)
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    def start(self, position, velocity):
        # Set equilibrium point to current state
        self.position_des = position   # (x, y, z) in mm
        self.orientation_des = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:].copy())  # Convert angles from Euler ZYZ (in degrees) to quaternion  
        self.velocity_des = velocity  # [Vx, Vy, Vz, ωx, ωy, ωz] in [m/s, rad/s]
        self.q_dot_prev = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity.copy()   # convert from deg/s to rad/s

    @property
    def velocity_current(self):
        EE_dot = self.Robot_RT_State.actual_tcp_velocity   # [Vx, Vy, Vz, ωx, ωy, ωz]
        X_dot = np.zeros(6)
        X_dot[:3] = 0.001 * EE_dot[:3]   # convert from mm/s to m/s
        X_dot[3:] = 0.0174532925 * EE_dot[3:]  # convert from deg/s to rad/s  
        return X_dot
    
    @property
    def position_error(self):
        # actual robot flange position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
        current_position = self.Robot_RT_State.actual_tcp_position[:3]   #  (x, y, z) in mm
        return 0.001 * (current_position - self.position_des)  # convert from mm to m
    
    @property
    def orientation_error(self):
        current_orientation = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # Convert angles from Euler ZYZ (in degrees) to quaternion        

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
        current_rotation_matrix = current_rotation.as_matrix()  # this returns a 3x3 rotation matrix
        rot_error = current_rotation_matrix @ rot_error
        return rot_error.reshape(-1)
    
    def _svd_solve(self, M, threshold=1e-10):
        U, s, V_transp = np.linalg.svd(M)

        # Option-1
        # S_inv = np.diag(s**-1)
        
        # Option-2, Handle small singular values
        s_inv = np.zeros_like(s)
        for i in range(len(s)):
            if s[i] > threshold:
                s_inv[i] = 1.0 / s[i]
            else:
                s_inv[i] = 0.0  # Or apply damping: s[i]/(s[i]^2 + lambda^2)
        
        # Reconstruct inverse
        S_inv = np.zeros_like(M)
        for i in range(len(s)):
            S_inv[i,i] = s_inv[i]
        
        # M^-1 = V * S^-1 * U^T
        M_inv = V_transp.T @ S_inv @ U.T   # V = V_transp.T
        return M_inv
        
    def Mx(self, Mq, J):
        # Mq_inv = np.linalg.inv(Mq)  # This was calculated based on LU-Decomposition which is numerically not very stable
        Mq_inv = self._svd_solve(Mq)  # SVD is more numerically stable when dealing with matrices that might be ill-conditioned
        Mx_inv = J @ (Mq_inv @ J.T)
        if abs(np.linalg.det(Mx_inv)) >= 1e-4:
            # Mx = np.linalg.inv(Mx_inv)
            Mx = self._svd_solve(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-5)
        return Mx
    
    def saturate_torque(self, tau):
        """Limit both the torque rate of change and peak torque values for Doosan A0509 robot"""
        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.85
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0])  # Nm

        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # Clip torque values to stay within limits (both positive and negative)
        tau_saturated = np.clip(tau, -max_torque_limits, max_torque_limits)
        return tau_saturated
    
    def _limit_vel(self, delX):
        scale = np.ones(6)
        lamb = np.array(self.Kp[0,0] * 3 + self.Ko[0,0] * 3) / self.Kv[0,0]
        
        # Apply the sat gains to the x,y,z components
        norm_xyz = np.linalg.norm(delX[:3])
        sat_gain_xyz = self.max_vel[0] / self.Kp * self.Kv
        scale_xyz = self.max_vel[0] / self.Kp * self.Kv
        
        if norm_xyz > sat_gain_xyz:
            scale[:3] *= scale_xyz / norm_xyz
        
        # Apply the sat gains to the a,b,g components
        norm_abg = np.linalg.norm(delX[3:])
        sat_gain_abg = self.max_vel[1] / self.Ko * self.Kv
        scale_abg = self.max_vel[1] / self.Ko * self.Kv
        
        if norm_abg > sat_gain_abg:
            scale[3:] *= scale_abg / norm_abg
        delX = self.Kv * scale * lamb * delX

        return delX

    def calc_friction_torque(self):
        motor_torque = self.Robot_RT_State.actual_motor_torque   # in Nm
        joint_torque = self.Robot_RT_State.actual_joint_torque   # in Nm
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity   # convert from deg/s to rad/s

        term_1 = np.dot(self.K_o, (motor_torque - joint_torque - self.tau_f)) * 0.001
        term_2 = np.dot(self.K_o, np.dot(self.J_m, (self.q_dot_prev - q_dot)))
        self.tau_f = self.tau_f + term_1 + term_2

        self.q_dot_prev = q_dot.copy()

    def control_input(self):
        # define EE-Position & Orientation error in task-space
        error = np.zeros((self.n, 1))
        error[:3,0] = self.position_error
        error[3:,0] = self.orientation_error
                
        if np.all(self.velocity_des == 0.0):
            return self.Kp @ error
        else:
            error_dot = (self.velocity_current - self.velocity_des)[:, np.newaxis]  # velocity_current -> EE-velocity in task-space [Vx, Vy, Vz, ωx, ωy, ωz] in [m/s, rad/s]
            return self.Kp @ error + self.Kv @ error_dot

    def run_controller(self, position, velocity):
        self.start(position, velocity)
        tau_task = np.zeros((self.n,1))
        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate
        
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                # Find Jacobian matrix
                J = self.Robot_RT_State.jacobian_matrix

                # Find Inertia matrix in joint space
                Mq = self.Robot_RT_State.mass_matrix
                
                # Compute control
                Mx = self.Mx(Mq, J)
                U = self.control_input()

                if np.all(self.velocity_des == 0.0):
                    # if there's no desired velocity in task space, compensate for velocity in joint space 
                    q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity   # convert from deg/s to rad/s
                    tau_task = - Mq @ (self.Kv @ q_dot[:,np.newaxis]) - J.T @ (Mx @ U)
                else:
                    tau_task = - J.T @ (Mx @ U)

                # compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque 

                # estimate frictional torque in Nm
                self.calc_friction_torque()
            
                # Compute desired torque
                tau_d = G_torque[:, np.newaxis] + tau_task #+ self.tau_f[:, np.newaxis]

                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d)

                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()
                writedata.time = 0.0
                self.torque_publisher.publish(writedata)

                print(np.linalg.norm(self.position_error))

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
        task = OSC()
        rospy.sleep(2.0)  # Give time for initialization

        # Start controller in a separate thread
        Xd = np.array([100, 0, 600])  # in mm
        Xd_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # in [m/s, rad/s]
        controller_thread = Thread(target=task.run_controller, args=(Xd, Xd_dot)) 
        controller_thread.daemon = True
        controller_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.01)  # This keeps the plot window responsive

    except rospy.ROSInterruptException:
        pass

    finally:
        plt.close('all')  # Clean up plots on exit
