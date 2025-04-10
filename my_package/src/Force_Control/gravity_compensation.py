#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, RealTimePlot


class GravityCompensation(Robot):
    def __init__(self):
        self.shutdown_flag = False  

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_plots_1()   # to plot torques
        # self.plotter.setup_task_plot()   # to plot EE-velocities

        # Initial estimated frictional torque
        self.fric_torques = np.zeros(self.n)

        self.J_m = np.diag([0.0004956, 0.0004956, 0.0001839, 0.00009901, 0.00009901, 0.00009901])
        self.K_o = np.diag([0.1, 0.1, 0.2, 0.15, 0.25, 0.5])

        super().__init__()

    def _svd_solve(self, M, threshold=1e-10):
        U, s, V_transp = np.linalg.svd(M)

        # Option-1
        S_inv = np.diag(s**-1)
        
        # # Option-2, Handle small singular values
        # s_inv = np.zeros_like(s)
        # for i in range(len(s)):
        #     if s[i] > threshold:
        #         s_inv[i] = 1.0 / s[i]
        #     else:
        #         s_inv[i] = 0.0  # Or apply damping: s[i]/(s[i]^2 + lambda^2)
        
        # # Reconstruct inverse
        # S_inv = np.zeros_like(M)
        # for i in range(len(s)):
        #     S_inv[i,i] = s_inv[i]
        
        # M^-1 = V * S^-1 * U^T
        M_inv = V_transp.T @ S_inv @ U.T   # V = V_transp.T
        return M_inv

    @property
    def q_ddot(self):
        Mq = self.Robot_RT_State.mass_matrix
        C = self.Robot_RT_State.coriolis_matrix
        G = self.Robot_RT_State.gravity_torque   # in Nm
        tau = self.Robot_RT_State.actual_joint_torque   # in Nm

        if abs(np.linalg.det(Mq)) >= 1e-4:
            # Mq_inv = np.linalg.inv(Mq)
            Mq_inv = self._svd_solve(Mq)
        else:
            Mq_inv = np.linalg.pinv(Mq)

        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity   # convert deg/s to rad/s
        q_ddot = Mq_inv @ (tau[:, np.newaxis] - C @ q_dot[:, np.newaxis] - G[:, np.newaxis])
        return q_ddot.reshape(-1)  # in rad/s^2
    
    @property
    def current_acceleration(self):
        q = 0.0174532925 * self.Robot_RT_State.actual_joint_position   # convert deg to rad
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity   # convert deg/s to rad/s
        
        J,_,_ = self.kinematic_model.Jacobian(q)
        J_dot,_,_ = self.kinematic_model.Jacobian_dot(q, q_dot)
       
        X_ddot = J_dot @ q_dot[:, np.newaxis] + J @ self.q_ddot[:, np.newaxis]
        return X_ddot.reshape(-1)

    def plot_data(self):
        try:
            self.plotter.update_data_1(self.data.actual_motor_torque, 
                                       self.data.raw_force_torque, 
                                       self.data.actual_joint_torque, 
                                       self.data.raw_joint_torque)  # external_tcp_force
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    # def plot_data(self):
    #     X_dot = np.zeros(6)
    #     try:
    #         # J = self.Robot_RT_State.jacobian_matrix
    #         q = 0.0174532925 * self.Robot_RT_State.actual_joint_position   # convert deg to rad
    #         J, _, _ = self.kinematic_model.Jacobian(q)
    #         calc_vel = 0.0174532925 * (J @ self.Robot_RT_State.actual_joint_velocity[:, np.newaxis]).reshape(-1)

    #         X_dot[:3] = 0.001 * self.Robot_RT_State.actual_tcp_velocity[:3]   # convert from mm/s to m/s
    #         X_dot[3:] = 0.0174532925 * self.Robot_RT_State.actual_tcp_velocity[3:]  # convert from deg/s to rad/s  
    #         self.plotter.update_task_data(X_dot, calc_vel)
    #     except Exception as e:
    #         rospy.logwarn(f"Error adding plot data: {e}")

    def calc_friction_torque(self):
        motor_torque = self.Robot_RT_State.actual_motor_torque
        joint_torque = self.Robot_RT_State.actual_joint_torque
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity  # convert from deg/s to rad/s

        term_1 = np.dot(self.K_o, (motor_torque - joint_torque - self.fric_torques)) * 0.001
        term_2 = np.dot(self.K_o, np.dot(self.J_m, (self.q_dot_prev - q_dot)))
        self.fric_torques = self.fric_torques + term_1 + term_2

        self.q_dot_prev = q_dot.copy()

    def run_controller(self):
        self.q_dot_prev = self.Robot_RT_State.actual_joint_velocity.copy() 
        rate = rospy.Rate(self.write_rate)
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                G_torques = self.Robot_RT_State.gravity_torque  # calculate gravitational torque in Nm
                self.calc_friction_torque() #  estimate frictional torque in Nm

                print(self.q_ddot)
    
                torque = G_torques #+ self.fric_torques
                writedata = TorqueRTStream()
                writedata.tor = torque
                writedata.time = 0.0
                
                self.torque_publisher.publish(writedata)
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
        task = GravityCompensation()
        rospy.sleep(2)  # Give time for initialization
        
        # Start G control in a separate thread
        control_thread = Thread(target=lambda: task.run_controller())
        control_thread.daemon = True
        control_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.1)  # This keeps the plot window responsive
            
    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close('all')  # Clean up plots on exit










