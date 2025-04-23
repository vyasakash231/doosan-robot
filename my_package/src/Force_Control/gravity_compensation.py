#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, RealTimePlot, Filters


class GravityCompensation(Robot):
    def __init__(self):
        self.shutdown_flag = False  

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_gravity()   # to plot torques
        # self.plotter.setup_task_plot()   # to plot EE-velocities

        # Initial estimated frictional torque
        self.fric_torques = np.zeros(self.n)

        # self.J_m = np.diag([0.0004956, 0.0004956, 0.0001839, 0.00009901, 0.00009901, 0.00009901])
        self.Ko = np.diag([0.1, 0.05, 0.125, 0.05, 0.1, 0.2])

        self.activate_tool_compensation = True

        self.filter = Filters()

        super().__init__()
    
    @property
    def external_wrench_in_base_frame(self):
        """I have remove force-Fz due to EE-block whose weight is approx. 2.25Kg"""
        m, g = 2.25, -9.81
        Force_due_to_EE_weight = np.array([[0.0],[0.0],[m*g]])  # Fz = m x g = 2.25Kg x -9.81m/s^2
        F_E_E = self.filter.low_pass_filter_torque(np.array(self.Robot_RT_State.raw_force_torque))   # Wrench in EE frame

        EE_pose = np.array(self.Robot_RT_State.actual_tcp_position)   #  (x, y, z, a, b, c) in mm, deg
        R_E_0 = self.euler2mat(EE_pose[3:])  # euler-angles in deg
        F_E_0 = R_E_0 @ F_E_E[:3][:,np.newaxis] - Force_due_to_EE_weight 
        M_E_0 = R_E_0 @ F_E_E[3:][:,np.newaxis] + np.cross(0.001*EE_pose[:3], F_E_0.reshape(-1))[:, np.newaxis]
        ext_wrench = np.concatenate((F_E_0.reshape(-1), M_E_0.reshape(-1)))  # Wrench in Base frame
        return ext_wrench

    @property
    def q_ddot(self):
        Mq = self.Robot_RT_State.mass_matrix
        C = self.Robot_RT_State.coriolis_matrix
        G = self.Robot_RT_State.gravity_torque   # in Nm
        tau = self.Robot_RT_State.actual_joint_torque   # in Nm

        if abs(np.linalg.det(Mq)) >= 1e-4:
            # Mq_inv = np.linalg.inv(Mq)   # very unstable
            Mq_inv = self._svd_solve(Mq)   
        else:
            Mq_inv = np.linalg.pinv(Mq)

        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert deg/s to rad/s
        q_ddot = Mq_inv @ (tau[:, np.newaxis] - self.fric_torques[:, np.newaxis] - C @ q_dot[:, np.newaxis] - G[:, np.newaxis])
        return q_ddot.reshape(-1)  # in rad/s^2
    
    @property
    def current_acceleration(self):
        q = 0.0174532925 * self.Robot_RT_State.actual_joint_position   # convert deg to rad
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert deg/s to rad/s
        
        J,_,_ = self.kinematic_model.Jacobian(q)
        J_dot,_,_ = self.kinematic_model.Jacobian_dot(q, q_dot)
       
        X_ddot = J_dot @ q_dot[:, np.newaxis] + J @ self.q_ddot[:, np.newaxis]
        return X_ddot.reshape(-1)

    def plot_data(self):
        try:
            J = self.Robot_RT_State.jacobian_matrix
            external_wrench = self.external_wrench_in_base_frame
            ext_torque = - J.T @ external_wrench[:, np.newaxis]

            self.plotter.update_gravity(self.data.actual_motor_torque, 
                                       self.data.raw_force_torque, 
                                       self.data.actual_joint_torque, 
                                        # self.data.external_joint_torque, 
                                    #    self.data.raw_joint_torque
                                       external_wrench
                                        # ext_torque.reshape(-1)
                                       ) 
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    # def plot_data(self):
    #     X_dot = np.zeros(6)
    #     try:
    #         # J = self.Robot_RT_State.jacobian_matrix
    #         q = 0.0174532925 * self.Robot_RT_State.actual_joint_position   # convert deg to rad
    #         J, _, _ = self.kinematic_model.Jacobian(q)
    #         calc_vel = 0.0174532925 * (J @ self.Robot_RT_State.actual_joint_velocity_abs[:, np.newaxis]).reshape(-1)

    #         X_dot[:3] = 0.001 * self.Robot_RT_State.actual_tcp_velocity[:3]   # convert from mm/s to m/s
    #         X_dot[3:] = 0.0174532925 * self.Robot_RT_State.actual_tcp_velocity[3:]  # convert from deg/s to rad/s  
    #         self.plotter.update_task_data(X_dot, calc_vel)
    #     except Exception as e:
    #         rospy.logwarn(f"Error adding plot data: {e}")

    def calc_friction_torque(self):
        motor_torque = self.Robot_RT_State.actual_motor_torque
        joint_torque = self.Robot_RT_State.actual_joint_torque
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs  # convert from deg/s to rad/s

        term_1 = np.dot(self.Ko, (motor_torque - joint_torque - self.fric_torques)) * 0.005
        self.fric_torques = self.fric_torques + term_1 

        self.q_dot_prev = q_dot.copy()

    def run_controller(self):
        self.q_dot_prev = self.Robot_RT_State.actual_joint_velocity_abs.copy() 
        rate = rospy.Rate(self.write_rate)
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                G_torques = self.Robot_RT_State.gravity_torque  # calculate gravitational torque in Nm
                self.calc_friction_torque() #  estimate frictional torque in Nm

                # position = self.Robot_RT_State.actual_tcp_position[:3].copy()    # (x, y, z) in mm
                # orientation = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:].copy())  # Convert angles from Euler ZYZ (in degrees) to quaternion        

                # q = 0.0174532925 * np.array(self.Robot_RT_State.actual_joint_position)   # convert from deg to rad
                # pose,_,_ = self.kinematic_model.FK(q)
                # print(position, orientation)
                # print(pose)
                # print("===========================================================")
                
                torque = G_torques + self.fric_torques 
                writedata = TorqueRTStream()
                writedata.tor = torque
                writedata.time = 0.0

                # print(get_tool_force())
                
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










