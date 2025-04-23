#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, RealTimePlot
from scipy.spatial.transform import Rotation


class JointStiffnessControl(Robot):
    def __init__(self):
        self.shutdown_flag = False  

        # Initialize the plotter in the main thread
        self.plotter = RealTimePlot()
        self.plotter.setup_gravity()

        # parameters for friction estimation
        self.K_o = np.diag([0.1, 0.1, 0.2, 0.2, 0.25, 0.4])

        # Initial estimated frictional torque
        self.tau_f = np.zeros(self.n)

        super().__init__()

    def start(self):
        """Initialize controller with current robot state"""
        # Set equilibrium point to current state
        self.q_des = 0.0174532925 * self.Robot_RT_State.actual_joint_position.copy()
        # self.q_dot_des = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs.copy()
        self.q_dot_prev = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity.copy() 
        rospy.loginfo("CartesianImpedanceController: Controller started")

    def set_compliance_parameters(self, joint_stiffness):
        self.K_joint = joint_stiffness * np.eye(6)  # Update stiffness matrix
        self.D_joint = 1.25 * np.sqrt(joint_stiffness) * np.eye(6)   # Update damping matrix (critically damped)

    def saturate_torque(self, tau):
        """
        Limit both the torque rate of change and peak torque values for Doosan A0509 robot
        """
        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.9
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0]) # Nm

        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # Clip torque values to stay within limits (both positive and negative)
        tau_saturated = np.clip(tau, -max_torque_limits, max_torque_limits)
        return tau_saturated

    def plot_data(self):
        try:
            self.plotter.update_gravity(self.data.actual_motor_torque, 
                                        self.data.raw_force_torque, 
                                        self.data.actual_joint_torque, 
                                        self.data.raw_joint_torque)
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

    def calc_friction_torque(self):
        motor_torque = self.Robot_RT_State.actual_motor_torque
        joint_torque = self.Robot_RT_State.actual_joint_torque
        q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity  # convert from deg/s to rad/s

        term_1 = np.dot(self.K_o, (motor_torque - joint_torque - self.tau_f)) * 0.01
        self.tau_f = self.tau_f + term_1 

        self.q_dot_prev = q_dot.copy()

    def run_controller(self, joint_stiffness):
        self.start()
        self.set_compliance_parameters(joint_stiffness)
        tau_task = np.zeros((6,1))

        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate
        
        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                self.q = 0.0174532925 * self.Robot_RT_State.actual_joint_position     #  convert deg to rad
                self.q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity     #  convert deg/s to rad/s

                # Compute control
                error = (self.q - self.q_des)[:, np.newaxis]
                # error_dot = (self.q_dot - self.q_dot_des)[:, np.newaxis]
                error_dot = self.q_dot[:, np.newaxis]  # where, self.q_dot_des = 0
                tau_task = - self.D_joint @ error_dot - self.K_joint @ error
                
                # compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque 

                # estimate frictional torque in Nm
                self.calc_friction_torque()
            
                # Compute desired torque
                tau_d = tau_task + G_torque[:, np.newaxis] + self.tau_f[:, np.newaxis]

                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d)

                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()
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
        task = JointStiffnessControl()
        rospy.sleep(2)  # Give time for initialization

        # Start controller in a separate thread
        controller_thread = Thread(target=task.run_controller, args=(50.0,))  # translation stiff -> N/m, rotational stiffness -> Nm/rad 
        controller_thread.daemon = True
        controller_thread.start()
        
        # Keep the main thread running for the plot
        while not rospy.is_shutdown():
            plt.pause(0.01)  # This keeps the plot window responsive

    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close('all')  # Clean up plots on exit










