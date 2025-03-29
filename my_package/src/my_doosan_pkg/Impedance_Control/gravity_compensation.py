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
        self.plotter.setup_plots_1()

        # Initial estimated frictional torque
        self.fric_torques = np.zeros(self.n)

        self.J_m = np.diag([0.0004956, 0.0004956, 0.0001839, 0.00009901, 0.00009901, 0.00009901])
        self.K_o = np.diag([0.1, 0.1, 0.2, 0.15, 0.25, 0.5])

        super().__init__()

    def plot_data(self, data):
        try:
            self.plotter.update_data_1(data.actual_motor_torque, data.raw_force_torque, data.actual_joint_torque, data.raw_joint_torque)
        except Exception as e:
            rospy.logwarn(f"Error adding plot data: {e}")

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
    
                torque = G_torques + self.fric_torques
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










