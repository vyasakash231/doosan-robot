#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from pose_transform import eul2quat


class DoosanRecord:
    def __init__(self):
        rospy.init_node('manual_control_node')
        rospy.on_shutdown(self.shutdown)
        
        # Wait for essential services
        rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')
        rospy.wait_for_service('/dsr01a0509/force/task_compliance_ctrl')
        # rospy.wait_for_service('/dsr01a0509/force/set_stiffnessx')
        rospy.wait_for_service('/dsr01a0509/force/release_compliance_ctrl')
        
        # Create service proxies
        self.set_robot_mode = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
        self.task_compliance_ctrl = rospy.ServiceProxy('/dsr01a0509/force/task_compliance_ctrl', TaskComplianceCtrl)
        self.set_stiffness = rospy.ServiceProxy('/dsr01a0509/force/set_stiffnessx', SetStiffnessx)
        self.release_compliance = rospy.ServiceProxy('/dsr01a0509/force/release_compliance_ctrl', ReleaseComplianceCtrl)
        
        # Publishers
        self.stop_pub = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)
        
        # Subscribers
        self.state_sub = rospy.Subscriber('/dsr01a0509/state', RobotState, self.state_callback)

        # Set robot to manual mode
        self.set_robot_mode(0)  # 0 : ROBOT_MODE_MANUAL, (robot LED lights up blue) --> use it for recording demonstration

        # Get button state
        self.get_buttons_service = rospy.ServiceProxy('/dsr01a0509/system/get_buttons_state', GetButtonsState)
        
        self.gripper_close_width = 0
        self.gripper_open_width  = 0.06
        self.gripper_sensitivity= 0.03

    @property
    def button(self):
        return self.get_buttons_service().state[0]   # 0th button is used demo based learning
    
    @property
    def current_velocity(self):
        X_dot = np.zeros(6)
        X_dot[:3] = 0.001 * self.current_vel[:3]   # convert from mm/s to m/s
        X_dot[3:] = 0.0174532925 * self.current_vel[3:]  # convert from deg/s to rad/s  
        return X_dot
    
    def state_callback(self, msg):
        """Store complete info of robot and joint angle as current_posj in degrees"""
        self.q = 0.0174532925 * np.array(msg.current_posj)   # convert from deg to rad
        self.q_dot = 0.0174532925 * np.array(msg.current_velj)   # convert from deg/s to rad/s
        self.current_position = 0.001 * np.array(msg.current_posx)[:3]   # (x, y, z), converted from mm to m
        self.current_euler = 0.0174532925 * np.array(msg.current_posx)[3:]   # (a, b, c) follows Euler ZYZ notation, convert from deg/s to rad/s
        self.current_quat = eul2quat(np.array(msg.current_posx)[3:])   # orientation in quaternion
        self.current_linear_vel = 0.001 * np.array(msg.current_velx)[:3]   # (Vx, Vy, Vz), convert from mm/s to m/s
        self.current_angular_vel = 0.0174532925 * np.array(msg.current_velx)[3:]   # (ωx, ωy, ωz), convert from deg/s to rad/s

    def traj_record(self, trigger=0.005):  # trigger is 5mm
        # Default is [500, 500, 500, 100, 100, 100] -> Reducing these values will make the robot more compliant
        stiffness = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.set_stiffness(stiffness, 0, 0.0) 
        rospy.loginfo("Stiffness set successfully")

        rospy.sleep(0.5)

        compliance = [10, 10, 10, 10, 10, 10]
        self.task_compliance_ctrl(compliance, 0, 0.0)  # time=0 for immediate effect
        rospy.loginfo("Compliance control enabled successfully")

        init_pose = self.current_position
        robot_perturbation = 0
        print("Move robot to start recording.")

        # TO increase the amount of data collected, increase the frequency
        rate = rospy.Rate(25)   # 25Hz = 40ms, 50Hz = 20ms
        
        # observe small movement to start recoding
        while robot_perturbation < trigger:
            robot_perturbation = np.sqrt((self.current_position[0] - init_pose[0])**2 + (self.current_position[1] - init_pose[1])**2 + (self.current_position[2] - init_pose[2])**2)
        
        self.recorded_q = self.q  # in rad
        self.recorded_trajectory = self.current_position  # in m
        self.recorded_orientation = self.current_quat  # quaternions
        self.recorded_linear_velocity = self.current_linear_vel  # in m/s
        self.recorded_angular_velocity = self.current_angular_vel  # in rad/s
        # self.recorded_gripper = self.gripper_open_width
   
        while self.button:  # if the cockpit button is pressed
            # if self.gripper_width < (self.gripper_open_width - self.gripper_sensitivity):
            #     print("Close gripper")
            #     self.grip_value = 0   # Close the gripper
            # else:
            #     print("Open gripper")
            #     self.grip_value = self.gripper_open_width   # Open the gripper
           
            self.recorded_q = np.vstack((self.recorded_q, self.q))  # shape: (N, 6) 
            self.recorded_trajectory = np.vstack((self.recorded_trajectory, self.current_position))  # shape: (N, 3) 
            self.recorded_orientation = np.vstack((self.recorded_orientation, self.current_quat))  # shape: (N, 4)
            self.recorded_linear_velocity = np.vstack((self.recorded_linear_velocity, self.current_linear_vel))  # shape: (N, 3)
            self.recorded_angular_velocity = np.vstack((self.recorded_angular_velocity, self.current_angular_vel))  # shape: (N, 3)
            # self.recorded_gripper = np.vstack((self.recorded_gripper, self.grip_value))

            rate.sleep()

        # goal = np.concatenate((self.current_position, self.current_quat))
        rospy.loginfo("Ending trajectory recording")

    def shutdown(self):
        """Cleanup when shutting down"""
        try:
            self.release_compliance()  # Release compliance control
            self.set_robot_mode(1)  # 1 : ROBOT_MODE_AUTONOMOUS, this will stop teaching mode (robot LED lights up in white)
            self.stop_pub.publish(stop_mode=STOP_TYPE_SLOW)  # Quick stop
            rospy.loginfo("Robot shutdown complete")
        except Exception as e:
            rospy.logerr(f"Error during shutdown: {e}")
        return

    def save(self, name='demo'):
        curr_dir=os.getcwd()
        np.savez(curr_dir+ '/data/' + str(name) + '.npz',
                q=self.recorded_q,
                traj=self.recorded_trajectory,
                ori=self.recorded_orientation,
                vel=self.recorded_linear_velocity,
                omega=self.recorded_angular_velocity
                #  grip=self.recorded_gripper
                )

    def load(self, name='demo'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.recorded_q=data['q'],
        self.recorded_trajectory = data['traj']
        self.recorded_orientation = data['ori']
        self.recorded_linear_velocity = data['vel']
        self.recorded_angular_velocity = data['omega']
        # self.recorded_gripper = data['grip']

if __name__ == "__main__":
    # move to initial position first
    p1= posj(0,0,90,0,90,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
    movej(p1, vel=40, acc=20)
    
    time.sleep(1.0)

    try:
        controller = DoosanRecord()
        rospy.loginfo("Robot setup complete - Ready for manual demonstration")
        controller.traj_record()
        controller.save(name="demo")
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
