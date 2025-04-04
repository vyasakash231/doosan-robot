#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *


class DoosanManualControl:
    def __init__(self):
        rospy.init_node('manual_control_node')
        rospy.on_shutdown(self.shutdown)
        
        # Wait for essential services
        rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')
        # rospy.wait_for_service('/dsr01a0509/force/set_stiffnessx')
        
        # Create service proxies
        self.set_robot_mode = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
        self.set_stiffness = rospy.ServiceProxy('/dsr01a0509/force/set_stiffnessx', SetStiffnessx)
        
        # Publishers
        self.stop_pub = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)
        
        # Subscribers
        self.joint_sub = rospy.Subscriber('/dsr01a0509/joint_states', JointState, self.joint_callback)
        self.state_sub = rospy.Subscriber('/dsr01a0509/state', RobotState, self.state_callback)
    
    def setup_manual_mode(self):
        """Set up the robot for manual demonstration recording"""
        try:
            # Set robot to manual mode
            self.set_robot_mode(0)  # 0 : ROBOT_MODE_MANUAL, (robot LED lights up blue) --> use it for recording demonstration
            rospy.sleep(1.0)
            
            # Default is [500, 500, 500, 100, 100, 100] -> Reducing these values will make the robot more compliant
            stiffness = [10, 10, 10, 10, 10, 10]
            self.set_stiffness(stiffness, 0, 0.0) 
            rospy.loginfo("Robot stiffness adjusted")
        
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
    
    def joint_callback(self, msg):
        """Store joint positions (in radians)"""
        self.joint_positions = [round(i,4) for i in list(msg.position)] # 
    
    def state_callback(self, msg):
        """Store complete info of robot and joint angle as current_posj in degrees"""
        self.q = 0.0174532925 * np.array(msg.current_posj)   # convert from deg to rad
        self.q_dot = 0.0174532925 * np.array(msg.current_velj)   # convert from deg/s to rad/s
        self.current_pose = np.array(msg.current_posx)  # (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
        self.current_vel = np.array(msg.current_velx)    # (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm/s, deg/s]
    
    def shutdown(self):
        """Cleanup when shutting down"""
        try:
            self.stop_pub.publish(stop_mode=STOP_TYPE_QUICK)  # Quick stop
            self.set_robot_mode(1)  # 1 : ROBOT_MODE_AUTONOMOUS, this will stop teaching mode (robot LED lights up in white)
            rospy.loginfo("Robot shutdown complete")
        except Exception as e:
            rospy.logerr(f"Error during shutdown: {e}")

if __name__ == "__main__":
    controller = None
    try:
        controller = DoosanManualControl()
        controller.setup_manual_mode()
        rospy.loginfo("Robot setup complete - Ready for manual demonstration")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
