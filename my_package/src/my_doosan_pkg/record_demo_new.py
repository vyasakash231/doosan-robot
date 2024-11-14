#!/usr/bin/env python3

import rospy
import os
from math import *
import numpy as np
import time
import threading  # Threads are a way to run multiple tasks concurrently within a single process. By using threads, you can perform multiple operations simultaneously, which can be useful for tasks like handling asynchronous events, running background tasks.
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp"))) # get import path : DSR_ROBOT.py 

import DR_init  # at doosan-robot/common/imp/
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *  # at doosan-robot/common/imp/
from DR_common import *  # at doosan-robot/common/imp/

from dsr_msgs.msg import RobotStop, RobotState
from dsr_msgs.srv import *
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation
import rospy
import numpy as np

class DoosanManualControl:
    def __init__(self):
        rospy.init_node('manual_control_node')
        rospy.on_shutdown(self.shutdown)
        
        # Wait for essential services
        rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')
        rospy.wait_for_service('/dsr01a0509/force/task_compliance_ctrl')
        rospy.wait_for_service('/dsr01a0509/force/set_stiffnessx')
        rospy.wait_for_service('/dsr01a0509/force/release_compliance_ctrl')
        
        # Create service proxies
        self.set_robot_mode = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
        self.task_compliance_ctrl = rospy.ServiceProxy('/dsr01a0509/force/task_compliance_ctrl', TaskComplianceCtrl)
        self.set_stiffness = rospy.ServiceProxy('/dsr01a0509/force/set_stiffnessx', SetStiffnessx)
        self.release_compliance = rospy.ServiceProxy('/dsr01a0509/force/release_compliance_ctrl', ReleaseComplianceCtrl)
        
        # Publishers
        self.stop_pub = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)
        
        # Subscribers
        self.joint_sub = rospy.Subscriber('/dsr01a0509/joint_states', JointState, self.joint_callback)
        self.state_sub = rospy.Subscriber('/dsr01a0509/state', RobotState, self.state_callback)
        self.tf_sub = rospy.Subscriber('/tf', TFMessage, self.tf_callback)
        
        # Initialize robot state
        self.joint_positions = None
        self.robot_state = None
        self.tf_matrices = np.zeros((6,4,4))
    
    def setup_manual_mode(self):
        """Set up the robot for manual demonstration recording"""
        try:
            # Set robot to manual mode
            self.set_robot_mode(ROBOT_MODE_MANUAL)
            rospy.sleep(0.5)
            
            # Set lower stiffness values for easier manual movement
            # Default is [500, 500, 500, 100, 100, 100]
            # Reducing these values will make the robot more compliant
            stiffness = [200, 200, 200, 50, 50, 50]  # Reduced values for more compliance
            try:
                self.set_stiffness(stiffness, 0)  # time=0 for immediate effect
                rospy.loginfo("Stiffness set successfully")
            except Exception as e:
                rospy.logerr(f"Failed to set stiffness: {e}")
                return False
            
            rospy.sleep(0.5)
            
            # Enable compliance control
            # Default is [3000, 3000, 3000, 200, 200, 200]
            # Using lower values for more compliance
            compliance = [1000, 1000, 1000, 100, 100, 100]
            try:
                self.task_compliance_ctrl(compliance, 0)  # time=0 for immediate effect
                rospy.loginfo("Compliance control enabled successfully")
            except Exception as e:
                rospy.logerr(f"Failed to enable compliance control: {e}")
                return False
            
            rospy.loginfo("Robot setup complete - Ready for manual demonstration")
            return True
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
    
    def joint_callback(self, msg):
        """Store joint positions (in radians)"""
        self.joint_positions = [round(i,4) for i in list(msg.position)]
    
    def state_callback(self, msg):
        """Store robot state"""
        self.robot_state = msg
        
    def tf_callback(self, msg):
        """Process transformation matrices"""
        i = 0
        T_e_0 = np.eye(4)
        
        for transform in msg.transforms:
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            
            t = np.array([translation.x, translation.y, translation.z])
            r = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
            
            H = np.eye(4)
            H[:3,:3] = r.as_matrix()
            H[:3,-1] = t.T
            
            T_e_0 = np.dot(T_e_0, H)
            self.tf_matrices[i,:,:] = H
            i += 1
    
    def shutdown(self):
        """Cleanup when shutting down"""
        try:
            # Release compliance control
            self.release_compliance()
            # Quick stop
            self.stop_pub.publish(stop_mode=STOP_TYPE_QUICK)
            rospy.loginfo("Robot shutdown complete")
        except Exception as e:
            rospy.logerr(f"Error during shutdown: {e}")

if __name__ == "__main__":
    try:
        controller = DoosanManualControl()
        if controller.setup_manual_mode():
            rospy.spin()
    except rospy.ROSInterruptException:
        pass