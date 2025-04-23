#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot

class LearningFromDemonstration(DoosanRecord):
    def __init__(self):
        self.speedl_publisher = rospy.Publisher('/dsr01a0509/speedl_stream', SpeedlStream, queue_size=10)    # SpeedlStream -> Topic message is the asynchronous motion message, and the next command is executed at the same time the motion begins. 
        self.servol_publisher = rospy.Publisher('/dsr01a0509/servol_stream', ServolStream, queue_size=10)    # ServolStream -> Topic message is the asynchronous motion message, and the next command is executed at the same time the motion begins.  
        super().__init__()

    def move_to_pose(self, pos, orient):
        X = 1000 * pos   # convert m to mm
        O = quat2euler(orient)  # convert quaternion to euler ZYZ (in degrees)
        p = posx(X[0], X[1], X[2], O[3], O[4], O[5])  # posx(x, y, z, w, p, r) This function designates the task space in coordinate values.
        
        writedata = ServolStream()
        writedata.pos = p
        writedata.time = 1.0
        self.torque_publisher.publish(writedata)
        
    def execute_task(self):
        stiffness = [500.0, 500.0, 500.0, 100.0, 100.0, 100.0]
        self.set_stiffness(stiffness, 0, 0.0) 

        start_position = self.recorded_trajectory[:,0]
        start_orientation = self.recorded_orientation[:,0]
        self.move_to_pose(start_position, start_orientation)
