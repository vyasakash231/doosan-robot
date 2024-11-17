#!/usr/bin/env python
import rospy
from rt_shared import BaseRtNode, NUMBER_OF_JOINT, rt_connected, rt_output_set, rt_started
import numpy as np
from threading import Thread
from dsr_msgs.msg import TorqueRtStream, ServojRtStream, ServolRtStream

class RtInitNodeBase(BaseRtNode):
    """Base class for RT initialization node"""
    def __init__(self):
        super().__init__('rt_init')
        self.client_threads = []
        
    def connect_rt_control_client(self):
        """Template method for RT control connection"""
        raise NotImplementedError
        
    def set_rt_control_output_client(self):
        """Template method for RT control output setting"""
        raise NotImplementedError
        
    def start_rt_control_client(self):
        """Template method for RT control start"""
        raise NotImplementedError
        
    def cleanup(self):
        """Clean up threads and resources"""
        for thread in self.client_threads:
            if thread.is_alive():
                thread.join()
                rospy.loginfo(f"{thread.name} joined")

class TorqueRtNodeBase(BaseRtNode):
    """Base class for torque RT node"""
    def __init__(self):
        super().__init__('torque_rt')
        # Control parameters
        self.q = np.zeros(NUMBER_OF_JOINT)
        self.q_dot = np.zeros(NUMBER_OF_JOINT)
        self.q_d = np.array([0, 0, 90, 0, 90, 0])
        self.q_dot_d = np.zeros(NUMBER_OF_JOINT)
        self.trq_g = np.zeros(NUMBER_OF_JOINT)
        self.trq_d = np.zeros(NUMBER_OF_JOINT)
        
        # Control gains
        self.kp = np.array([0.01, 0.01, 0.08, 0.01, 0.01, 0.01])
        self.kd = np.zeros(NUMBER_OF_JOINT)

    def torque_rt_stream_publisher(self):
        """Template method for torque RT stream publishing"""
        raise NotImplementedError

class ServojRtNodeBase(BaseRtNode):
    """Base class for servoj RT node"""
    def __init__(self):
        super().__init__('servoj_rt')
        self.pos_d = np.zeros(NUMBER_OF_JOINT)
        self.vel_d = np.zeros(NUMBER_OF_JOINT)
        self.acc_d = np.zeros(NUMBER_OF_JOINT)
        self.time_d = 0.0

    def servoj_rt_stream_publisher(self):
        """Template method for servoj RT stream publishing"""
        raise NotImplementedError

class ServolRtNodeBase(BaseRtNode):
    """Base class for servol RT node"""
    def __init__(self):
        super().__init__('servol_rt')
        self.pos_d = np.zeros(NUMBER_OF_JOINT)
        self.vel_d = np.zeros(NUMBER_OF_JOINT)
        self.acc_d = np.zeros(NUMBER_OF_JOINT)
        self.time_d = 0.0

    def servol_rt_stream_publisher(self):
        """Template method for servol RT stream publishing"""
        raise NotImplementedError

class ReadDataRtNodeBase(BaseRtNode):
    """Base class for reading RT data"""
    def __init__(self):
        super().__init__('read_data_rt')
        self.client_thread = None
        
    def read_data_rt_client(self):
        """Template method for reading RT data"""
        raise NotImplementedError
        
    def cleanup(self):
        """Clean up thread and resources"""
        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.join()
            rospy.loginfo("client_thread joined")
