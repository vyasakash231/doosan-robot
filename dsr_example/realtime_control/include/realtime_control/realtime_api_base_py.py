#!/usr/bin/env python
from dataclasses import dataclass
import numpy as np
from typing import List, Optional
import rospy
from rt_shared import RT_STATE, NUMBER_OF_JOINT, rt_state_lock
import threading
from enum import Enum, auto

# Type alias for 6x6 matrix and 6x1 vector using NumPy
Matrix6f = np.ndarray  # 6x6 matrix
Vector6f = np.ndarray  # 6x1 vector

class RobotMode(Enum):
    MANUAL = 0
    AUTONOMOUS = 1
    MEASURE = 2

class RobotState(Enum):
    INITIALIZING = 0
    STANDBY = 1
    MOVING = 2
    SAFE_OFF = 3
    TEACHING = 4
    SAFE_STOP = 5
    EMERGENCY_STOP = 6
    HOMMING = 7
    RECOVERY = 8
    SAFE_STOP2 = 9
    SAFE_OFF2 = 10

class SetOnRtMonitoringDataNode:
    """Base class for RT monitoring node"""
    def __init__(self):
        rospy.init_node('set_on_rt_monitoring_data')
        self._timer = None
        self._context_timer = None
        self._shutdown = threading.Event()

    def _on_rt_monitoring_data(self, data):
        """Callback for RT monitoring data"""
        if self._shutdown.is_set():
            return

        with rt_state_lock:
            RT_STATE.time_stamp = data.time_stamp
            
            # Update vectors
            np.copyto(RT_STATE.actual_joint_position, data.actual_joint_position)
            np.copyto(RT_STATE.actual_joint_velocity, data.actual_joint_velocity)
            np.copyto(RT_STATE.gravity_torque, data.gravity_torque)
            
            # Update matrices
            np.copyto(RT_STATE.mass_matrix, np.array(data.mass_matrix).reshape(6, 6))
            np.copyto(RT_STATE.coriolis_matrix, np.array(data.coriolis_matrix).reshape(6, 6))
            np.copyto(RT_STATE.jacobian_matrix, np.array(data.jacobian_matrix).reshape(6, 6))

class ReadDataRtNodeBase:
    """Base class for reading RT data"""
    def __init__(self):
        rospy.init_node('read_data_rt')
        self._timer = None
        self._context_timer = None
        self._shutdown = threading.Event()

    def read_data_rt_api(self):
        """Template method for reading RT data"""
        raise NotImplementedError

class TorqueRtNodeBase:
    """Base class for torque RT control"""
    def __init__(self):
        rospy.init_node('torque_rt')
        self._timer = None
        self._context_timer = None
        self._shutdown = threading.Event()
        self.trq_d = np.zeros(NUMBER_OF_JOINT)

    def torque_rt_api(self):
        """Template method for torque RT control"""
        raise NotImplementedError

    def gravity_compensation(self) -> Vector6f:
        """Template method for gravity compensation"""
        raise NotImplementedError

    def external_force_resist(self) -> Vector6f:
        """Template method for external force resistance"""
        raise NotImplementedError
