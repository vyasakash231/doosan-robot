#!/usr/bin/env python
import numpy as np
from threading import Lock
import threading
from dataclasses import dataclass
from typing import List, Dict
import rospy

# Constants
NUMBER_OF_JOINT = 6
NUMBER_OF_TASK = 6

# Global flags (using threading.Event for atomic operations)
rt_connected = threading.Event()
rt_output_set = threading.Event()
rt_started = threading.Event()

@dataclass
class RT_STATE:
    """Data class representing robot state information"""
    def __init__(self):
        # Timestamp
        self.time_stamp: float = 0.0
        
        # Joint positions and velocities
        self.actual_joint_position: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.actual_joint_position_abs: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.actual_joint_velocity: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.actual_joint_velocity_abs: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        
        # TCP and flange positions/velocities
        self.actual_tcp_position: np.ndarray = np.zeros(NUMBER_OF_TASK)
        self.actual_tcp_velocity: np.ndarray = np.zeros(NUMBER_OF_TASK)
        self.actual_flange_position: np.ndarray = np.zeros(NUMBER_OF_TASK)
        self.actual_flange_velocity: np.ndarray = np.zeros(NUMBER_OF_TASK)
        
        # Torque and force measurements
        self.actual_motor_torque: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.actual_joint_torque: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.raw_joint_torque: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.raw_force_torque: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.external_joint_torque: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.external_tcp_force: np.ndarray = np.zeros(NUMBER_OF_TASK)
        
        # Target positions/velocities
        self.target_joint_position: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.target_joint_velocity: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.target_joint_acceleration: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.target_motor_torque: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.target_tcp_position: np.ndarray = np.zeros(NUMBER_OF_TASK)
        self.target_tcp_velocity: np.ndarray = np.zeros(NUMBER_OF_TASK)
        
        # Matrices
        self.jacobian_matrix: np.ndarray = np.zeros((NUMBER_OF_JOINT, NUMBER_OF_JOINT))
        self.gravity_torque: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.coriolis_matrix: np.ndarray = np.zeros((NUMBER_OF_JOINT, NUMBER_OF_JOINT))
        self.mass_matrix: np.ndarray = np.zeros((NUMBER_OF_JOINT, NUMBER_OF_JOINT))
        
        # Robot state and configuration
        self.solution_space: int = 0
        self.singularity: float = 0.0
        self.operation_speed_rate: float = 0.0
        self.joint_temperature: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        
        # I/O states
        self.controller_digital_input: int = 0
        self.controller_digital_output: int = 0
        self.controller_analog_input_type: np.ndarray = np.zeros(2, dtype=np.uint8)
        self.controller_analog_input: np.ndarray = np.zeros(2)
        self.controller_analog_output_type: np.ndarray = np.zeros(2, dtype=np.uint8)
        self.controller_analog_output: np.ndarray = np.zeros(2)
        self.flange_digital_input: int = 0
        self.flange_digital_output: int = 0
        self.flange_analog_input: np.ndarray = np.zeros(4)
        
        # Encoder data
        self.external_encoder_strobe_count: np.ndarray = np.zeros(2, dtype=np.uint8)
        self.external_encoder_count: np.ndarray = np.zeros(2, dtype=np.uint32)
        
        # Goal positions
        self.goal_joint_position: np.ndarray = np.zeros(NUMBER_OF_JOINT)
        self.goal_tcp_position: np.ndarray = np.zeros(NUMBER_OF_TASK)
        
        # Robot mode and state
        self.robot_mode: int = 0  # ROBOT_MODE_MANUAL(0), ROBOT_MODE_AUTONOMOUS(1), ROBOT_MODE_MEASURE(2)
        self.robot_state: int = 0  # STATE_INITIALIZING(0), STATE_STANDBY(1), etc.
        self.control_mode: int = 0  # position control mode, torque mode

class BaseRtNode:
    """Base class for RT nodes with common functionality"""
    def __init__(self, node_name: str):
        self._node_name = node_name
        self._lock = Lock()
        self._shutdown_flag = threading.Event()
        
    def shutdown(self):
        """Safely shutdown the node"""
        self._shutdown_flag.set()
        
    @property
    def is_shutdown(self):
        """Check if node is shutting down"""
        return self._shutdown_flag.is_set()

class ContextSwitchesCounter:
    """Counter for monitoring context switches"""
    def __init__(self, who=None):  # who parameter replaced with None as RUSAGE_THREAD not needed in Python
        self._count = 0
        self._lock = Lock()
        self._initialized = False
        
    def get(self):
        """Get the number of context switches"""
        with self._lock:
            # In Python, we can't directly get context switches
            # This is a placeholder - you might need to implement
            # platform-specific code to get real values
            return 0

# Robot Constants
ROBOT_MODE = {
    'MANUAL': 0,
    'AUTONOMOUS': 1,
    'MEASURE': 2
}

ROBOT_STATE = {
    'INITIALIZING': 0,
    'STANDBY': 1,
    'MOVING': 2,
    'SAFE_OFF': 3,
    'TEACHING': 4,
    'SAFE_STOP': 5,
    'EMERGENCY_STOP': 6,
    'HOMMING': 7,
    'RECOVERY': 8,
    'SAFE_STOP2': 9,
    'SAFE_OFF2': 10
}

# Shared state
rt_state = RT_STATE()
rt_state_lock = Lock()
