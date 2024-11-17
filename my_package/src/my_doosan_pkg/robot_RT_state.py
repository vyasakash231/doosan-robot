import numpy as np

# Define RT_STATE class first
class RT_STATE:
    def __init__(self):
        self.time_stamp = 0.0
        self.actual_joint_position = np.zeros(6)
        self.actual_joint_velocity = np.zeros(6)
        self.actual_joint_position_abs = np.zeros(6)
        self.actual_joint_velocity_abs = np.zeros(6)
        self.actual_tcp_position = np.zeros(6)
        self.actual_tcp_velocity = np.zeros(6)
        self.actual_flange_position = np.zeros(6)
        self.actual_flange_velocity = np.zeros(6)
        self.actual_motor_torque = np.zeros(6)
        self.actual_joint_torque = np.zeros(6)
        self.raw_joint_torque = np.zeros(6)
        self.raw_force_torque = np.zeros(6)
        self.external_joint_torque = np.zeros(6)
        self.external_tcp_force = np.zeros(6)
        self.target_joint_position = np.zeros(6)
        self.target_joint_velocity = np.zeros(6)
        self.target_joint_acceleration = np.zeros(6)
        self.target_motor_torque = np.zeros(6)
        self.target_tcp_position = np.zeros(6)
        self.target_tcp_velocity = np.zeros(6)
        self.gravity_torque = np.zeros(6)
        self.joint_temperature = np.zeros(6)
        self.goal_joint_position = np.zeros(6)
        self.goal_tcp_position = np.zeros(6)
        self.coriolis_matrix = np.zeros((6, 6))
        self.mass_matrix = np.zeros((6, 6))
        self.jacobian_matrix = np.zeros((6, 6))
        self.control_mode = 0.0