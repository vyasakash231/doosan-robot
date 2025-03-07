import numpy as np

# Define RT_STATE class first
class RT_STATE:
    def __init__(self):
        self.time_stamp = 0.0
        self.actual_joint_position = np.zeros(6)
        self.actual_joint_velocity = np.zeros(6)
        self.actual_joint_position_abs = np.zeros(6)
        self.actual_joint_velocity_abs = np.zeros(6)
        self.actual_tcp_position = np.zeros(6)   # (Tool Center Point) is the specific point on the attached tool that is used to define the precise location where the robot should interact with an object
        self.actual_tcp_velocity = np.zeros(6)
        self.actual_flange_position = np.zeros(6)   # refers to the location of the mounting plate at the end of the robot's wrist, where tools are attached
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
        self.goal_joint_position = np.zeros(6)
        self.goal_tcp_position = np.zeros(6)
        self.coriolis_matrix = np.zeros((6, 6))
        self.mass_matrix = np.zeros((6, 6))
        self.jacobian_matrix = np.zeros((6, 6))

    def store_data(self, data):
        for i in range(6):
            self.actual_joint_position[i] = data.actual_joint_position[i]
            self.actual_joint_velocity[i] = data.actual_joint_velocity[i]
            self.actual_joint_position_abs[i] = data.actual_joint_position_abs[i]
            self.actual_joint_velocity_abs[i] = data.actual_joint_velocity_abs[i]
            self.actual_tcp_position[i] = data.actual_tcp_position[i]   #  (X, Y, Z, A, B, C), where (A, B, C) follows Euler ZYZ notation [mm, deg]
            self.actual_tcp_velocity[i] = data.actual_tcp_velocity[i]
            self.actual_flange_position[i] = data.actual_flange_position[i]   #  (x, y, z)
            self.actual_flange_velocity[i] = data.actual_flange_velocity[i]
            self.actual_motor_torque[i] = data.actual_motor_torque[i]
            self.actual_joint_torque[i] = data.actual_joint_torque[i]
            self.raw_joint_torque[i] = data.raw_joint_torque[i]
            self.raw_force_torque[i] = data.raw_force_torque[i]
            self.external_joint_torque[i] = data.external_joint_torque[i]
            self.external_tcp_force[i] = data.external_tcp_force[i]
            # self.target_joint_position[i] = data.target_joint_position[i]
            # self.target_joint_velocity[i] = data.target_joint_velocity[i]
            # self.target_joint_acceleration[i] = data.target_joint_acceleration[i]
            # self.target_motor_torque[i] = data.target_motor_torque[i]
            # self.target_tcp_position[i] = data.target_tcp_position[i]
            # self.target_tcp_velocity[i] = data.target_tcp_velocity[i]
            self.gravity_torque[i] = data.gravity_torque[i]
            # self.goal_joint_position[i] = data.goal_joint_position[i]
            # self.goal_tcp_position[i] = data.goal_tcp_position[i]

            for j in range(6):
                self.coriolis_matrix[i,j] = data.coriolis_matrix[i].data[j]
                self.mass_matrix[i,j] = data.mass_matrix[i].data[j]
                self.jacobian_matrix[i,j] = data.jacobian_matrix[i].data[j]
