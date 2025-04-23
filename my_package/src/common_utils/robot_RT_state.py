import numpy as np

# Define RT_STATE class first
class RT_STATE:
    def __init__(self):
        self.time_stamp = 0.0
        self.actual_joint_position = np.zeros(6)  # actual joint position from incremental encoder at motor side(used for control) [deg]
        self.actual_joint_velocity = np.zeros(6)  # actual joint velocity from incremental encoder at motor side [deg/s]
        self.actual_joint_position_abs = np.zeros(6)  # actual joint position from absolute encoder at link side (used for exact link position) [deg]
        self.actual_joint_velocity_abs = np.zeros(6)  # actual joint velocity from absolute encoder at link side [deg/s]
        self.actual_tcp_position = np.zeros(6)   # (Tool Center Point) actual robot tcp position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
        self.actual_tcp_velocity = np.zeros(6)   # actual robot tcp velocity w.r.t. base coordinates [mm, deg/s]
        self.actual_tcp_position = np.zeros(6)   # refers to the location of the mounting plate at the end of the robot's wrist, where tools are attached
        self.actual_flange_velocity = np.zeros(6)  
        self.actual_motor_torque = np.zeros(6)   # actual motor torque applying gear ratio = gear_ratio * current2torque_constant * motor current [Nm]
        self.actual_joint_torque = np.zeros(6)   # estimated joint torque by robot controller [Nm]
        self.raw_joint_torque = np.zeros(6)   # calibrated joint torque sensor data [Nm]
        self.raw_force_torque = np.zeros(6)   # calibrated force torque sensor data w.r.t. flange coordinates [N, Nm]
        self.external_joint_torque = np.zeros(6)   # estimated external joint torque [Nm]
        self.external_tcp_force = np.zeros(6)    # estimated tcp force w.r.t. base coordinates [N, Nm]
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

    # def store_data(self, data):
    #     for i in range(6):
    #         self.actual_joint_position[i] = data.actual_joint_position[i]
    #         self.actual_joint_velocity[i] = data.actual_joint_velocity[i]
    #         self.actual_joint_position_abs[i] = data.actual_joint_position_abs[i]
    #         self.actual_joint_velocity_abs[i] = data.actual_joint_velocity_abs[i]
    #         self.actual_tcp_position[i] = data.actual_tcp_position[i] 
    #         self.actual_tcp_velocity[i] = data.actual_tcp_velocity[i]
    #         # self.actual_flange_position[i] = data.actual_flange_position[i] 
    #         # self.actual_flange_velocity[i] = data.actual_flange_velocity[i]
    #         self.actual_motor_torque[i] = data.actual_motor_torque[i]
    #         self.actual_joint_torque[i] = data.actual_joint_torque[i]
    #         self.raw_joint_torque[i] = data.raw_joint_torque[i]
    #         self.raw_force_torque[i] = data.raw_force_torque[i]
    #         self.external_joint_torque[i] = data.external_joint_torque[i]
    #         self.external_tcp_force[i] = data.external_tcp_force[i]
    #         # self.target_joint_position[i] = data.target_joint_position[i]
    #         # self.target_joint_velocity[i] = data.target_joint_velocity[i]
    #         # self.target_joint_acceleration[i] = data.target_joint_acceleration[i]
    #         # self.target_motor_torque[i] = data.target_motor_torque[i]
    #         # self.target_tcp_position[i] = data.target_tcp_position[i]
    #         # self.target_tcp_velocity[i] = data.target_tcp_velocity[i]
    #         self.gravity_torque[i] = data.gravity_torque[i]
    #         # self.goal_joint_position[i] = data.goal_joint_position[i]
    #         # self.goal_tcp_position[i] = data.goal_tcp_position[i]

    #         for j in range(6):
    #             self.coriolis_matrix[i,j] = data.coriolis_matrix[i].data[j]
    #             self.mass_matrix[i,j] = data.mass_matrix[i].data[j]
    #             self.jacobian_matrix[i,j] = data.jacobian_matrix[i].data[j]

    def store_data(self, data):
        self.actual_joint_position = np.array(data.actual_joint_position)
        self.actual_joint_velocity = np.array(data.actual_joint_velocity)
        self.actual_joint_position_abs = np.array(data.actual_joint_position_abs)
        self.actual_joint_velocity_abs = np.array(data.actual_joint_velocity_abs)
        self.actual_tcp_position = np.array(data.actual_tcp_position)
        self.actual_tcp_velocity = np.array(data.actual_tcp_velocity)
        self.actual_flange_position = np.array(data.actual_flange_position)
        self.actual_flange_velocity = np.array(data.actual_flange_velocity)
        self.actual_motor_torque = np.array(data.actual_motor_torque)
        self.actual_joint_torque = np.array(data.actual_joint_torque)
        self.raw_joint_torque = np.array(data.raw_joint_torque)
        self.raw_force_torque = np.array(data.raw_force_torque)
        self.external_joint_torque = np.array(data.external_joint_torque)
        self.external_tcp_force = np.array(data.external_tcp_force)
        self.gravity_torque = np.array(data.gravity_torque)

        for i in range(6):
            self.coriolis_matrix[i, :] = data.coriolis_matrix[i].data
            self.mass_matrix[i, :] = data.mass_matrix[i].data
            self.jacobian_matrix[i, :] = data.jacobian_matrix[i].data

