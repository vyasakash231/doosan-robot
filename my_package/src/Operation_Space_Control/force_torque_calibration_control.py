#!/usr/bin/env python3

import rospy
import PyKDL
import numpy as np
import yaml
import os
import math
from std_srvs.srv import Empty, EmptyResponse
from geometry_msgs.msg import WrenchStamped, Vector3Stamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf_conversions.posemath as pm
import angles

# Constants
G_FORCE = 9.80665

class FTCalib:
    """
    Force-Torque Calibration class that matches the C++ Calibration::FTCalib.
    This class handles the calibration routine for a Force-Torque sensor.
    """
    def __init__(self):
        self.measurements = []
        self.A = np.zeros((0, 10))
        self.b = np.zeros((0, 1))
        
    def addMeasurement(self, gravity_msg, ft_avg_msg):
        """Add a measurement from the F/T sensor with corresponding gravity vector"""
        # Extract gravity vector
        gx = gravity_msg.vector.x
        gy = gravity_msg.vector.y
        gz = gravity_msg.vector.z
        
        # Extract force-torque measurements
        fx = ft_avg_msg.wrench.force.x
        fy = ft_avg_msg.wrench.force.y
        fz = ft_avg_msg.wrench.force.z
        tx = ft_avg_msg.wrench.torque.x
        ty = ft_avg_msg.wrench.torque.y
        tz = ft_avg_msg.wrench.torque.z
        
        # Store the measurement
        measurement = {'gravity': [gx, gy, gz], 'wrench': [fx, fy, fz, tx, ty, tz]}
        self.measurements.append(measurement)
        
        # Update the calibration matrices
        A_row_f = np.array([
            [gx, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [gy, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [gz, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])
        
        A_row_t = np.array([
            [0, gz, -gy, 0, 0, 0, 0, 1, 0, 0],
            [0, -gz, 0, gx, 0, 0, 0, 0, 1, 0],
            [0, gy, -gx, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        A_new = np.vstack([A_row_f, A_row_t])
        b_new = np.array([[fx], [fy], [fz], [tx], [ty], [tz]])
        
        # Update the linear system
        if self.A.shape[0] == 0:
            self.A = A_new
            self.b = b_new
        else:
            self.A = np.vstack([self.A, A_new])
            self.b = np.vstack([self.b, b_new])
    
    def getCalib(self):
        """Get the current calibration estimate"""
        if len(self.measurements) < 3:
            rospy.logwarn("Not enough measurements for calibration (need at least 3)")
            return np.zeros(10)
        
        # Solve the linear system Ax = b using least squares
        x, residuals, rank, s = np.linalg.lstsq(self.A, self.b, rcond=None)
        return x.flatten()


class FtSensorCalibController:
    """
    Controller for calibrating a force-torque sensor, including offset and gravity compensation.
    """
    def __init__(self):
        # Initialize variables
        self.ft_wrench_raw = PyKDL.Wrench()
        self.offset_kdl = PyKDL.Wrench()
        self.p_sensor_tool_com_kdl = PyKDL.Vector()
        self.base_tool_weight_com = PyKDL.Wrench()
        self.tool_mass = 0.0
        self.ft_offset_force = PyKDL.Vector()
        self.ft_offset_torque = PyKDL.Vector()
        self.p_sensor_tool_com = PyKDL.Vector()
        
        self.ft_calib_poses = []
        self.ft_calib_q = []
        self.ft_calib_q_home = []
        self.number_of_poses = 0
        self.pose_counter = 0
        self.do_compensation = False
        
        self.joint_msr_states = None
        self.joint_handles = []
        self.kdl_chain = None
        self.ik_solver = None
        self.fk_solver = None
        self.ft_calib = None
        
        self.last_publish_time = rospy.Time()
        self.publish_rate = 10  # Default rate, will be updated from parameter
        self.calibration_loop_rate = 10  # Default rate, will be updated from parameter
        self.p2p_traj_duration = 1.0  # Default duration, will be updated from parameter

    def init(self, robot, nh):
        """Initialize the controller"""
        self.kdl_chain = self.create_kdl_chain()  
        self.joint_msr_states = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())
        
        # Subscribe to raw FT sensor topic
        ft_sensor_topic_name = nh.get_param("topic_name")
        self.sub_ft_sensor = rospy.Subscriber(ft_sensor_topic_name, WrenchStamped, self.ft_raw_topic_callback)
        
        # Get parameters
        self.calibration_loop_rate = nh.get_param("calibration_loop_rate")
        self.publish_rate = nh.get_param("publish_rate")
        self.p2p_traj_duration = nh.get_param("p2p_traj_duration")
        
        # Initialize solvers
        self.ik_solver = PyKDL.ChainIkSolverPos_LMA(self.kdl_chain)
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.kdl_chain)
        
        # Initialize FT calibration
        self.ft_calib = FTCalib()
        
        # Publishers
        self.pub_joint_traj_ctl = rospy.Publisher("/lwr/joint_trajectory_controller/command", JointTrajectory, queue_size=1)
        self.pub_ft_sensor_no_offset = rospy.Publisher("ft_sensor_no_offset", WrenchStamped, queue_size=1)
        self.pub_ft_sensor_no_gravity = rospy.Publisher("ft_sensor_no_gravity", WrenchStamped, queue_size=1)
        
        # Service servers
        self.move_next_calib_pose_service = rospy.Service("move_next_calib_pose", Empty, self.move_next_calib_pose)
        self.move_home_pose_service = rospy.Service("move_home_pose", Empty, self.move_home_pose)
        self.save_calib_data_service = rospy.Service("save_calib_data", Empty, self.save_calib_data)
        self.start_compensation_service = rospy.Service("start_compensation", Empty, self.start_compensation)
        self.do_estimation_step_service = rospy.Service("do_estimation_step", Empty, self.do_estimation_step)
        self.start_autonomus_estimation_service = rospy.Service("start_autonomus_estimation", Empty, self.start_autonomus_estimation)
        
        # Load calibration poses
        self.get_calibration_q(nh)
        
        # Reset pose counter
        self.pose_counter = 0
        
        # Check if existing calibration data should be used
        recover = nh.get_param("recover_existing_data", False)
        if recover:
            self.recover_existing_data()
        
        return True
    
    def create_kdl_chain(self):
        """Create the kinematic chain for the robot"""
        # This is a placeholder - in a real implementation, you would
        # create the KDL chain from URDF or parameters
        chain = PyKDL.Chain()
        # Add segments to the chain based on your robot
        # ...
        return chain

    def starting(self, time):
        """Called when the controller is starting"""
        self.last_publish_time = time
    
    def update(self, time, period):
        """Controller update function"""
        if not self.do_compensation:
            return
        
        # Do offset compensation
        ft_wrench_no_offset = self.ft_wrench_raw - self.offset_kdl
        
        # Get the current robot configuration
        for i in range(self.kdl_chain.getNrOfJoints()):
            self.joint_msr_states[i] = self.joint_handles[i].getPosition()
        
        # Evaluate forward kinematics
        fk_frame = PyKDL.Frame()
        self.fk_solver.JntToCart(self.joint_msr_states, fk_frame)
        R_ft_base = fk_frame.M.Inverse()
        
        # Compensate for tool weight
        # Move the reference point of the weight from the COM of the tool to the wrist
        gravity_transformation = PyKDL.Frame(fk_frame.M.Inverse(), self.p_sensor_tool_com_kdl)
        
        # Compensate for the weight of the tool
        ft_wrench_no_gravity = ft_wrench_no_offset - gravity_transformation * self.base_tool_weight_com
        
        # Publish data at the specified rate
        if time > self.last_publish_time + rospy.Duration(1.0 / self.publish_rate):
            self.last_publish_time += rospy.Duration(1.0 / self.publish_rate)
            
            self.publish_data(ft_wrench_no_offset, self.pub_ft_sensor_no_offset)
            self.publish_data(ft_wrench_no_gravity, self.pub_ft_sensor_no_gravity)
    
    def ft_raw_topic_callback(self, msg):
        """Callback for the raw FT sensor data"""
        self.ft_wrench_raw = pm.fromMsg(msg.wrench)
    
    def get_calibration_q(self, nh):
        """Get calibration joint configurations from parameters"""
        self.number_of_poses = nh.get_param("calib_number_q")
        
        for i in range(self.number_of_poses):
            q = nh.get_param(f"calib_q/q{i}")
            self.ft_calib_q.append(q)
        
        self.ft_calib_q_home = nh.get_param("home_q")
    
    def send_joint_trajectory_msg(self, q_des):
        """Send a joint trajectory message to the controller"""
        traj_msg = JointTrajectory()
        point = JointTrajectoryPoint()
        
        # Set joint names
        traj_msg.joint_names = [
            "lwr_a1_joint",
            "lwr_a2_joint",
            "lwr_e1_joint",
            "lwr_a3_joint",
            "lwr_a4_joint",
            "lwr_a5_joint",
            "lwr_a6_joint"
        ]
        
        # Set positions
        point.positions = [q_des[i] for i in range(self.kdl_chain.getNrOfJoints())]
        point.time_from_start = rospy.Duration(self.p2p_traj_duration)
        
        traj_msg.points.append(point)
        self.pub_joint_traj_ctl.publish(traj_msg)
    
    def move_home_pose(self, req):
        """Move the robot to the home pose"""
        self.send_joint_trajectory_msg(self.ft_calib_q_home)
        return EmptyResponse()
    
    def move_next_calib_pose(self, req):
        """Move the robot to the next calibration pose"""
        if self.number_of_poses <= self.pose_counter:
            print(f"Manual pose {self.pose_counter}")
            return EmptyResponse()
        
        print(f"Sending pose {self.pose_counter} to joint_trajectory_controller")
        self.send_joint_trajectory_msg(self.ft_calib_q[self.pose_counter])
        
        return EmptyResponse()
    
    def add_measurement(self, gravity_ft, ft_raw_avg):
        """Add a measurement to the calibration system"""
        # Transform data for use in the FTCalib class
        frame_id = "ft_frame"
        
        gravity_msg = Vector3Stamped()
        gravity_msg.header.frame_id = frame_id
        gravity_msg.vector.x = gravity_ft.x()
        gravity_msg.vector.y = gravity_ft.y()
        gravity_msg.vector.z = gravity_ft.z()
        
        ft_avg_msg = WrenchStamped()
        ft_avg_msg.header.frame_id = frame_id
        ft_avg_msg.wrench.force.x = ft_raw_avg.force.x()
        ft_avg_msg.wrench.force.y = ft_raw_avg.force.y()
        ft_avg_msg.wrench.force.z = ft_raw_avg.force.z()
        ft_avg_msg.wrench.torque.x = ft_raw_avg.torque.x()
        ft_avg_msg.wrench.torque.y = ft_raw_avg.torque.y()
        ft_avg_msg.wrench.torque.z = ft_raw_avg.torque.z()
        
        # Add measurement
        self.ft_calib.addMeasurement(gravity_msg, ft_avg_msg)
    
    def start_autonomus_estimation(self, req):
        """Start the autonomous estimation process"""
        wait = rospy.Duration(self.p2p_traj_duration + 2)
        
        for i in range(self.number_of_poses):
            # Move robot
            print(f"Sending pose {i} to joint_trajectory_controller")
            self.send_joint_trajectory_msg(self.ft_calib_q[i])
            
            # Wait for trajectory end
            rospy.sleep(wait)
            
            # Do a step of the estimation process
            self.estimation_step()
            print(f"estimation n.: {i}")
            
            # Update counter
            self.pose_counter = i
        
        # Get current estimation
        ft_calib = self.ft_calib.getCalib()
        
        self.tool_mass = ft_calib[0]
        
        # Calculate the center of mass position
        self.p_sensor_tool_com.x(ft_calib[1] / self.tool_mass)
        self.p_sensor_tool_com.y(ft_calib[2] / self.tool_mass)
        self.p_sensor_tool_com.z(ft_calib[3] / self.tool_mass)
        
        # Calculate force offsets
        self.ft_offset_force.x(-ft_calib[4])
        self.ft_offset_force.y(-ft_calib[5])
        self.ft_offset_force.z(-ft_calib[6])
        
        # Calculate torque offsets
        self.ft_offset_torque.x(-ft_calib[7])
        self.ft_offset_torque.y(-ft_calib[8])
        self.ft_offset_torque.z(-ft_calib[9])
        
        # Print the current estimation
        print("-------------------------------------------------------------")
        print("Current calibration estimate:")
        print("\n\n")
        
        print(f"Mass: {self.tool_mass}\n\n")
        
        print("Tool CoM (in ft sensor frame):")
        print(f"[{self.p_sensor_tool_com.x()}, {self.p_sensor_tool_com.y()}, {self.p_sensor_tool_com.z()}]")
        print("\n\n")
        
        print("FT offset:")
        print(f"[{self.ft_offset_force.x()}, {self.ft_offset_force.y()}, {self.ft_offset_force.z()}, "
              f"{self.ft_offset_torque.x()}, {self.ft_offset_torque.y()}, {self.ft_offset_torque.z()}]")
        print("\n\n")
        print("-------------------------------------------------------------\n\n\n")
        
        return EmptyResponse()
    
    def estimation_step(self):
        """Perform a single step of the estimation process"""
        loop_rate = rospy.Rate(self.calibration_loop_rate)
        number_measurements = 100
        ft_raw_avg = PyKDL.Wrench()
        
        # Average 101 measurements from the FT sensor
        ft_raw_avg = self.ft_wrench_raw
        for i in range(number_measurements):
            ft_raw_avg = ft_raw_avg + self.ft_wrench_raw
            loop_rate.sleep()
        
        ft_raw_avg = -ft_raw_avg / float(number_measurements + 1)
        
        # Get the current robot configuration
        for i in range(self.kdl_chain.getNrOfJoints()):
            self.joint_msr_states[i] = self.joint_handles[i].getPosition()
        
        # Evaluate forward kinematics
        fk_frame = PyKDL.Frame()
        self.fk_solver.JntToCart(self.joint_msr_states, fk_frame)
        R_ft_base = fk_frame.M.Inverse()
        
        # Evaluate the gravity as if it was measured by an IMU
        # whose reference frame is aligned with vito_anchor
        gravity = PyKDL.Vector(0, 0, -G_FORCE)
        
        # Rotate the gravity in FT frame
        gravity_ft = R_ft_base * gravity
        
        # Add measurement to the filter
        self.add_measurement(gravity_ft, ft_raw_avg)
        
        # Save data to allow data recovery if needed
        self.save_calib_meas(gravity_ft, ft_raw_avg, self.pose_counter, self.joint_msr_states)
    
    def do_estimation_step(self, req):
        """Perform a step of the estimation process and update pose counter"""
        # Do a step of the estimation process
        self.estimation_step()
        
        # Update pose counter
        self.pose_counter += 1
        
        # Get current estimation
        ft_calib = self.ft_calib.getCalib()
        
        self.tool_mass = ft_calib[0]
        
        # Calculate the center of mass position
        self.p_sensor_tool_com.x(ft_calib[1] / self.tool_mass)
        self.p_sensor_tool_com.y(ft_calib[2] / self.tool_mass)
        self.p_sensor_tool_com.z(ft_calib[3] / self.tool_mass)
        
        # Calculate force offsets
        self.ft_offset_force.x(-ft_calib[4])
        self.ft_offset_force.y(-ft_calib[5])
        self.ft_offset_force.z(-ft_calib[6])
        
        # Calculate torque offsets
        self.ft_offset_torque.x(-ft_calib[7])
        self.ft_offset_torque.y(-ft_calib[8])
        self.ft_offset_torque.z(-ft_calib[9])
        
        # Print the current estimation
        print("-------------------------------------------------------------")
        print("Current calibration estimate:")
        print("\n\n")
        
        print(f"Mass: {self.tool_mass}\n\n")
        
        print("Tool CoM (in ft sensor frame):")
        print(f"[{self.p_sensor_tool_com.x()}, {self.p_sensor_tool_com.y()}, {self.p_sensor_tool_com.z()}]")
        print("\n\n")
        
        print("FT offset:")
        print(f"[{self.ft_offset_force.x()}, {self.ft_offset_force.y()}, {self.ft_offset_force.z()}, "
              f"{self.ft_offset_torque.x()}, {self.ft_offset_torque.y()}, {self.ft_offset_torque.z()}]")
        print("\n\n")
        print("-------------------------------------------------------------\n\n\n")
        
        return EmptyResponse()
    
    def recover_existing_data(self):
        """Recover existing calibration data from a YAML file"""
        pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_name = os.path.join(pkg_path, "config/ft_calib_meas.yaml")
        
        with open(file_name, 'r') as f:
            ft_data_yaml = yaml.safe_load(f)
        
        number = ft_data_yaml["calib_meas_number"]
        
        gravity_ft = PyKDL.Vector()
        ft_wrench_avg = PyKDL.Wrench()
        
        for i in range(number):
            # Load data from YAML
            gravity_vec = ft_data_yaml["calib_meas"][f"pose{i}"]["gravity"]
            ft_force_avg = ft_data_yaml["calib_meas"][f"pose{i}"]["force_avg"]
            ft_torque_avg = ft_data_yaml["calib_meas"][f"pose{i}"]["torque_avg"]
            
            # Convert to KDL types
            for j in range(3):
                gravity_ft[j] = gravity_vec[j]
                ft_wrench_avg.force[j] = ft_force_avg[j]
                ft_wrench_avg.torque[j] = ft_torque_avg[j]
            
            # Add measurement
            self.add_measurement(gravity_ft, ft_wrench_avg)
            self.pose_counter += 1
        
        print(f"Recovered calibration data for {number} poses")
    
    def save_calib_meas(self, gravity, ft_wrench_avg, index, q_kdl):
        """Save calibration measurement to a YAML file"""
        pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_name = os.path.join(pkg_path, "config/ft_calib_meas.yaml")
        
        with open(file_name, 'r') as f:
            ft_data_yaml = yaml.safe_load(f)
        
        # Convert KDL types to Python lists
        gravity_vec = [gravity[i] for i in range(3)]
        ft_force_avg = [ft_wrench_avg.force[i] for i in range(3)]
        ft_torque_avg = [ft_wrench_avg.torque[i] for i in range(3)]
        q = [q_kdl[i] for i in range(7)]
        
        # Update YAML data
        ft_data_yaml["calib_meas_number"] = index + 1
        
        if "calib_meas" not in ft_data_yaml:
            ft_data_yaml["calib_meas"] = {}
        
        ft_data_yaml["calib_meas"][f"pose{index}"] = {"gravity": gravity_vec, "force_avg": ft_force_avg, "torque_avg": ft_torque_avg, "q": q}
        
        # Write YAML file
        with open(file_name, 'w') as f:
            yaml.dump(ft_data_yaml, f)
    
    def save_calib_data(self, req):
        """Save calibration data to a YAML file"""
        pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_name = os.path.join(pkg_path, "config/ft_calib_data.yaml")
        
        with open(file_name, 'r') as f:
            ft_data_yaml = yaml.safe_load(f)
        
        # Convert data to standard types
        frame_id = "lwr_7_link"
        ft_offset = [0] * 6
        p_sensor_tool_com = [0] * 6
        
        for i in range(3):
            ft_offset[i] = self.ft_offset_force[i]
            ft_offset[i + 3] = self.ft_offset_torque[i]
            p_sensor_tool_com[i] = self.p_sensor_tool_com[i]
            p_sensor_tool_com[i + 3] = 0
        
        # Populate YAML using field names
        # required by the package ros-indigo-gravity-compensation
        ft_data_yaml["bias"] = ft_offset
        ft_data_yaml["gripper_com_frame_id"] = frame_id
        ft_data_yaml["gripper_com_pose"] = p_sensor_tool_com
        ft_data_yaml["gripper_mass"] = self.tool_mass
        
        # Save YAML file
        with open(file_name, 'w') as f:
            yaml.dump(ft_data_yaml, f)
        
        return EmptyResponse()
    
    def load_calib_data(self):
        """Load calibration data from a YAML file"""
        pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_name = os.path.join(pkg_path, "config/ft_calib_data.yaml")
        
        with open(file_name, 'r') as f:
            ft_data_yaml = yaml.safe_load(f)
        
        # Get data from the YAML file
        self.tool_mass = ft_data_yaml["gripper_mass"]
        p_sensor_tool_com = ft_data_yaml["gripper_com_pose"]
        offset = ft_data_yaml["bias"]
        
        # Evaluate end-effector weight
        self.base_tool_weight_com = PyKDL.Wrench(PyKDL.Vector(0, 0, -self.tool_mass * G_FORCE), PyKDL.Vector(0, 0, 0))
        
        # Transform to KDL
        for i in range(3):
            self.p_sensor_tool_com_kdl[i] = p_sensor_tool_com[i]
            self.offset_kdl.force[i] = offset[i]
            self.offset_kdl.torque[i] = offset[i + 3]
        
        return True
    
    def start_compensation(self, req):
        """Start force-torque compensation"""
        # Load calibration data from YAML file
        self.load_calib_data()
        
        # Enable offset and gravity compensation
        self.do_compensation = True
        
        return EmptyResponse()
    
    def publish_data(self, wrench, pub):
        """Publish force-torque data"""
        # Create the message
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = rospy.Time.now()
        wrench_msg.wrench.force.x = wrench.force.x()
        wrench_msg.wrench.force.y = wrench.force.y()
        wrench_msg.wrench.force.z = wrench.force.z()
        wrench_msg.wrench.torque.x = wrench.torque.x()
        wrench_msg.wrench.torque.y = wrench.torque.y()
        wrench_msg.wrench.torque.z = wrench.torque.z()
        
        # Publish the message
        pub.publish(wrench_msg)


# Main function for running the node
if __name__ == "__main__":
    rospy.init_node("ft_sensor_calib_controller")
    
    # Create and initialize controller
    controller = FtSensorCalibController()
    
    # Get node handle for parameters
    nh = rospy.NodeHandle("~")
    
    # Initialize the controller
    robot = None  # Would be passed from the hardware interface in C++
    controller.init(robot, nh)
    
    # Spin to keep the node alive
    rospy.spin()