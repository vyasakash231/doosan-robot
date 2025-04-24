#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from .robot_RT_state import RT_STATE
from .robot_kinematic_model import Robot_KM

class Robot(ABC):
    n = 6  # No of joints

    # Modified-DH Parameters (same as conventional/standard DH parameters)
    alpha = np.array([0, -np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2])   
    a = np.array([0, 0, 0.409, 0, 0, 0])  # data from parameter data-sheet (in meters)
    d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])  # data from parameter data-sheet (in meters)
    d_nn = np.array([[0.0], [0.0], [0.0]])  # TCP coord in end-effector frame
    DH_params="modified"

    def __init__(self):
        self.kinematic_model = Robot_KM(self.n, self.alpha, self.a, self.d, self.d_nn, self.DH_params)

        self.is_rt_connected = False

        self.Robot_RT_State = RT_STATE()
        
        # Initialize RT control services
        self.initialize_rt_service_proxies()
        
        self.my_publisher = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)
        
        # Real-time data Publisher --> rospy.Publisher(topic_name, message_type, queue_size)
        self.speedj_publisher = rospy.Publisher('/dsr01a0509/speedj_rt_stream', SpeedJRTStream, queue_size=10)    # SpeedJRTStream -> Topic message that controls the joint velocity from an external controller.
        self.speedl_publisher = rospy.Publisher('/dsr01a0509/servoj_rt_stream', ServoJRTStream, queue_size=10)    # ServoJRTStream -> Topic message that controls the joint position from an external controller.   
        
        # For 3 kHz data, 100ms would be about 300 messages
        self.torque_publisher = rospy.Publisher('/dsr01a0509/torque_rt_stream', TorqueRTStream, queue_size=300)    # TorqueRTStream -> Topic message that controls the motor torque from an external controller.

        self.RT_observer_client = rospy.ServiceProxy('/dsr01a0509/realtime/read_data_rt', ReadDataRT)    # This function reads the real-time output data from the robot controller.

        self.read_rate = 3000  # in Hz (0.333 ms)
        self.write_rate = 1000  # in Hz (1 ms)

        self.client_thread_ = Thread(target=self.read_data_rt_client)
        self.client_thread_.daemon = True  # Make thread daemon so it exits when main thread exits
        self.client_thread_.start()

        rospy.on_shutdown(self.cleanup)

    def initialize_rt_service_proxies(self):
        try:
            service_timeout = 3.0
            services = [
                ('/dsr01a0509/system/set_robot_mode', SetRobotMode),
                ('/dsr01a0509/realtime/connect_rt_control', ConnectRTControl),   # This service connects to robot controller via Real-time External Control.
                ('/dsr01a0509/realtime/set_rt_control_input', SetRTControlInput),    # This service set the input data (external controller →  robot controller) communication configuration supported by real-time external control.
                ('/dsr01a0509/realtime/set_rt_control_output', SetRTControlOutput),    # This service set the output data (robot controller →  external controller) communication configuration supported by real-time external control.
                ('/dsr01a0509/realtime/start_rt_control', StartRTControl),    # Starts sending/receiving the set input/output data.
                ('/dsr01a0509/realtime/stop_rt_control', StopRTControl),     # Finishes sending/receiving the set input/output data.
                ('/dsr01a0509/realtime/disconnect_rt_control', DisconnectRTControl),    # This service disconnects real-time external control.
            ]

            # Wait for all services with timeout
            for service_name, _ in services:
                try:
                    rospy.wait_for_service(service_name, timeout=service_timeout)
                except rospy.ROSException as e:
                    rospy.logerr(f"Service {service_name} not available: {e}")
                    raise

            # Create service proxies
            self.set_robot_mode = rospy.ServiceProxy(services[0][0], services[0][1])
            self.connect_rt_control = rospy.ServiceProxy(services[1][0], services[1][1])
            self.set_rt_control_input = rospy.ServiceProxy(services[2][0], services[2][1])
            self.set_rt_control_output = rospy.ServiceProxy(services[3][0], services[3][1])
            self.start_rt_control = rospy.ServiceProxy(services[4][0], services[4][1])
            self.stop_rt_control = rospy.ServiceProxy(services[5][0], services[5][1])
            self.disconnect_rt_control = rospy.ServiceProxy(services[6][0], services[6][1])

            self.joint_vel_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velj_rt', SetVelJRT)   # The global joint speed is set in (deg/sec)
            self.joint_acc_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accj_rt', SetAccJRT)  # The global joint acceleration is set in (deg/sec^2)
            self.ee_vel_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velx_rt', SetVelXRT)   # This function sets the global task velocity in (mm/sec, deg/s)
            self.ee_acc_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accx_rt', SetAccXRT)

            self.connect_to_rt_control()

        except Exception as e:
            rospy.logerr(f"Failed to initialize RT control services: {e}")
            sys.exit(1)

    def connect_to_rt_control(self):
        try:
            mode_req = SetRobotModeRequest()
            mode_req.robot_mode = ROBOT_MODE_AUTONOMOUS
            robot_mode = self.set_robot_mode(mode_req)
            if not robot_mode.success:
                raise Exception("Failed to set robot mode")
            rospy.loginfo("Robot set to autonomous mode")

            connect_req = ConnectRTControlRequest()
            connect_req.ip_address = "192.168.137.100"
            connect_req.port = 12347   
            connect_response = self.connect_rt_control(connect_req)
            if not connect_response.success:
                raise Exception("Failed to connect RT control")
            
            set_output_req = SetRTControlOutputRequest()
            set_output_req.period = 0.001   # Communication Period (sec). Range: 0.001~1 [sec]
            set_output_req.loss = 4    # In succession, if the input data or the servo control command is lost due to over the set count, the real-time control connection is disconnected.
            set_output_response = self.set_rt_control_output(set_output_req)
            if not set_output_response.success:
                raise Exception("Failed to set RT control output")

            start_response = self.start_rt_control(StartRTControlRequest())
            if not start_response.success:
                raise Exception("Failed to start RT control")
                        
            self.is_rt_connected = True
            rospy.loginfo("Successfully connected to RT control")

        except Exception as e:
            rospy.logerr(f"Failed to establish RT control connection: {e}")
            self.cleanup()
            sys.exit(1)

    def cleanup(self):
        """Improved cleanup function with better error handling"""
        if self.shutdown_flag:  # Prevent multiple cleanup calls
            return
        
        self.shutdown_flag = True
        rospy.loginfo("Initiating cleanup process...")

        try:
            # Send stop command first
            stop_msg = RobotStop()
            stop_msg.stop_mode = 1  # STOP_TYPE_QUICK
            self.my_publisher.publish(stop_msg)
            rospy.sleep(0.1)  # Give time for stop command to process

            if self.is_rt_connected:
                try:
                    # Stop RT control with timeout
                    stop_future = self.stop_rt_control(StopRTControlRequest())
                    rospy.sleep(0.5)
                    
                    # Disconnect RT control
                    if not rospy.is_shutdown():  # Only try to disconnect if ROS isn't shutting down
                        self.disconnect_rt_control(DisconnectRTControlRequest())
                    
                    self.is_rt_connected = False
                    rospy.loginfo("RT control cleanup completed successfully")
                
                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn(f"Non-critical error during cleanup: {e}")
                    # Continue cleanup process despite errors
        
        except Exception as e:
            rospy.logerr(f"Critical error during cleanup: {e}")
        finally:
            rospy.loginfo("Cleanup process finished")

    def read_data_rt_client(self):
        rate = rospy.Rate(self.read_rate)  # 3000 Hz
        
        while not rospy.is_shutdown() and not self.shutdown_flag:
            try:
                if not self.is_rt_connected:
                    rate.sleep()
                    continue

                request = ReadDataRTRequest()
                response = self.RT_observer_client(request)

                # Plot Data
                self.data = response.data
                self.plot_data()
                
                # Store Real-Time data
                self.Robot_RT_State.store_data(response.data)
                 
            except (rospy.ServiceException, rospy.ROSException) as e:
                if not self.shutdown_flag:  # Only log if we're not shutting down
                    rospy.logwarn(f"Service call failed: {e}") 
            rate.sleep()

    #@abstractmethod   # Force child classes to implement this method
    def plot_data(self):
        pass  

    def _svd_solve(self, A):
        U, s, V_transp = np.linalg.svd(A)

        # Option-1
        S_inv = np.diag(s**-1)
        
        # Option-2, Handle small singular values
        # s_inv = np.zeros_like(s)
        # for i in range(len(s)):
        #     if s[i] > threshold:
        #         s_inv[i] = 1.0 / s[i]
        #     else:
        #         s_inv[i] = 0.0  # Or apply damping: s[i]/(s[i]^2 + lambda^2)
        
        # # Reconstruct inverse
        # S_inv = np.zeros_like(M)
        # for i in range(len(s)):
        #     S_inv[i,i] = s_inv[i]

        # A^-1 = V * S^-1 * U^T
        A_inv = V_transp.T @ S_inv @ U.T   # V = V_transp.T
        return A_inv

    def euler2mat(self, euler_angles):  # euler_angles in degrees
        """
        Convert Euler ZYZ rotation angles to a 3D rotation matrix.
        
        Args:
        z1_angle (float): First rotation angle around Z-axis in radians
        y_angle (float): Rotation angle around Y-axis in radians
        z2_angle (float): Second rotation angle around Z-axis in radians
        
        Returns:
        numpy.ndarray: 3x3 rotation matrix
        """
        z1_angle, y_angle, z2_angle = np.radians(euler_angles)

        # Rotation matrices for individual axes
        Rz1 = np.array([
            [math.cos(z1_angle), -math.sin(z1_angle), 0],
            [math.sin(z1_angle), math.cos(z1_angle), 0],
            [0, 0, 1]
        ])
        
        Ry = np.array([
            [math.cos(y_angle), 0, math.sin(y_angle)],
            [0, 1, 0],
            [-math.sin(y_angle), 0, math.cos(y_angle)]
        ])
        
        Rz2 = np.array([
            [math.cos(z2_angle), -math.sin(z2_angle), 0],
            [math.sin(z2_angle), math.cos(z2_angle), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations in ZYZ order
        """
        * The rotation order (Z1 * Y * Z2) is typically referred to as the "intrinsic" ZYZ rotation sequence
        * The rotation order (Z2 * Y * Z1) is typically referred to as the "extrinsic" ZYZ rotation sequence

        The key difference is that intrinsic rotations are performed relative to the object's current orientation, 
        while extrinsic rotations are performed relative to the fixed global coordinate system.
        """
        R = Rz1 @ Ry @ Rz2
        return R
    
    def mat2quat(self, rmat):
        M = np.asarray(rmat).astype(np.float32)[:3, :3]

        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([
                    [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                    [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                    [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                    [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
                    ])
        K /= 3.0

        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
        if q1[0] < 0.0:
            np.negative(q1, q1)
        inds = np.array([1, 2, 3, 0])
        return q1[inds]
    
    def eul2quat(self, euler_angles):
        rmat = self.euler2mat(euler_angles)
        M = np.asarray(rmat).astype(np.float32)
        q = self.mat2quat(M)
        return q
    
    def unit_vector(self, data, axis=None, out=None):
        """
        Returns ndarray normalized by length, i.e. eucledian norm, along axis.

        Args:
            data (np.array): data to normalize
            axis (None or int): If specified, determines specific axis along data to normalize
            out (None or np.array): If specified, will store computation in this variable

        Returns:
            None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
        """
        if out is None:
            data = np.array(data, dtype=np.float32, copy=True)
            if data.ndim == 1:
                data /= sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data * data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data
    
    def quat_slerp(self, quat0, quat1, fraction, shortestpath=True):
        """
        Return spherical linear interpolation between two quaternions.

        Args:
            quat0 (np.array): (x,y,z,w) quaternion startpoint
            quat1 (np.array): (x,y,z,w) quaternion endpoint
            fraction (float): fraction of interpolation to calculate
            shortestpath (bool): If True, will calculate the shortest path

        Returns:
            np.array: (x,y,z,w) quaternion distance
        """
        EPS = np.finfo(float).eps * 4.0
        
        q0 = unit_vector(quat0[:4])
        q1 = unit_vector(quat1[:4])
        
        if fraction == 0.0:
            return q0
        elif fraction == 1.0:
            return q1
        
        d = np.dot(q0, q1)
        
        if abs(abs(d) - 1.0) < EPS:
            return q0
        
        if shortestpath and d < 0.0:
            # invert rotation
            d = -d
            q1 *= -1.0
        angle = acos(np.clip(d, -1, 1))
        
        if abs(angle) < EPS:
            return q0
        
        isin = 1.0 / sin(angle)
        q0 *= sin((1.0 - fraction) * angle) * isin
        q1 *= sin(fraction * angle) * isin
        q0 += q1
        return q0
        
