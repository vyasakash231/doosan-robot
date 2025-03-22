#!/usr/bin/env python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../common/imp"))) # get import path : DSR_ROBOT.py 
from basic_import import *

from scipy.spatial.transform import Rotation 
from scipy.linalg import expm

from camera import Camera
# from common import transformation_matrix, weight_Func   

np.set_printoptions(suppress=True)  # to aviod scientific notation while printing numpy array

class Visual_Servoing:
    def __init__(self,camera_view):
        # Initiate/Creates a node with name 'my_turtlebot'
        rospy.init_node('visual_servoing',anonymous = False)
        
        rospy.wait_for_service('/dsr01a0509/motion/move_joint')  # Wait until the service becomes available
        self.move_joint = rospy.ServiceProxy('/dsr01a0509/motion/move_joint', MoveJoint)

        self.my_subscriber = rospy.Subscriber('/tf', TFMessage, self.call_back_func)  # In degrees

        self.T_e_0 = np.eye(4)
        self.T_matrix = np.zeros((6,4,4))

        self.rate = rospy.Rate(10)
        
        self.camera_view = camera_view

        # Create threads for camera task and robot movement task
        self.camera_thread_1 = threading.Thread(target=self.camera_task_1)
        self.camera_thread_2 = threading.Thread(target=self.camera_task_2)

        # Start the threads
        self.camera_thread_1.start()
        self.camera_thread_2.start()

    def twist_to_homogeneous(self,v):
        # Extract translational and rotational components from the twist vector
        v_trans = v[:3,-1]  # Translational velocities (vx, vy, vz)
        v_rot = v[3:,-1]    # Rotational velocities (wx, wy, wz)

        # Construct the skew-symmetric matrix representation of the twist vector
        v_skew = np.array([[0, -v_rot[2], v_rot[1], v_trans[0]],
                           [v_rot[2], 0, -v_rot[0], v_trans[1]],
                           [-v_rot[1], v_rot[0], 0, v_trans[2]],
                           [0, 0, 0, 0]])

        # Compute the exponential of the skew-symmetric matrix
        R = expm(v_skew)
        return R
    
    def homogeneous_to_twist(self,T):
        # Extract rotation matrix and translation vector
        R = T[:3, :3]
        p = T[:3, -1]
        
        # Compute angular velocity vector
        theta = np.arccos((np.trace(R) - 1) / 2)
        omega_skew = (R - R.T) / (2 * np.sin(theta))
        omega = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
        
        # Compute linear velocity vector
        v = p
        
        # Combine angular and linear velocities into twist vector
        xi = np.concatenate((omega, v))
        return xi

    def call_back_func(self,msg):
        T_matrix = np.zeros((6,4,4))
        # Process the TF messages
        i = 0
        I = np.eye(4)
        Homo_matrix = np.eye(4)
        for transform in msg.transforms:
            # Extract relevant information from the message
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # Print the information
            t = np.array([translation.x, translation.y, translation.z])
            r = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
            Homo_matrix[:3,:3] = r.as_matrix()
            Homo_matrix[:3,-1] = t.T
            T_i_0 = np.dot(I,Homo_matrix)

            T_matrix[i,:,:] = T_i_0
            I = T_i_0
            i += 1
        self.T_matrix = np.round(T_matrix,5)
        self.T_e_0 = np.round(T_i_0,5)

    def camera_task_1(self):
            self.camera_view.detect_marker()     

    def camera_task_2(self):
            while True:
                self.Jp = self.camera_view.Jp  
                self.Sd = self.camera_view.Sd
                self.S = self.camera_view.S
                # self.O_g_c = self.camera_view.O_g_c # (3,1) coord of goal in camera coord frame
                # self.O_c = self.camera_view.O_c # (3,1)  marker center current coord in camera coord frame

    def jacobian_matrix(self,n,T_matrix,O_C_0):
        R,O = T_matrix[:,:3,:3], T_matrix[:,:3,-1].T  # (6,3,3) and (6,3).T
        # (R, O) = transformation_matrix(n,alpha,a,d,theta)

        # O_C_0 = np.transpose(np.array([O[:,-1]]))
        Jz = np.zeros((3,n))
        Jw = np.zeros((3,n))

        for i in range(0,n):
            Z_i_0 = np.transpose(np.array([R[i,:,2]]))
            O_i_0 = np.transpose(np.array([O[:,i]]))

            cross_prod = np.cross(Z_i_0, O_C_0 - O_i_0, axis=0)
            Jz[:,i] = np.reshape(cross_prod,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)
            Jw[:,i] = np.reshape(Z_i_0,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)
        J = np.concatenate((Jz,Jw),axis=0)
        return J

    def task_perform(self):
        #self.move_joint([0,0,0,0,0,0], 30, 10, 0, 0, 0, 0, 0)  # move to home
        self.move_joint([0,10,-60,0,-110,0], 30, 10, 0, 0, 0, 0, 0)  # move to home
        self.rate.sleep()  # dynamically choose the correct time to sleep

        # DH Parameters
        n = 6  # No of joints
        m = 6
        alpha = np.array([0, -pi/2, 0, pi/2, -pi/2, pi/2])   
        a = np.array([0, 0, 0.409, 0, 0, 0])
        d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])

        joint_offset = np.array([0, -pi/2, pi/2, 0, 0, 0])  # Difference btw Hardware/Gazebo & DH convention

        time.sleep(1) # sleep for 1 sec

        R_cam2gripper = np.array([[-0.0015199, 0.99996921, 0.00769817],
                                  [-0.99996814, -0.00145948, -0.00784797],
                                  [-0.0078365, -0.00770985, 0.99993957]])
        
        t_cam2gripper = np.array([[0.05482128],[0.03155016],[0.02497541]]) # in meters

        # O_e_w = self.T_e_0[:3,-1]
        # R_e_w = self.T_e_0[:3,:3]
        # O_c_e = t_cam2gripper
        # R_c_e = R_cam2gripper
        # R_c_w = R_e_w @ R_c_e

        # O_p_w = np.reshape(O_e_w,(3,1)) + R_e_w @ O_c_e + R_c_w @ np.reshape(self.O_p_c,(3,1))
        # print(O_p_w)

        # T_c_e = np.eye((4))
        # T_c_e[:3,:3], T_c_e[:3,-1] = R_cam2gripper, np.reshape(t_cam2gripper,(3,)) 

        d_S = self.Sd - self.S
        
        q = np.radians(np.array(get_current_posj())) #np.array([0,0,0,0,0,0]) # Hardware & Gazebo Joint Angle at Home Position
        theta = q + joint_offset  # Initial Joint position as per DH convention (In radians)  
        
        # iterate through multiple frames, in a live video feed
        while np.linalg.norm(d_S) > 0.1: 
            # T_c_0 = self.T_e_0 @ T_c_e

            O_C_0 = self.T_e_0[:3,:3] @ t_cam2gripper  # coord of camera frame wrt to world

            # Cam_vel_wrt_C = np.linalg.inv(self.Jp) @ d_S  # camera vecocity in camera frame

            Je = self.jacobian_matrix(n,self.T_matrix,O_C_0)  # Calculate J

            d_X = np.linalg.pinv(self.Jp) @ d_S

            d_theta = np.linalg.pinv(Je) @ d_X

            # Calculating Next joint Position
            theta_new = np.degrees(theta) + np.reshape(d_theta,(n,))  # In degrees
            q = np.radians(theta_new) - joint_offset  # for Hardware and Gazebo, # In radians

            theta = np.radians(theta_new)  # for DH-convention, In radians

            self.move_joint(np.degrees(q), 20, 10, 0, 0, 0, 0, 0)
            self.rate.sleep()  # dynamically choose the correct time to sleep
            print(np.degrees(q))
            
            d_S = self.Sd - self.S
            # T_c_0_old = T_c_0_new

        self.camera_thread_1.join()
        print('Thread is killed')

if __name__ == "__main__":
    # Start Camera
    camera_view = Camera()
    VS = Visual_Servoing(camera_view)
    print("yes")
    VS.task_perform()
