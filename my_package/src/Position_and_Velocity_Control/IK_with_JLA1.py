#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot, weight_Func


class Task_Space_Control(Robot):
    def __init__(self):
        self.shutdown_flag = False
        
        super().__init__()
          
        # Set JOINT-LIMITS
        self.joint_vel_limits([100, 100, 100, 100, 100, 100])   # set global joint velocity limit
        self.joint_acc_limits([100, 100, 100, 100, 100, 100])   # set global joint acceleration limit
        self.ee_vel_limits(100,10)   # set global Task velocity limit
        self.ee_acc_limits(200,10)   # set global Task acceleration limit

    @property
    def current_pose(self):
        pose = np.zeros((self.n+1,1))
        pose[:3,0] = 0.001 * self.Robot_RT_State.actual_tcp_position[:3]  # converting position from mm to m
        pose[3:,0] = self.eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # Convert angles from Euler ZYZ (in degrees) to quaternion 
        return pose[:3,[0]]    

    @property
    def theta(self):
        return 0.0174532925 * self.Robot_RT_State.actual_joint_position   #  convert deg to rad

    def goal_state(self, pose):
        self.goal_pose = 0.001 * pose   # convert from mm to m
        self.initial_error = (self.goal_pose - self.current_pose).copy()

    def _adaptive_gain(self, idx, error):
        # ADAPTIVE GAIN: formulation
        e_norm = np.linalg.norm(error)
        e0_norm = np.linalg.norm(self.initial_error)
        K = 20 * (1-exp(-idx*0.05)) * exp((e0_norm - e_norm) / e0_norm)  # ADAPTIVE GAIN factor
        return K

    def JLA_1(self):  # Inequality Constraint Method
        i = 0
        self.del_X = np.ones((3,1))  # defining an array to start the loop
        self.d_theta = np.zeros((6,1))  # defining an array to store joint velocity
        self.q_plt = self.theta  # Hardware & Gazebo Joint Angle at Home Position

        joint_limits = self.kinematic_model.q_limit  # in radians
        Jc = np.eye(self.n)  # Jacobian of the additional task
        epsilon = 0.5 * np.ones(self.n)  # Activation buffer region width

        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate
        m = 3  # mth norm of a vector
        
        try:
            while not rospy.is_shutdown() and np.linalg.norm(self.del_X[:,i]) > 0.005:   # if error is less then 5mm 
                # Check if each element of tetha lies within the corresponding range in q_range
                if np.all((joint_limits[:, 0] <= self.theta) & (self.theta <= joint_limits[:, 1]), axis=0) == 'False':
                    rospy.signal_shutdown('Joint Limit breached')
                    
                # Calculating Weight Matrix
                We, Wc, Wv = weight_Func(m, self.n, joint_limits, self.theta, epsilon)

                # Calculate Jacobain
                _,Je,_ = self.kinematic_model.Jacobian(self.theta)  # Calculate J

                # Jn = np.linalg.inv(np.transpose(Je) @ We @ Je + np.transpose(Jc) @ Wc @ Jc + Wv) @ np.transpose(Je) @ We
                Jn = self._svd_solve(np.transpose(Je) @ We @ Je + np.transpose(Jc) @ Wc @ Jc + Wv) @ np.transpose(Je) @ We

                # pose error
                error = self.goal_pose - self.current_pose  # [Xg - X]
                
                # Control Input => Joint Velocity
                joint_velocity = self._adaptive_gain(i, error) * (Jn @ error)  # joint_velocity = K * inv(J) @ [Xg - X]

                writedata = SpeedJRTStream()
                writedata.vel = joint_velocity.reshape(-1).tolist()  
                writedata.time = 0.1
                self.speedj_publisher.publish(writedata)  # Publish Joint Velocity in degree/sec

                # store position error data for plotting
                self.del_X = np.hstack((self.del_X, error)) 

                # Store Joint Velocity for plotting
                self.d_theta = np.hstack((self.d_theta, joint_velocity))

                # store joint angles for plotting
                self.q_plt = np.vstack((self.q_plt, self.theta)) # In radians

                # print(f"error at {i}th iteration: {np.linalg.norm(self.del_X[:,i])}")

                i = i + 1
                rate.sleep()

        except rospy.ROSInterruptException:
            pass

        finally:
            self.cleanup()
            self.plot(i)

    def plot(self, i):
        (Row,Column) = self.del_X.shape

        plt.figure()
        rms_values = np.sqrt(np.mean(self.del_X**2, axis=0)) # Compute RMS along each column
        plt.plot(range(1,Column),self.del_X[0,1:Column], 'r--')
        plt.plot(range(1,Column),self.del_X[1,1:Column], 'b--')
        plt.plot(range(1,Column),self.del_X[2,1:Column], 'g--')
        plt.plot(range(1,len(rms_values)),rms_values[1:], 'k-')

        plt.xlabel('No of Iteration')
        plt.ylabel('$\Delta$X')
        plt.legend(['$e_{X}$','$e_{Y}$','$e_{Z}$','$RMS_{Error}$'])
        plt.grid()
                
        plt.figure()
        plt.plot(range(0,i+1),self.d_theta[0,:], 'r-')
        plt.plot(range(0,i+1),self.d_theta[1,:], 'b-')
        plt.plot(range(0,i+1),self.d_theta[2,:], 'g-')
        plt.plot(range(0,i+1),self.d_theta[3,:], 'y-')
        plt.plot(range(0,i+1),self.d_theta[4,:], 'c-')
        plt.plot(range(0,i+1),self.d_theta[5,:], 'm-')

        plt.xlabel('No of Iteration')
        plt.ylabel('$Joint Velocity$')
        plt.legend(['$Joint_{1}$','$Joint_{2}$','$Joint_{3}$','$Joint_{4}$','$Joint_{5}$','$Joint_{6}$'])
        plt.grid()

        joints = ['theta_1','theta_2','theta_3','theta_4','theta_5','theta_6']
        plt.figure(figsize=(20,10))
        plt.tight_layout(pad=3.0) # give some spacing btw two subplts
        for i in range(1,Task_Space_Control.n+1):
            plt.subplot(2,3,i) 
            plt.plot(range(0,Column), np.degrees(self.q_plt[:,i-1]), '-.')
            plt.ylim(-360,360)
            plt.grid()
            plt.ylabel(joints[i-1])
            plt.xlabel('iteration')
        plt.show()


if __name__ == "__main__":
    p1= posj(0,0,90,0,90,0)  # posj(q1, q2, q3, q4, q5, q6), set initial joint angles
    movej(p1, vel=40, acc=20)
    time.sleep(1.0)

    try:
        rospy.init_node('My_service_node')  # Initialize the ROS node

        task = Task_Space_Control()  # task = Instance / object of Task_space_Control class 

        rospy.sleep(2.0)  # Give buffer time for Service to activate 

        # GOAL POSITION
        pose = np.array([[200],[300],[700]])   # in [mm, deg]
        task.goal_state(pose)

        task.JLA_1()  # Publish new joint angle using JLA_1

    except rospy.ROSInterruptException:
        pass 
    finally:
        pass
