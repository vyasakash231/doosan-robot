import numpy as np
import numpy.linalg as LA
from math import *


class Robot_KM:
    def __init__(self,n,alpha,a,d,d_nn,DH_params="modified"):
        self.n = n
        self.alpha = alpha
        self.a = a
        self.d = d
        self.d_nn = d_nn
        self.DH_params = DH_params

        # Difference btw Hardware/Gazebo & DH convention
        self.offset = np.array([0.0, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0])

        # Joint limit
        self.q_limit = np.radians(np.array([[-360,360],[-95,95],[-135,135],[-360,360],[-135,135],[-360,360]]))

        # base transform to account for robot's reference frame
        self.T_base = np.array([
            [1.0, 0.0, 0.0, -0.00064],   # Approximate x offset (in meters)
            [0.0, 1.0, 0.0, -0.00169],   # Approximate y offset (in meters)
            [0.0, 0.0, 1.0, 0.0],     # No z offset
            [0.0, 0.0, 0.0, 1.0]
        ])

    def _transformation_matrix(self,theta):
        # I = np.eye(4)
        I = self.T_base.copy()
        R = np.zeros((self.n,3,3))
        O = np.zeros((3,self.n))

        if self.DH_params == "modified":
            # Transformation Matrix
            for i in range(self.n):
                T = np.array([[         cos(theta[i])          ,          -sin(theta[i])         ,           0        ,          self.a[i]           ],
                              [sin(theta[i])*cos(self.alpha[i]), cos(theta[i])*cos(self.alpha[i]), -sin(self.alpha[i]), -self.d[i]*sin(self.alpha[i])],                                               
                              [sin(theta[i])*sin(self.alpha[i]), cos(theta[i])*sin(self.alpha[i]),  cos(self.alpha[i]),  self.d[i]*cos(self.alpha[i])],     
                              [               0                ,                 0               ,           0        ,               1              ]])
                
                T_new = np.dot(I,T)
                R[i,:,:] = T_new[:3,:3]
                O[:3,i] = T_new[:3,3]
                I = T_new

        if self.DH_params == "standard":
            # Transformation Matrix
            for i in range(self.n):
                T = np.array([[cos(theta[i]), -sin(theta[i])*cos(self.alpha[i]),  sin(theta[i])*sin(self.alpha[i]), self.a[i]*cos(theta[i])],
                              [sin(theta[i]),  cos(theta[i])*cos(self.alpha[i]), -cos(theta[i])*sin(self.alpha[i]), self.a[i]*sin(theta[i])],                                               
                              [       0     ,       sin(self.alpha[i])         ,       cos(self.alpha[i])         ,        self.d[i]       ],     
                              [       0     ,                 0                ,                 0                ,               1        ]])
                
                T_new = np.dot(I,T)
                R[i,:,:] = T_new[:3,:3]
                O[:3,i] = T_new[:3,3]
                I = T_new

        P_00 = O[:,[-1]] + np.dot(R[-1,:,:], self.d_nn)
        return  R, O, P_00

    def Jacobian(self, theta):
        theta = theta + self.offset
        R, O, O_E = self._transformation_matrix(theta)

        Jz = np.zeros((3,self.n))
        Jw = np.zeros((3,self.n))

        for i in range(self.n):
            Rm = R[i,:,:]
            Z_i_0 = Rm[:,[2]]
            O_i_0 = O[:,[i]]
            O_E_i_0 = O_E - O_i_0
            
            cross_prod = np.cross(Z_i_0, O_E_i_0, axis=0)
            
            Jz[:,i] = cross_prod.reshape(-1)   # conver 2D of shape (3,1) to 1D of shape (3,)
            Jw[:,i] = Z_i_0.reshape(-1)   # conver 2D of shape (3,1) to 1D of shape (3,)

        jacobian = np.concatenate((Jz,Jw),axis=0)
        return jacobian.astype(np.float64), Jz.astype(np.float64), Jw.astype(np.float64)

    def Jacobian_dot(self, theta, theta_dot, H=None):
        theta = theta + self.offset

        """ https://doi.org/10.48550/arXiv.2207.01794 """
        if H is None:
            H = self.Hessian(theta)

        J_dot = np.zeros((6,self.n))
        
        for i in range(self.n):
            J_dot[:,[i]] = H[i,:,:].T @ theta_dot[:, np.newaxis]
        Jz_dot = J_dot[:3,:]
        Jw_dot = J_dot[3:,:]
        return J_dot.astype(np.float64), Jz_dot.astype(np.float64), Jw_dot.astype(np.float64)

    # only for Revolute joints
    def Hessian(self, theta):
        theta = theta + self.offset

        """ 
        Hessian_v = [H_1; H_2; ... ; H_6] = [(nxn)_1; (nxn)_2; ... ; (nxn)_6], where, H_i -> ith stacks of (n,n) matrix,
        Eqn (37) from this paper - https://doi.org/10.1109/CIRA.2005.1554272
        """    
        H = np.zeros((self.n, self.n, 6))  #  last index in Hessian_v is stack

        R, O, _ = self._transformation_matrix(theta)

        R_n_0 = R[self.n-1,:,:]
        O_n_0 = O[:,[self.n-1]]
        O_E_n = self.d_nn 
        O_E_0 = O_n_0 + np.dot(R_n_0,O_E_n)
        
        for i in range(self.n):
            Ri = R[i,:,:]
            Z_i_0 = Ri[:,[2]]
            for j in range(self.n):
                Rj = R[j,:,:]
                Z_j_0 = Rj[:,[2]]
                O_j_0 = O[:,[j]]
                O_E_j_0 = O_E_0 - O_j_0

                if i <= j:
                    cross_prod_j = np.cross(Z_j_0, O_E_j_0, axis=0)
                    H_z = np.cross(Z_i_0, cross_prod_j, axis=0)

                    if i != j:
                        H_w = np.cross(Z_i_0, Z_j_0, axis=0)
                    else:
                        H_w = np.zeros((3,1))

                    H[i,j,:] = np.concatenate((H_z.reshape(-1), H_w.reshape(-1)))
                else:
                    H[i,j,:] = H[j,i,:].copy()
        return H
    
    def taskspace_coord(self,theta):
        theta = theta + self.offset
        _, O, P_00 = self._transformation_matrix(theta)

        X_cord = np.array([0,O[0,0],O[0,1],O[0,2],O[0,3],O[0,4],O[0,5],P_00[0,0]])
        Y_cord = np.array([0,O[1,0],O[1,1],O[1,2],O[1,3],O[1,4],O[1,5],P_00[1,0]])
        Z_cord = np.array([0,O[2,0],O[2,1],O[2,2],O[2,3],O[2,4],O[2,5],P_00[2,0]])
        return X_cord, Y_cord, Z_cord   
    
    def rot2quat(self, rmat):
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
        inds = np.array([1, 2, 3, 0])  # order -> (x,y,z,w) 
        return q1[inds]
    
    def _skew4x4(self, z):
        mat = np.zeros((4,4))

        mat[0,1:] = -z  # [-ωx,-ωy,-ωz]
        mat[1:,0] = z  # [ωx,ωy,ωz]
        mat[1,2] = z[2]  # ωz
        mat[1,3] = -z[1]  # -ωy
        mat[2,1] = -z[2]  # -ωz
        mat[2,3] = z[0]  # ωx
        mat[3,1] = z[1]  # ωy
        mat[3,2] = -z[0]  # -ωx
        return mat
    
    def FK(self, theta, theta_dot=None, theta_ddot=None, level="pos"):
        theta = theta + self.offset
        
        if level == "pos":
            R, _, EE_pos = self._transformation_matrix(theta)
            EE_quat = self.rot2quat(R[-1,:,:])   # quaternion (x,y,z,w)
            self.Xe = np.concatenate([1e3*EE_pos.reshape(-1), EE_quat])  # end-effector pose [(X,Y,Z) -> meters, (x,y,z,w) -> quaternions]
            
            return self.Xe.astype(np.float64), [], []

        if level == "vel":
            R, _, EE_pos = self._transformation_matrix(theta)
            EE_quat = self.rot2quat(R[-1,:,:])  # quaternion (x,y,z,w)
            self.Xe = np.concatenate([1e3*EE_pos.reshape(-1), EE_quat])  # end-effector pose [(X,Y,Z) -> meters, (x,y,z,w) -> quaternions]

            J,_,_ = self.J(theta)   # Jacobian (6xn)
            
            if theta_dot.ndim != 2:
                theta_dot = theta_dot[:,np.newaxis]
            
            Xe_dot = J @ theta_dot   # end-effector velocity [(Vx,Vy,Vz) -> meters/s, (ωx,ωy,ωz) -> rad/s]
            
            """Q_dot = 0.5 * Ω(ω) * Q, where Q_dot is the quaternion derivative which depends on the angular velocity ω"""
            # omega = Xe_dot[3:].reshape(-1)
            # EE_quat_dot = 0.5 * self._skew4x4(omega) @ EE_quat[:,np.newaxis]  # (dx,dy,dz,dw)
            # Xe_dot = np.concatenate([Xe_dot[:3].reshape(-1), EE_quat_dot.reshape(-1)])   # end-effector velocity [(Vx,Vy,Vz) -> meters/s, (dx,dy,dz,dw)]

            return self.Xe.astype(np.float64), Xe_dot.astype(np.float64), []
        
        # if level == "acc":
        #     R, _, EE_pos = self._transformation_matrix(theta)
        #     EE_quat = self.rot2quat(R[-1,:,:])  # quaternion (x,y,z,w)
        #     self.Xe = np.concatenate([1e3*EE_pos.reshape(-1), EE_quat])  # end-effector pose [(X,Y,Z) -> meters, (x,y,z,w) -> quaternions]

        #     J,_,_ = self.J(theta)   # Jacobian (6xn)
        #     J_dot,_,_ = self.J_dot(theta, theta_dot)   # Jacobian_dot (6xn)

        #     if theta_dot.ndim != 2:
        #         theta_dot = theta_dot[:, np.newaxis]
        #     if theta_ddot.ndim != 2:
        #         theta_ddot = theta_ddot[:,np.newaxis]

        #     Xe_dot = J @ theta_dot    # end-effector velocity [(Vx,Vy,Vz) -> meters/s, (ωx,ωy,ωz) -> rad/s]
        #     Xe_ddot = J @ theta_ddot + J_dot @ theta_dot    # end-effector acceleration  [(Ax,Ay,Az) -> meters/s^2, (ωx_dot,ωy_ot,ωz_dot) -> rad/s^2]
            
        #     return self.Xe.astype(np.float64), Xe_dot.astype(np.float64), Xe_ddot.astype(np.float64)  

