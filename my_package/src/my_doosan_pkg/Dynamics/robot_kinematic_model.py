import numpy as np
import numpy.linalg as LA
from math import *

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0),
    'sxyx': (0, 0, 1, 0),
    'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0),
    'syzx': (1, 0, 0, 0),
    'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0),
    'syxy': (1, 1, 1, 0),
    'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0),
    'szyx': (2, 1, 0, 0),
    'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1),
    'rxyx': (0, 0, 1, 1),
    'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1),
    'rxzy': (1, 0, 0, 1),
    'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1),
    'ryxy': (1, 1, 1, 1),
    'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1),
    'rxyz': (2, 1, 0, 1),
    'rzyz': (2, 1, 1, 1),
}

_TUPLE2AXES = {v: k for k, v in _AXES2TUPLE.items()}


class Robot_KM:
    def __init__(self,n,alpha,a,d,d_nn,DH_params="modified"):
        self.n = n
        self.alpha = alpha
        self.a = a
        self.d = d
        self.d_nn = d_nn
        self.DH_params = DH_params

        # off-set in joint angles
        self.offset = np.array([0.0, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0])

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

    def J(self, theta):
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

    def J_dot(self, theta, theta_dot, H=None):
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
    
    def FK(self, theta, theta_dot=None, theta_ddot=None, level="pos"):
        theta = theta + self.offset
        
        if level == "pos":
            R, _, EE_pos = self._transformation_matrix(theta)  # end-effector position
            # EE_orient = self.rotm2eulxyz(R[-1,:,:])  # end-effector orientation
            # # EE_orient = self.mat2euler(R[-1,:,:], axes='rzyz')
            # self.Xe = np.concatenate([1000*EE_pos.reshape(-1), np.degrees(EE_orient)])  # end-effector position
            # return self.Xe.astype(np.float64), [], []
            return R[-1,:,:], [], []

        # if level == "vel":
        #     _,J,_ = self.J(theta)   # only linear velocity

        #     if theta_dot.ndim != 2:
        #         theta_dot = theta_dot.reshape((self.n, 1))

        #     Xe_dot = J @ theta_dot   # end-effector velocity
        #     _, _, P_00 = self._transformation_matrix(theta)  # end-effector position
        #     self.Xe = np.array([P_00[[0],0],P_00[[1],0],P_00[[2],0]])  # end-effector position
        #     return self.Xe.astype(np.float64), Xe_dot.astype(np.float64), []
        
        # if level == "acc":
        #     _,J,_ = self.J(theta)   # only linear velocity

        #     _,J_dot,_ = self.J_dot(theta, theta_dot)

        #     if theta_dot.ndim != 2:
        #         theta_dot = theta_dot.reshape((self.n, 1))
        #     if theta_ddot.ndim != 2:
        #         theta_ddot = theta_ddot.reshape((self.n, 1))

        #     Xe_dot = J @ theta_dot    # end-effector velocity
        #     Xe_ddot = J @ theta_ddot + J_dot @ theta_dot    # end-effector acceleration
        #     _, _, P_00 = self._transformation_matrix(theta)  # end-effector position
        #     self.Xe = np.array([P_00[[0],0],P_00[[1],0],P_00[[2],0]])  # end-effector position
        #     return self.Xe.astype(np.float64), Xe_dot.astype(np.float64), Xe_ddot.astype(np.float64)  

