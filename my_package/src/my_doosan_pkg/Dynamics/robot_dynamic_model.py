import os
import sys
import numpy as np
import numpy.linalg as LA
np.set_printoptions(suppress=True)
from sympy import *

from .euler_lagrange import Euler_Lagrange


class Robot_Dynamics:
    # TODO: Forward Dynamics: Brute Force (Matrix Inversion), Articulated-Body Algorithm
    # TODO: Inverse Dynamics: Recursive Newton-Euler Algorithm
    # TODO: Mass Matrix: Composite Rigid Body Algorithm 
    
    def __init__(self, kinematic_property={}, mass=[], COG_wrt_body=[], MOI_about_body_COG=[], file_name=None):
        self.n = kinematic_property['dof']
        self.alpha = kinematic_property['alpha']
        self.a = kinematic_property['a']
        self.d = kinematic_property['d']
        self.d_nn = kinematic_property['d_nn']

        self.m = mass
        self.CG = COG_wrt_body  # location of COG wrt to DH-frame / body frame
        if len(MOI_about_body_COG) == 0:  # if list is empty then MOI is zero
            self.MOI = [np.array([[0,0,0],[0,0,0],[0,0,0]]) for _ in range(len(self.m))]
        else:  # else MOI is non-zero 
            self.MOI = MOI_about_body_COG  # MOI of the link about COG based on DH-frame / body frame

        robot_DM = Euler_Lagrange(self.n, self.CG, self.MOI, self.d_nn)  # Symbolic Dynamic Model using Euler-Lagrange Method
        
        # Check if the file already exists
        if os.path.exists('../models/'+file_name+'.pkl'):
            print(f"{file_name} already exists. Skipping recreation.")
        else:
            # Code to create the file and save the data
            M_sym, C_vec_sym, C_mat_sym, G_sym = robot_DM.mcg_matrix(self.alpha, self.a, self.d, self.m)  # Symbolic Matrices
            robot_DM.save_equations(M_sym, C_vec_sym, C_mat_sym, G_sym, file_name)  # Save Matrices
            print(f"{file_name} has been created.")

        # Load the pre-computed symbolic equations
        self.equations = robot_DM.load_equations(file_name)

    def compute_dynamics(self, q, q_dot):
        M_args = q
        C_args = np.concatenate((q, q_dot))
        G_args = q

        M = self.equations['M_lambda'](*M_args)
        C_vec = self.equations['C_vec_lambda'](*C_args)
        C_mat = self.equations['C_mat_lambda'](*C_args)
        G = self.equations['G_lambda'](*G_args)
        return M, C_vec, C_mat, G

    """Eqn (18) from https://doi.org/10.1109/JRA.1987.1087068"""
    def Mx(self, Mq, J):
        threshold = 1e-3
        Mx_inv = J @ LA.inv(Mq) @ J.T
        if abs(np.linalg.det(Mx_inv)) >= threshold:
            Mx = LA.inv(Mx_inv)
        else:
            Mx = LA.pinv(Mx_inv)
        return Mx

    """Eqn (3.11) from the Book -> Cartesian Impedance Control of Redundant and Flexible-Joint Robots"""
    def Cx(self, Mq, Cq, J, J_dot):
        J_inv = J.T @ np.linalg.inv(J @ J.T)   # pseudoinverse of the Jacobian
        Cx = J_inv.T @ (Cq - Mq @ J_inv @ J_dot) @ J_inv
        return Cx  

    """Eqn (25) from https://doi.org/10.1109/JRA.1987.1087068"""
    def Gx(self, Gq, J):
        J_inv = J.T @ np.linalg.inv(J @ J.T)   # pseudoinverse of the Jacobian
        Gx = J_inv.T @ Gq
        return Gx

    def compute_MCGx(self, Mq, Cq, Gq, J, J_dot):
        Mx = self.Mx(Mq, J)
        Cx = self.Cx(Mq, Cq, J, J_dot)   # This is a matrix not vector
        Gx = self.Gx(Gq, J)
        return Mx, Cx, Gx
    
