from math import *
import numpy as np


def transformation_matrix(n,alpha,a,d,theta):
    I = np.eye(4)
    R = np.zeros((n,3,3))
    O = np.zeros((3,n))

    # Transformation Matrix
    for i in range(0,n):
        T = np.array([[      cos(theta[i])        ,      -sin(theta[i])        ,        0      ,        a[i]        ],
                      [sin(theta[i])*cos(alpha[i]), cos(theta[i])*cos(alpha[i]), -sin(alpha[i]), -d[i]*sin(alpha[i])],                                               
                      [sin(theta[i])*sin(alpha[i]), cos(theta[i])*sin(alpha[i]),  cos(alpha[i]),  d[i]*cos(alpha[i])],     
                      [             0             ,             0              ,        0      ,          1         ]])

        T_new = np.dot(I,T)
        R[i,:,:] = T_new[0:3,0:3]
        O[0:3,i] = T_new[0:3,3]
        I = T_new
        i= i + 1

    # T_final = I
    # d_nn = np.array([[0.138],[0],[0],[1]])
    # P_00_home = np.dot(T_final,d_nn)
    # P_00 = P_00_home[0:3]
    return(R,O)

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def forward_kinematics(n,alpha,a,d,theta,le=0):
    I = np.eye(4)
    R = np.zeros((n,3,3))
    O = np.zeros((3,n))

    # Transformation Matrix
    for i in range(0,n):
        T = np.array([[    cos(theta[i])        ,        -sin(theta[i])      ,        0      ,         a[i]       ],
                    [sin(theta[i])*cos(alpha[i]), cos(theta[i])*cos(alpha[i]), -sin(alpha[i]), -d[i]*sin(alpha[i])],                                               
                    [sin(theta[i])*sin(alpha[i]), cos(theta[i])*sin(alpha[i]),  cos(alpha[i]),  d[i]*cos(alpha[i])],     
                    [             0             ,              0             ,        0      ,           1        ]])

        T_new = np.dot(I,T)
        R[i,:,:] = T_new[0:3,0:3]
        O[0:3,i] = T_new[0:3,3]
        I = T_new
        i= i + 1

    T_final = I
    d_nn = np.array([[le],[0],[0],[1]])
    P_00_home = np.dot(T_final,d_nn)
    P_00 = P_00_home[0:3]

    X_cord = np.array([0,O[0,0],O[0,1],O[0,2],O[0,3],O[0,4],O[0,5],P_00[0,0]])
    Y_cord = np.array([0,O[1,0],O[1,1],O[1,2],O[1,3],O[1,4],O[1,5],P_00[1,0]])
    Z_cord = np.array([0,O[2,0],O[2,1],O[2,2],O[2,3],O[2,4],O[2,5],P_00[2,0]])
    return(X_cord[-1],Y_cord[-1],Z_cord[-1])

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def jacobian_matrix(n,alpha,a,d,theta,le=0):
    (R, O) = transformation_matrix(n,alpha,a,d,theta)

    R_n_0 = R[n-1,:,:]
    O_n_0 = np.transpose(np.array([O[:,n-1]]))
    O_E_n = np.array([[le],[0],[0]])
    O_E = O_n_0 + np.dot(R_n_0,O_E_n)

    Jz = np.zeros((3,n))
    Jw = np.zeros((3,n))

    for i in range(0,n):
        Z_i_0 = np.transpose(np.array([R[i,:,2]]))
        O_i_0 = np.transpose(np.array([O[:,i]]))
        O_E_i_0 = O_E - O_i_0

        cross_prod = np.cross(Z_i_0,O_E_i_0,axis=0)
        Jz[:,i] = np.reshape(cross_prod,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)
        Jw[:,i] = np.reshape(Z_i_0,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)

    J = np.concatenate((Jz,Jw),axis=0)
    return np.round(J, 3)

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def weight_Func(m,n,q_range,q,epsilon):
    const = 400
    We = np.zeros((m,m))
    for i in range(0,m):
        We[i,i] = 50

    Wc = np.zeros((n,n))
    for i in range(0,n):
        if q[0,i] < q_range[i,0]:
            Wc[i,i] = const
        elif q_range[i,0] <= q[0,i] <= (q_range[i,0] + epsilon[0,i]):
            Wc[i,i] = (const/2)*(1 + cos(pi*((q[0,i] - q_range[i,0])/epsilon[0,i])))
        elif (q_range[i,0] + epsilon[0,i]) < q[0,i] < (q_range[i,1] - epsilon[0,i]):
            Wc[i,i] = 0
        elif (q_range[i,1] - epsilon[0,i]) <= q[0,i] <= q_range[i,1]:
            Wc[i,i] = (const/2)*(1 + cos(pi*((q_range[i,1] - q[0,i])/epsilon[0,i])))
        else:
            Wc[i,i] = const

    Wv = np.zeros((n,n))
    for i in range(0,n):
        Wv[i,i] = 0.5
    return We, Wc, Wv

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def cost_func(n,K,q,q_range,m):
    # Initiate
    c = np.zeros((n,))
    b = np.zeros((n,))
    del_phi_del_q = np.zeros((n,))
    q_c = np.mean(q_range,axis = 1); # column vector containing the mean of each row
    del_q = q_range[:,1] - q_range[:,0]; # Total working range of each joint

    for i in range(0,n):
        if q[0,i] >= q_c[i]:
            c[i] = pow((K[i,i]*((q[0,i] - q_c[i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q[0,i] - q_c[i])/del_q[i])),m-1)
        elif q[0,i] < q_c[i]:
            c[i] = pow((K[i,i]*((q_c[i] - q[0,i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q_c[i] - q[0,i])/del_q[i])),(m-1))

    L = np.sum(c)

    for j in range(0,n):
        if q[0,j] >= q_c[j]:
            del_phi_del_q[j] = pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])
        elif q[0,j] < q_c[j]:
            del_phi_del_q[j] = -pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])

    v = -del_phi_del_q
    return v