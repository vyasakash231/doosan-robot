from math import *
import numpy as np

EPS = np.finfo(float).eps * 4.0

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
        if q[i] < q_range[i,0]:
            Wc[i,i] = const
        elif q_range[i,0] <= q[i] <= (q_range[i,0] + epsilon[i]):
            Wc[i,i] = (const/2)*(1 + cos(pi*((q[i] - q_range[i,0])/epsilon[i])))
        elif (q_range[i,0] + epsilon[i]) < q[i] < (q_range[i,1] - epsilon[i]):
            Wc[i,i] = 0
        elif (q_range[i,1] - epsilon[i]) <= q[i] <= q_range[i,1]:
            Wc[i,i] = (const/2)*(1 + cos(pi*((q_range[i,1] - q[i])/epsilon[i])))
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
    del_phi_del_q = np.zeros((n,1))
    q_c = np.mean(q_range,axis = 1); # column vector containing the mean of each row
    del_q = q_range[:,1] - q_range[:,0]; # Total working range of each joint

    for i in range(0,n):
        if q[i] >= q_c[i]:
            c[i] = pow((K[i,i]*((q[i] - q_c[i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q[i] - q_c[i])/del_q[i])),m-1)
        elif q[i] < q_c[i]:
            c[i] = pow((K[i,i]*((q_c[i] - q[i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q_c[i] - q[i])/del_q[i])),(m-1))

    L = np.sum(c)

    for j in range(0,n):
        if q[j] >= q_c[j]:
            del_phi_del_q[j] = pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])
        elif q[j] < q_c[j]:
            del_phi_del_q[j] = -pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])

    v = -del_phi_del_q
    return v

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

"""Convert Euler angles (ZYZ) to quaternions"""
def euler2quaternion(a, b, g):
    # Convert angles in degrees to radians
    a = radians(a)
    b = radians(b)
    g = radians(g)

    # Rotation matrix from Euler angles
    R_d = [[cos(a)*cos(b)*cos(g) - sin(a)*sin(g), -cos(a)*cos(b)*sin(g) - sin(a)*cos(g), cos(a)*sin(b)],
           [sin(a)*cos(b)*cos(g) + cos(a)*sin(g), -sin(a)*cos(b)*sin(g) + cos(a)*cos(g), sin(a)*sin(b)],
           [          -sin(b)*cos(g)            ,             sin(b)*sin(g)            ,  cos(b)      ]]

    # Set of quaternions giving the same orientation expressed by Euler angles
    q0 = 0.5 * sqrt(1 + R_d[0][0] + R_d[1][1] + R_d[2][2])
    q1 = (1 / (4 * q0)) * (R_d[2][1] - R_d[1][2])
    q2 = (1 / (4 * q0)) * (R_d[0][2] - R_d[2][0])
    q3 = (1 / (4 * q0)) * (R_d[1][0] - R_d[0][1])
    return q0, q1, q2, q3

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def unit_vector(data, axis=None, out=None):
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
    
def quat_slerp(quat0, quat1, fraction, shortestpath=True):
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