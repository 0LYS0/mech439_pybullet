import pybullet as p
import numpy as np
from math import *
import time


def Rot_axis(axis, q):
    '''
    make rotation matrix along axis
    '''
    if axis==1:
        R = np.asarray([[1,0,0],
                        [0,cos(q),-sin(q)],
                        [0,sin(q),cos(q)]])
    if axis==2:
        R = np.asarray([[cos(q),0,sin(q)],
                        [0,1,0],
                        [-sin(q),0,cos(q)]])
    if axis==3:
        R = np.asarray([[cos(q),-sin(q),0],
                        [sin(q),cos(q),0],
                        [0,0,1]])
    return R


def Rot_axis_series(axis_list, rad_list):
    '''
    zyx rotation matrix - caution: axis order: z,y,x
    '''
    R = Rot_axis(axis_list[0], rad_list[0])
    for ax_i, rad_i in zip(axis_list[1:], rad_list[1:]):
        R = np.matmul(R, Rot_axis(ax_i,rad_i))
    return R


def Rot_rpy(rpy):
    return np.transpose(Rot_axis_series([1,2,3],np.negative(rpy)))


def Rot2zyx(R):
    '''
    rotation matrix to zyx angles - caution: axis order: z,y,x
    '''
    sy = sqrt(R[0,0]**2 + R[1,0]**2)

    if sy > 1e-10:
        x = atan2(R[2,1] , R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else:
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0
    return np.asarray([z,y,x])


def Rot2rpy(R):
    return np.asarray(list(reversed(Rot2zyx(R))))


def xyzrpyToSE3(xyz, rpy):

    xyzSE3 = np.array([[1, 0, 0, xyz[0]],
                       [0, 1, 0, xyz[1]],
                       [0, 0, 1, xyz[2]],
                       [0, 0, 0, 1]])

    rotX = np.array([[1, 0, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0]), 0],
                     [0, sin(rpy[0]), cos(rpy[0]), 0],
                     [0, 0, 0, 1]])

    rotY = np.array([[cos(rpy[1]), 0, sin(rpy[1]), 0],
                     [0, 1, 0, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1]), 0],
                     [0, 0, 0, 1]])

    rotZ = np.array([[cos(rpy[2]), -sin(rpy[2]), 0, 0],
                     [sin(rpy[2]), cos(rpy[2]), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    return xyzSE3 @ rotZ @ rotY @ rotX


def calculate_q_bi(T_0b, T_0i):
    T_bi = np.linalg.inv(T_0b) @ T_0i
    q_bi = T_bi[0:3, 3]
    return q_bi


def calculate_omega_bi(T_0b, T_0i, axis):
    T_bi = np.linalg.inv(T_0b) @ T_0i
    # print(T_bi[0:3, 0])
    # print(T_bi[0:3, 1])
    # print(T_bi[0:3, 2])
    if axis == 'x':
        omega_bi = T_bi[0:3, 0]
    elif axis == 'y':
        omega_bi = T_bi[0:3, 1]
    elif axis == 'z':
        omega_bi = T_bi[0:3, 2]
    return omega_bi


def calculate_body_screw(T_0b, T_0i, axis):
    q_bi = calculate_q_bi(T_0b, T_0i)
    omega_bi = calculate_omega_bi(T_0b, T_0i, axis)
    v_bi = np.cross(q_bi, omega_bi)
    B_i = np.concatenate([omega_bi, v_bi])
    return B_i


def TransInv(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3:4]

    Tinv = np.identity(4)
    Tinv[0:3, 0:3] = R.T
    Tinv[0:3, 3:4] = -R.T@p
    return Tinv


def VecToso3(V):
    '''
    :param V: numpy array [[x],[y],[z]]
    :return V_ceil: [[0,-z,y],[z,0,-x],[-y,x,0]]
    '''
    x = V[0, 0]
    y = V[1, 0]
    z = V[2, 0]
    V_ceil = np.array([[0, -z, y],
                       [z, 0, -x],
                       [-y, x, 0]])
    return V_ceil


def so3ToVec(so3):
    '''
    :param so3: [[0,-z,y],[z,0,-x],[-y,x,0]]
    :return:
    '''
    x = so3[2,1]
    y = so3[0,2]
    z = so3[1,0]
    vec = np.array([[x],[y],[z]])
    return vec


def VecTose3(vec):
    '''
    :param vec: [[],[],[],[],[],[]]
    :return:
    '''
    omg = vec[0:3, 0:1]
    v = vec[3:6, 0:1]
    omg_ceil = VecToso3(omg)
    top = np.concatenate([omg_ceil, v], 1)
    btm = np.array([[0, 0, 0, 0]])
    se3 = np.concatenate([top, btm], 0)
    return se3


def se3ToVec(se3):
    w = so3ToVec(se3[0:3, 0:3])
    v = se3[0:3, 3:4]
    return np.concatenate([w, v])


def Adjoint(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3:4]
    p_ceil = VecToso3(p)
    p_ceilR = p_ceil@R
    zero = np.zeros((3, 3))
    top = np.concatenate([R, zero], 1)
    btm = np.concatenate([p_ceilR, R], 1)
    Adj = np.concatenate([top, btm], 0)
    return Adj


def ad(V):
    w = V[0:3]
    v = V[3:6]

    res = np.zeros([6, 6])
    res[0:3, 0:3] = VecToso3(w)
    res[3:6, 3:6] = VecToso3(w)
    res[3:6, 0:3] = VecToso3(v)

    return res


def AxisAng3(V):
    '''
    :param V: [[],[],[]]
    :return: [axis, theta]
    '''
    theta = np.linalg.norm(V, axis=0)[0]
    if np.abs(theta)<0.000001:
        return [np.array([[0], [0], [0]]), 0]
    axis = V/theta
    return [axis, theta]


def MatrixExp3(M):
    vec3 = so3ToVec(M)
    if (np.linalg.norm(vec3) < 0.000001):
        R = np.eye(3)
    else:
        [omghat, theta] = AxisAng3(vec3)
        omgmat = M/theta
        R = np.eye(3) + sin(theta)*omgmat + (1-cos(theta))*(omgmat@omgmat)
    return R


def MatrixExp6(M):
    '''
    :param T:
    :return:
    '''
    omg = so3ToVec(M[0:3, 0:3])
    # v = M[0:3, 3:4]
    [omghat, theta] = AxisAng3(omg)
    if np.abs(theta)<0.000001:
        R = np.eye(3)
        top = np.concatenate([R, M[0:3, 3:4]], 1)
        btm = np.array([[0, 0, 0, 1]])
        T = np.concatenate([top, btm], 0)
        return T
    v = M[0:3, 3:4]/theta
    omgmat = M[0:3, 0:3]/theta
    R = np.eye(3) + sin(theta)*omgmat + (1-cos(theta))*(omgmat@omgmat)
    G = np.eye(3)*theta + (1-cos(theta))*omgmat + (theta-sin(theta))*(omgmat@omgmat)
    Gv = G@v
    top = np.concatenate([R, Gv], 1)
    btm = np.array([[0, 0, 0, 1]])
    T = np.concatenate([top, btm], 0)
    return T


def MatrixLog3(R):
    acosInput = (np.trace(R)-1)/2
    # print('Matrix Log')
    # print(R)
    if acosInput >= 1:
        so3mat = np.array([[0,0,0],[0,0,0],[0,0,0]])
        return so3mat
    elif acosInput <= -1:
        if np.abs(1 + R[2, 2]) > 0.000001:
            omg = (1 / sqrt(2 * (1 + R[2, 2]))) * np.array([[R[0, 2]], [R[1, 2]], [1+R[2, 2]]])
        elif np.abs(1 + R[1, 1]) > 0.000001:
            omg = (1 / sqrt(2 * (1 + R[1, 1]))) * np.array([[R[0, 1]], [1+R[1, 1]], [R[2, 1]]])
        else:
            omg = (1 / sqrt(2 * (1 + R[0, 0]))) * np.array([[1+R[0, 0]], [R[1, 0]], [R[2, 0]]])
        so3mat = VecToso3(pi*omg)
        return so3mat
    else:
        theta = acos(acosInput)
        so3mat = (1/(2*sin(theta)))*(R-R.T) * theta
        return so3mat


def MatrixLog6(T):
    [w, theta] = AxisAng3(so3ToVec(MatrixLog3(T[0:3, 0:3])))
    w_hat = VecToso3(w)


    if (np.linalg.norm(w) < 0.000001):
        theta = np.linalg.norm(T[0:3, 3:4])
        if (theta < 0.000001):
            v = np.zeros([3, 1])
        else:
            v = T[0:3, 3:4]/theta
    else:
        v = (np.identity(3)/theta-w_hat/2 + (1/theta - 1/(2*tan(theta/2)))*w_hat@w_hat)@T[0:3, 3:4]

    S = np.concatenate([w, v])
    return VecTose3(S)*theta


def bodyJacobian(Blist, thetalist):
    T = np.eye(4)
    J_b = np.zeros(np.shape(Blist))
    joint_num = np.shape(Blist)[1]
    i = joint_num # i = 7 for 7 DOF manipulator(kuka-iiwa)
    J_b[:, i-1:i] = Blist[:, i-1:i]
    i = i-1
    # initially i=6
    while i > 0:
        T = T@MatrixExp6(VecTose3((-1)*Blist[:, i:i+1]*thetalist[i]))
        J_b[:, i-1:i] = Adjoint(T)@Blist[:, i-1:i]
        i = i-1
    return J_b


def FK(M, Slist, thetalist):

    n = np.shape(thetalist)[0]

    T = M
    for i in range(n):
        T = MatrixExp6(VecTose3(Slist[:,[n-1-i]])*thetalist[n-1-i, 0]) @ T

    return T


def exp3_dt(xi_ceil, xi_dot_ceil):


    xi = so3ToVec(xi_ceil)
    xi_dot = so3ToVec(xi_dot_ceil)

    xi_norm = np.linalg.norm(xi)

    if xi_norm < 0.000001:
        res = np.zeros([3, 3])
    else:
        alpha = sin(xi_norm) / xi_norm
        beta = 2 * (1 - cos(xi_norm)) / (xi_norm * xi_norm)

        alpha_dot = xi.T@xi_dot/(xi_norm**2)*(1 - alpha - (xi_norm**2)*beta/2)
        beta_dot = 2*xi.T@xi_dot/(xi_norm**2)*(alpha - beta)

        res = alpha_dot*xi_ceil + alpha*xi_dot_ceil + beta_dot/2*xi_ceil@xi_ceil + beta/2*(xi_dot_ceil@xi_ceil + xi_ceil@xi_dot_ceil)

    return res


def dexp3(xi):
    xi_norm = np.linalg.norm(xi)
    if xi_norm < 0.000001:
        return np.eye(3)
    else:
        alpha = sin(xi_norm)/xi_norm
        beta = 2*(1-cos(xi_norm))/(xi_norm*xi_norm)
        so3_xi = VecToso3(xi)

        res = np.identity(3) + (beta/2)*so3_xi + ((1-alpha)/(xi_norm*xi_norm))*(so3_xi@so3_xi)
        return res


def dexp3_dt(xi, xi_dot):

    xi_norm = np.linalg.norm(xi)

    if xi_norm < 0.000001:
        res = 1/2*VecToso3(xi_dot)
    else:
        alpha = sin(xi_norm) / xi_norm
        beta = 2 * (1 - cos(xi_norm)) / (xi_norm * xi_norm)

        xi_ceil = VecToso3(xi)
        xi_dot_ceil = VecToso3(xi_dot)

        res = beta/2*xi_dot_ceil + (1-alpha)/(xi_norm**2)*(xi_ceil@xi_dot_ceil + xi_dot_ceil@xi_ceil) + (alpha-beta)/(xi_norm**2)*(xi.T@xi_dot)*xi_ceil - \
              (3*(1-alpha)/(xi_norm**4) - beta/(2*xi_norm**2))*(xi.T@xi_dot)*xi_ceil@xi_ceil

    return res


def dexp6(lamb):
    xi = lamb[0:3, 0:1]
    eta = lamb[3:6, 0:1]

    res = np.zeros([6, 6])
    res[0:3, 0:3] = dexp3(xi)
    res[3:6, 3:6] = dexp3(xi)
    res[3:6, 0:3] = dexp3_dt(xi, eta)

    return res


def IDyn_NewtonEuler(Mlist, Glist, Slist, theta, dtheta, ddtheta, g = np.array([[0], [0], [0]]), Ftip = np.zeros([6, 1])):

    n = np.shape(theta)[0]

    Ai = np.zeros([6, n])
    AdTi = np.zeros([n+1, 6, 6])
    Vi = np.zeros([6, n+1])
    Vdi = np.zeros([6, n+1])

    Vdi[3:6, 0:1] = -g
    AdTi[n] = Adjoint(TransInv(Mlist[n]) @ Mlist[n-1])
    Fi = Ftip
    tau = np.zeros([n, 1])

    for i in range(n):
        if(i == 0):
            Mi = TransInv(Mlist[i]) @ np.identity(4)
        else:
            Mi = TransInv(Mlist[i]) @ Mlist[i - 1]

        Ai[:, [i]] = Adjoint(TransInv(Mlist[i])) @ Slist[:, i:i+1]
        AdTi[i] = Adjoint(MatrixExp6(VecTose3(-Ai[:, [i]] * theta[i])) @ Mi)

        Vi[:, [i+1]] = AdTi[i] @ Vi[:, [i]] + Ai[:, [i]]*dtheta[i]
        Vdi[:, [i+1]] = AdTi[i] @ Vdi[:, [i]] + Ai[:, [i]]*ddtheta[i] + ad(Vi[:, [i+1]]) @ Ai[:, [i]] * dtheta[i]

    for i in range(n-1, -1, -1):
        Fi = AdTi[i+1].T @ Fi + Glist[i] @ Vdi[:, [i+1]] - ad(Vi[:, [i+1]]).T @ (Glist[i] @ Vi[:, [i+1]])

        tau[i, 0] = Fi.T @ Ai[:, [i]]

    return tau


def getInertiaMatrix(Inertialist):
    '''
    :param vec: [[],[],[],[],[],[]]
    :return:
    '''

    ixx = Inertialist[0][0]
    ixy = Inertialist[1][0]
    ixz = Inertialist[2][0]
    iyy = Inertialist[3][0]
    iyz = Inertialist[4][0]
    izz = Inertialist[5][0]

    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

    return I


def getMassMatrix_NewtonEuler(Mlist, Glist, Slist, thetalist):

    n = np.shape(thetalist)[0]
    M = np.zeros([n, n])

    for i in range(n):

        dthetalist = np.zeros([n, 1])
        ddthetalist = np.zeros([n, 1])
        ddthetalist[i] = 1
        M[:, [i]] = IDyn_NewtonEuler(Mlist, Glist, Slist, thetalist, dthetalist, ddthetalist)

    return M


def getVelQuadraticVector_NewtonEuler(Mlist, Glist, Slist, thetalist, dthetalist):

    n = np.shape(thetalist)[0]

    ddthetalist = np.zeros([n, 1])

    c = IDyn_NewtonEuler(Mlist, Glist, Slist, thetalist, dthetalist, ddthetalist)

    return c


def getGravityVector_NewtonEuler(Mlist, Glist, Slist, thetalist, g_vector):

    n = np.shape(thetalist)[0]

    dthetalist = np.zeros([n, 1])
    ddthetalist = np.zeros([n, 1])

    g = IDyn_NewtonEuler(Mlist, Glist, Slist, thetalist, dthetalist, ddthetalist, g_vector)

    return g


def getMassMatrix(robotId, g_vector, thetalist):

    n = np.shape(thetalist)[0]
    M = np.zeros([n, n])

    p.setGravity(0, 0, 0)

    for i in range(n):

        dthetalist = np.zeros([n, 1])
        ddthetalist = np.zeros([n, 1])
        ddthetalist[i] = 1

        M[:, [i]] = np.asarray(p.calculateInverseDynamics(bodyUniqueId=robotId,
                                               objPositions=thetalist.reshape(n).tolist(),
                                               objVelocities= dthetalist.reshape(n).tolist(),
                                               objAccelerations = ddthetalist.reshape(n).tolist())).reshape([n, 1])

    p.setGravity(g_vector[0], g_vector[1], g_vector[2])

    return M


def getVelQuadraticVector(robotId, g_vector, thetalist, dthetalist):

    n = np.shape(thetalist)[0]

    p.setGravity(0, 0, 0)

    ddthetalist = np.zeros([n, 1])

    c = np.asarray(p.calculateInverseDynamics(bodyUniqueId=robotId,
                                              objPositions=thetalist.reshape(n).tolist(),
                                              objVelocities= dthetalist.reshape(n).tolist(),
                                              objAccelerations = ddthetalist.reshape(n).tolist())).reshape([n, 1])

    p.setGravity(g_vector[0], g_vector[1], g_vector[2])

    return c


def getGravityVector(robotId, g_vector, thetalist):

    n = np.shape(thetalist)[0]

    dthetalist = np.zeros([n, 1])
    ddthetalist = np.zeros([n, 1])

    g = np.asarray(p.calculateInverseDynamics(bodyUniqueId=robotId,
                                              objPositions=thetalist.reshape(n).tolist(),
                                              objVelocities= dthetalist.reshape(n).tolist(),
                                              objAccelerations = ddthetalist.reshape(n).tolist())).reshape([n, 1])

    p.setGravity(g_vector[0], g_vector[1], g_vector[2])

    return g


def getDthetaPotential(theta, theta_min = -120 * pi/180, theta_max = 120 * pi/180):

    alpha = 0.1
    theta_m = (theta_max + theta_min)/2

    return 2 * alpha * (theta - theta_m) / (theta_max - theta_min)



