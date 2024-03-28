
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from math import *
from threading import Thread

import pybullet as p
import pybullet_data

from RoboticsToolbox import *

CTRLMODE_GRAVITY_COMPENSATION = 0
CTRLMODE_JOINTSPACE_PD = 1
CTRLMODE_JOINTSPACE_HINF = 2

class pybullet_core:

    def __init__(self):
        '''
        #############################################################
        PYBULLET INITIALIZATION
        #############################################################
        '''
        ###
        self.__WhiteText = "\033[37m"
        self.__BlackText = "\033[30m"
        self.__RedText = "\033[31m"
        self.__BlueText = "\033[34m"

        self.__DefaultText = "\033[0m"
        self.__BoldText = "\033[1m"

        np.set_printoptions(precision=3)

        ### Simulator configuration

        self.__filepath = os.getcwd()

        self.startPosition = [0, 0, 0] ## base position
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0]) ## base orientation

        self.g_vector = np.array([0, 0, -9.81]).reshape([3, 1])

        self.dt = 1. / 240  # Simulation Frequency
        # self.dt = 1./1000  # Simulation Frequency



    def connect_pybullet(self, robot_name = 'IndyRP2', joint_limit = True, constraint_visualization = True):
        '''
        #############################################################
        PYBULLET CONNECTION
        #############################################################
        '''

        ### Open GUI
        self.physicsClient = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)


        ### Set perspective camera
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])


        ### Set physics scene
        p.setGravity(self.g_vector[0], self.g_vector[1], self.g_vector[2]) # set gravity
        p.performCollisionDetection()
        p.setTimeStep(self.dt)
        self._constraint_visualization = constraint_visualization


        ### Add plane
        # self.planeId = p.loadURDF("plane.urdf")


        ### Add robot from URDF
        flags = p.URDF_USE_INERTIA_FROM_FILE
        # flag = p.URDF_USE_INERTIA_FROM_FILE + p.URDF_USE_SELF_COLLISION + p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self._load_robot(robot_name=robot_name, flags=flags)


        ### Get robot's info
        self.numJoint = p.getNumJoints(self.robotId)
        if joint_limit == False:
            for idx in range(self.numJoint):
                p.changeDynamics(self.robotId, idx, jointLowerLimit=-314, jointUpperLimit=314)

        self._get_robot_info()

        print(self.__BoldText + self.__BlueText + "****** LOAD SUCCESS ******" + self.__DefaultText + self.__BlackText)
        print(self.__BoldText + "Robot name" + self.__DefaultText + ": {}".format(robot_name))
        print(self.__BoldText + "DOF" + self.__DefaultText + ": {}".format(self.numJoint))
        print(self.__BoldText + "Joint limit" + self.__DefaultText + ": {}".format(joint_limit))


        ### Set control mode
        self._control_mode = CTRLMODE_JOINTSPACE_PD


        ### Run core thread
        self.__isSimulation = False
        self._thread = Thread(target=self._coreThread)
        self._thread.start()


        ### Object ID buffer
        self._obstacle_buff = []
        self._waypoint_buff = []
        self._approach_buff = []


        ### Add visualshape ID
        self.visualShapeId_1 = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 0.5])
        self.visualShapeId_2 = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.005, rgbaColor=[0, 0, 1, 0.3])
        self.visualShapeId_3 = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.7])


        ### Start simulation
        self.__isSimulation = True


    def disconnect_pybullet(self):
        '''
        #############################################################
        PYBULLET DISCONNECTION
        #############################################################
        '''

        self.__isSimulation = False
        time.sleep(1)
        p.disconnect()
        print(self.__BoldText + self.__BlueText + "Disconnect Success!" + self.__DefaultText + self.__BlackText)


    ### Import robot
    def _load_robot(self, robot_name, flags):

        if robot_name == 'Indy7': # Indy7 (6DOF)

            self.robotId = p.loadURDF(self.__filepath + "/urdf/indy7/indy7.urdf",
                                      basePosition=self.startPosition,
                                      baseOrientation=self.startOrientation,
                                      flags=flags
                                      )
        elif robot_name == 'IndyRP2': # IndyRP2 (7DOF)

            self.robotId = p.loadURDF(self.__filepath + "/urdf/indyRP2/indyrp2.urdf",
                                      basePosition=self.startPosition,
                                      baseOrientation=self.startOrientation,
                                      flags = flags
                                      )
        else:
            print(
                self.__BoldText + self.__RedText + "There are no available robot: {}".format(robot_name) + self.__DefaultText + self.__BlackText)
            return


    ### Get robot's information
    def _get_robot_info(self):
        '''
        #############################################################
        CALCULATE BODY SCREWS & SPATIAL SCREW
        #############################################################
        ## T_0i : Configuration of i-th link-frame in base coordinates (Joint i's configuration)
        ## Mlist[i] : Configuration of i-th link's CoM in base coordinates (CoM i's configuration), Mlist[-1] is configuration of end-effecter
        ## T_LCoM : Configuration of last link's CoM in lask link-frame coordinates
        ## T_CoME : Configuration of end-effector in last link's CoM coordinates
        ## T_0E : Configuration on end-effector in base coordinates
        '''

        self.robot = {}
        self.robot['q'] = np.zeros([self.numJoint, 1])
        self.robot['qdot'] = np.zeros([self.numJoint, 1])

        self.robot['q_des'] = np.zeros([self.numJoint, 1])
        self.robot['qdot_des'] = np.zeros([self.numJoint, 1])
        self.robot['qddot_des'] = np.zeros([self.numJoint, 1])

        self.robot['J'] = np.zeros([6, self.numJoint])
        self.robot['Jdot'] = np.zeros([6, self.numJoint]) ## need to compute!
        self.robot['M'] = np.zeros([self.numJoint, self.numJoint])
        self.robot['C'] = np.zeros([self.numJoint, self.numJoint]) ## need to compute!
        self.robot['c'] = np.zeros([self.numJoint, 1])
        self.robot['g'] = np.zeros([self.numJoint, 1])
        self.robot['tau'] = np.zeros([self.numJoint, 1])

        self.robot['p'] = np.zeros([6, 1])
        self.robot['pdot'] = np.zeros([6, 1])

        self.eint = 0


        linkStates = p.getLinkStates(bodyUniqueId=self.robotId, linkIndices=range(self.numJoint))

        self.T_0i = np.zeros([self.numJoint, 4, 4])
        self.Mlist = np.zeros([self.numJoint+1, 4, 4])


        for i in range(self.numJoint):
            self.T_0i[i] = xyzrpyToSE3(linkStates[i][4], p.getEulerFromQuaternion(linkStates[i][5]))
            self.Mlist[i] = xyzrpyToSE3(linkStates[i][0], p.getEulerFromQuaternion(linkStates[i][1]))


        ## T_LCoM : configuration of last link's {CoM Frame} in last link's {Joint Frame}
        ## T_CoME : configuration of {End-Effector Frame} in last link's {CoM Frame}
        self.T_LCoM = xyzrpyToSE3(linkStates[self.numJoint-1][2], p.getEulerFromQuaternion(linkStates[self.numJoint-1][3]))
        self.T_CoME = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.065], [0, 0, 0, 1]])

        self.T_0E = self.T_0i[-1, :, :] @ self.T_LCoM @ self.T_CoME
        self.Mlist[-1] = self.Mlist[-2] @ self.T_CoME


        self.Blist = np.zeros([6, self.numJoint]) ## Body screw list
        for i in range(self.numJoint):
            self.Blist[:, i] = calculate_body_screw(self.T_0E, self.T_0i[i], 'z')
        self.Slist = Adjoint(self.T_0E) @ self.Blist ## Spatial screw list


        ### Constraint & flag
        self._jointpos_lower = [0 for _ in range(self.numJoint)]
        self._jointpos_upper = [0 for _ in range(self.numJoint)]
        self._jointvel = [0 for _ in range(self.numJoint)]
        self._jointforce = [0 for _ in range(self.numJoint)]

        self._jointpos_flag = [0 for _ in range(self.numJoint)]
        self._jointvel_flag = [0 for _ in range(self.numJoint)]
        self._jointforce_flag = [0 for _ in range(self.numJoint)]
        self._collision_flag = [0 for _ in range(self.numJoint)]


        ### Get joint constraints
        for idx in range(self.numJoint):
            jointInfo = p.getJointInfo(bodyUniqueId=self.robotId, jointIndex=idx)
            self._jointpos_lower[idx] = jointInfo[8]
            self._jointpos_upper[idx] = jointInfo[9]
            self._jointvel[idx] = jointInfo[10]
            self._jointforce[idx] = jointInfo[11]


        ### Add end-effector
        pos = self.T_0E[0:3, 3]
        ori = p.getQuaternionFromEuler(Rot2rpy(self.T_0E[0:3, 0:3]))
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.7])

        self._endID = p.createMultiBody(baseVisualShapeIndex=visualShapeId, basePosition=pos, baseOrientation=ori)


        ### Remove pybullet built-in position controller's effect
        p.setJointMotorControlArray(bodyUniqueId=self.robotId,
                                    jointIndices=range(self.numJoint),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=list(0 for _ in range(self.numJoint)),
                                    forces=list(0 for _ in range(self.numJoint)))


    ### Threads
    def _coreThread(self):

        while(1):

            if (self.__isSimulation == True):

                self._getRobotStates()
                self._controlRobot()
                p.stepSimulation()

            time.sleep(self.dt)


    def _getRobotStates(self):

        for idx in range(self.numJoint):
            states = p.getJointState(self.robotId, idx)

            self.robot['q'][idx, 0] = states[0]
            self.robot['qdot'][idx, 0] = states[1]

        self.robot['J'] = bodyJacobian(self.Blist, self.robot['q'])
        self.robot['Jdot'] = np.zeros([6, self.numJoint])

        self.robot['M'] = getMassMatrix(self.robotId, self.g_vector, self.robot['q'])
        self.robot['c'] = getVelQuadraticVector(self.robotId, self.g_vector, self.robot['q'], self.robot['qdot'])
        self.robot['g'] = getGravityVector(self.robotId, self.g_vector, self.robot['q'])

        T = FK(self.T_0E, self.Slist, self.robot['q'])
        self.robot['p'][0:3, 0] = Rot2rpy(T[0:3, 0:3])
        self.robot['p'][3:6, 0] = T[0:3, 3]
        self.robot['T'] = T

        p.resetBasePositionAndOrientation(
            bodyUniqueId=self._endID,
            posObj=self.robot['p'][3:6, 0],
            ornObj=p.getQuaternionFromEuler(self.robot['p'][0:3, 0])
        )


    def _controlRobot(self):

        if(self._constraint_visualization == True):
            self._constraint_check()
            self._constraint_visualizer()

        p.setJointMotorControlArray(bodyUniqueId=self.robotId,
                                    jointIndices=range(self.numJoint),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.robot['q_des'])


    ### Utils
    def _constraint_check(self):

        ### Joint position limit check
        for idx in range(self.numJoint):

            q = self.robot['q'][idx, 0]
            ql = self._jointpos_lower[idx]
            qu = self._jointpos_upper[idx]

            if q < ql or q > qu:
                if self._jointpos_flag[idx] == 0 or self._jointpos_flag[idx] == 1:
                    self._jointpos_flag[idx] = 2
                else:
                    self._jointpos_flag[idx] = 3
            else:
                if self._jointpos_flag[idx] == 2 or self._jointpos_flag[idx] == 3:
                    self._jointpos_flag[idx] = 0
                else:
                    self._jointpos_flag[idx] = 1

        ### Joint velocity limit check
        for idx in range(self.numJoint):

            qdot = self.robot['qdot'][idx, 0]

            if np.abs(qdot) > self._jointvel[idx]:
                if self._jointvel_flag[idx] == 0 or self._jointvel_flag[idx] == 1:
                    self._jointvel_flag[idx] = 2
                else:
                    self._jointvel_flag[idx] = 3
            else:
                if self._jointvel_flag[idx] == 2 or self._jointvel_flag[idx] == 3:
                    self._jointvel_flag[idx] = 0
                else:
                    self._jointvel_flag[idx] = 1

        ### Collision check
        for idx in range(self.numJoint):

            _contact_points_info = p.getContactPoints(bodyA=self.robotId, linkIndexA=idx)

            if len(_contact_points_info) != 0:
                if self._collision_flag[idx] == 0 or self._collision_flag[idx] == 1:
                    self._collision_flag[idx] = 2
                else:
                    self._collision_flag[idx] = 3
            else:
                if self._collision_flag[idx] == 2 or self._collision_flag[idx] == 3:
                    self._collision_flag[idx] = 0
                else:
                    self._collision_flag[idx] = 1


    def _constraint_visualizer(self):

        ### Joint position limit check
        for idx in range(self.numJoint):
            if self._jointpos_flag[idx] == 2:
                p.changeVisualShape(
                    objectUniqueId=self.robotId,
                    linkIndex=idx,
                    rgbaColor=[0.7, 0.7, 0, 1]
                )
            elif self._jointpos_flag[idx] == 0:
                p.changeVisualShape(
                    objectUniqueId=self.robotId,
                    linkIndex=idx,
                    rgbaColor=[1, 1, 1, 1]
                )

        ### Joint velocity limit check
        for idx in range(self.numJoint):

            if self._jointvel_flag[idx] == 2:
                p.changeVisualShape(
                    objectUniqueId=self.robotId,
                    linkIndex=idx,
                    rgbaColor=[0.7, 0, 0.7, 1]
                )
            elif self._jointvel_flag[idx] == 0:
                p.changeVisualShape(
                    objectUniqueId=self.robotId,
                    linkIndex=idx,
                    rgbaColor=[1, 1, 1, 1]
                )

        ### Collision check
        for idx in range(self.numJoint):

            if self._collision_flag[idx] == 2:
                p.changeVisualShape(
                    objectUniqueId=self.robotId,
                    linkIndex=idx,
                    rgbaColor=[0.7, 0, 0, 1]
                )
            elif self._collision_flag[idx] == 0:
                p.changeVisualShape(
                    objectUniqueId=self.robotId,
                    linkIndex=idx,
                    rgbaColor=[1, 1, 1, 1]
                )


    ### For jupyter notebook
    def MoveRobot(self, angle, verbose=False):

        self.robot['q_des'] = np.array(angle).reshape([self.numJoint, 1])

        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "Set desired joint angle: " + self.__DefaultText + self.__BlackText, end='')
            print(self.robot['q_des'].reshape([self.numJoint]))


    def AddCircleTrajectory(self, center, normal, radius, R, T, verbose=False):

        if len(self._waypoint_buff) != 0:
            for i in range(len(self._waypoint_buff)):
                if self._waypoint_buff[i][3] is not None:
                    p.removeBody(self._waypoint_buff[i][3])

            self._waypoint_buff = []

        N = int(T//self.dt)
        theta = np.linspace(0, 2*np.pi, N)

        traj = np.zeros([4, N])
        _normal = np.array([0, 0, 1]) + np.array(normal)/np.linalg.norm(normal)
        _normal = _normal/np.linalg.norm(_normal)



        for i in range(N):
            traj[0, i] = center[0] + radius * cos(theta[i])
            traj[1, i] = center[1] + radius * sin(theta[i])
            traj[2, i] = center[2]
            traj[3, i] = 1

        w = _normal
        v = -np.cross(_normal, np.array(center))
        T = MatrixExp6(VecTose3(np.array([w[0], w[1], w[2], v[0], v[1], v[2]]).reshape([6, 1])) * np.pi)

        traj = T@traj

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 1, 0.5])
        visualShapeId_s = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 0.5])
        for i in range(N):
            if (i%10 == 0):
                if i == 0:
                    WaypointId = p.createMultiBody(baseVisualShapeIndex=visualShapeId_s, basePosition=traj[0:3, i])
                else:
                    WaypointId = p.createMultiBody(baseVisualShapeIndex=visualShapeId, basePosition=traj[0:3, i])

            else:
                WaypointId = None

            self._waypoint_buff.append((i, traj[0:3, i], R, WaypointId))

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 0.5])
        ShapeId = p.createMultiBody(baseVisualShapeIndex=visualShapeId, basePosition=center)
        self._waypoint_buff.append((None, None, None, ShapeId))


        if (verbose == True):
            print(self.__BoldText + self.__BlueText + "****** ADD Trajectory ******" + self.__DefaultText + self.__BlackText)
            print(self.__BoldText + "Center" + self.__DefaultText + ": [{0}, {1}, {2}]".format(center[0], center[1], center[2]))
            print(self.__BoldText + "Normal" + self.__DefaultText + ": [{0}, {1}, {2}]".format(normal[0], normal[1], normal[2]))
            print(self.__BoldText + "Radius" + self.__DefaultText + ": {}".format(radius))


    def FollowingPath(self, mode=0, k=10):

        N = len(self._waypoint_buff)-1
        Tlist = np.zeros([4, 4, N])

        for i in range(N):
            Tlist[0:3, 0:3, i] = self._waypoint_buff[i][2]
            Tlist[0:3, 3, i] = self._waypoint_buff[i][1]
            Tlist[3, 3] = 1

        ### Generate approach path
        N_appr = int(2/self.dt)
        T_appr = np.zeros([4, 4, N_appr])
        T_curr = self.robot['T']

        T_start = Tlist[:,:, 0]

        if len(self._approach_buff) != 0:
            for i in range(len(self._approach_buff)):
                if self._approach_buff[i][2] is not None:
                    p.removeBody(self._approach_buff[i][2])
            self._approach_buff = []

        for i in range(N_appr):

            T_appr[:,:,i] = T_curr @ MatrixExp6(MatrixLog6(TransInv(T_curr)@T_start)*i/N_appr)
            if i%50 == 0:
                visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 0.5])
                ShapeId = p.createMultiBody(baseVisualShapeIndex=visualShapeId, basePosition=T_appr[0:3,3,i])
            else:
                ShapeId = None

            self._approach_buff.append((i, T_appr[:,:,i], ShapeId))


        ### Execute

        q = self.robot['q']
        q_list = q
        mani_list = []
        jp_list = []

        for i in range(N_appr-1):
            T_curr = FK(self.T_0E, self.Slist, q)
            Jb = bodyJacobian(self.Blist, q)
            J = Adjoint(T_curr) @ Jb
            Jinv = J.T@np.linalg.inv(J@J.T)

            mani = self.manipulability(q)
            mani_list.append(mani)
            jp = self.potential(q)
            jp_list.append(jp)

            V = se3ToVec((T_appr[:, :, i+1]-T_curr)/self.dt @ TransInv(T_appr[:, :, i+1]))

            if mode == 1:
                J1_temp = np.identity(self.numJoint) - Jinv @ J
                q = q + (Jinv @ V + J1_temp @ (k*self.dmanipulability(q).T)) * self.dt
            elif mode == 2:
                J1_temp = np.identity(self.numJoint) - Jinv @ J
                q = q + (Jinv @ V + J1_temp @ (-k * self.dpotential(q).T)) * self.dt
            elif mode == 3:
                J1_temp = np.identity(self.numJoint) - Jinv @ J
                J2 = self.dpotential(q)
                J2_hat = J2@J1_temp
                J2_hat_inv = J2_hat.T/np.linalg.norm(J2_hat)**2
                J2_temp = J1_temp @ (np.identity(self.numJoint) - J2_hat_inv @ J2_hat) @ J1_temp

                q = q + (Jinv @ V + J1_temp @ (-k[0] * self.dpotential(q).T) + J2_temp @ (k[1] * self.dmanipulability(q).T)) * self.dt
            else:
                q = q + Jinv @ V * self.dt

            q_list = np.concatenate((q_list, q), axis=1)
            self.MoveRobot(q.reshape(self.numJoint).tolist())
            time.sleep(self.dt)

        time.sleep(1)
        # q = self.robot['q']
        for i in range(N - 1):
            T_curr = FK(self.T_0E, self.Slist, q)
            Jb = bodyJacobian(self.Blist, q)
            J = Adjoint(T_curr) @ Jb
            Jinv = J.T @ np.linalg.inv(J @ J.T)

            mani = self.manipulability(q)
            mani_list.append(mani)
            jp = self.potential(q)
            jp_list.append(jp)

            V = se3ToVec((Tlist[:, :, i + 1] - T_curr) / self.dt @ TransInv(Tlist[:, :, i + 1]))

            if mode == 1:
                J1_temp = np.identity(self.numJoint) - Jinv @ J
                q = q + (Jinv @ V + J1_temp @ (k * self.dmanipulability(q).T)) * self.dt
            elif mode == 2:
                J1_temp = np.identity(self.numJoint) - Jinv @ J
                q = q + (Jinv @ V + J1_temp @ (-k * self.dpotential(q).T)) * self.dt
            elif mode == 3:
                J1_temp = np.identity(self.numJoint) - Jinv @ J
                J2 = self.dpotential(q)
                J2_hat = J2 @ J1_temp
                J2_hat_inv = J2_hat.T / np.linalg.norm(J2_hat) ** 2
                J2_temp = J1_temp @ (np.identity(self.numJoint) - J2_hat_inv @ J2_hat) @ J1_temp

                q = q + (Jinv @ V + J1_temp @ (-k[0] * self.dpotential(q).T) + J2_temp @ (
                            k[1] * self.dmanipulability(q).T)) * self.dt
            else:
                q = q + Jinv @ V * self.dt

            q_list = np.concatenate((q_list, q), axis=1)
            self.MoveRobot(q.reshape(self.numJoint).tolist())
            time.sleep(self.dt)

        mani = self.manipulability(q)
        mani_list.append(mani)
        jp = self.potential(q)
        jp_list.append(jp)

        return q_list, mani_list, jp_list


    def manipulability(self, q):
        Jb = bodyJacobian(self.Blist, q)
        mani = np.sqrt(np.linalg.det(Jb[3:6, :] @ Jb[3:6, :].T))
        return mani


    def dmanipulability(self, q, dq=1e-8):
        mani = self.manipulability(q)
        dmani = np.zeros([1, self.numJoint])

        for i in range(self.numJoint):
            q_ = q
            q_[i, 0] += dq
            mani_ = self.manipulability(q_)
            dmani[0, i] = (mani_ - mani)/dq

        return dmani


    def potential(self, q):
        res = 0
        for i in range(self.numJoint):
            ql = self._jointpos_lower[i]
            qu = self._jointpos_upper[i]
            res += 1/4*q[i, 0]**4

        return res

    def dpotential(self, q):
        res = np.zeros([1, self.numJoint])

        for i in range(self.numJoint):
            res[0, i] = q[i, 0]**3

        return res