
import os
import sys
import time
import json
import yaml
from math import *

import numpy as np
import pinocchio as pin

from .robotics_utils import *
from .rotation_utils import *

class PinocchioModel:
    def __init__(self, urdf_dir, T_W0=None):

        # Import robot
        if T_W0 is None:
            T_W0 = np.identity(4)

        self._robot_name = os.path.basename(urdf_dir)
        self._robot_type = os.path.basename(os.path.dirname(urdf_dir))

        # Open YAML file
        with open(urdf_dir + "/../robot_configs.yaml".format(self._robot_type)) as yaml_file:
            self._robot_configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

        xyz = self._robot_configs[self._robot_name]["EndEffector"]["position"]
        rpy = self._robot_configs[self._robot_name]["EndEffector"]["orientation"]
        self._T_CoME = xyzeul2SE3(xyz, rpy, seq='XYZ', degree=True)

        # self.reset_base(base_pos, base_quat)

        self._T_W0 = T_W0
        self._Ad_W0 = Adjoint(self._T_W0)

        self.pinModel = pin.buildModelFromUrdf(urdf_dir + "/../{}/model.urdf".format(self._robot_type))
            
        self.pinData = self.pinModel.createData()
        self.numJoints = self.pinModel.nq

        pin.forwardKinematics(self.pinModel, self.pinData, np.zeros([self.numJoints, 1]))

    def reset_base(self, T_W0):

        self._T_W0 = T_W0
        self._Ad_W0 = Adjoint(self._T_W0)

    def FK(self, q):
        pin.forwardKinematics(self.pinModel, self.pinData, np.asarray(q).reshape([-1, 1]))
        return self._T_W0 @ self.pinData.oMi[self.numJoints].np @ self._T_CoME

    def _single_CLIK(self, oMdes, q, ql, qu):

        eps = 1e-4
        IT_MAX = 200
        DT = 1e-1
        damp = 1e-12

        for _ in range(IT_MAX):
            pin.forwardKinematics(self.pinModel, self.pinData, q)
            dMi = oMdes.actInv(self.pinData.oMi[self.numJoints])
            err = pin.log(dMi).vector

            if np.linalg.norm(err) < eps:
                q_check = np.mod(q + np.pi, 2 * np.pi) - np.pi
                if (True not in (q_check < ql)) and (True not in (q_check > qu)):
                    return q_check
                else:
                    return None

            J = pin.computeJointJacobian(self.pinModel, self.pinData, q, self.numJoints)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.pinModel, q, v * DT)

        return None

    def CLIK(self, T_goal, ql, qu, q_init=None, N_trials=10):
        # Closed-Loop Inverse Kinematics
        # q_init: np.ndarray (1d array)
        '''
        : return np.ndarray (1D) radian
        '''

        oMdes = pin.SE3(TransInv(self._T_W0) @ T_goal @ TransInv(self._T_CoME))

        ql = np.asarray(ql).reshape(-1)
        qu = np.asarray(qu).reshape(-1)

        if q_init is None:
            q_init = pin.randomConfiguration(self.pinModel, ql, qu)

        for _ in range(N_trials):

            q_res = self._single_CLIK(oMdes, q_init, ql, qu)
            if q_res is not None:
                return q_res

            q_init = pin.randomConfiguration(self.pinModel, ql, qu)
        return None

    def Js(self, q):
        pin.forwardKinematics(self.pinModel, self.pinData, np.asarray(q).reshape([-1, 1]))
        J = pin.computeJointJacobians(self.pinModel, self.pinData)
        return self._Ad_W0 @ J[[3, 4, 5, 0, 1, 2], :]

    def Jb(self, q):
        return Adjoint(TransInv(self.FK(q))) @ self.Js(q)

    def Js_dot(self, q):
        pass

    def Jb_dot(self, q):
        pass

    def Minv(self, q):
        return pin.computeMinverse(self.pinModel, self.pinData, np.asarray(q).reshape([-1, 1]))

    def M(self, q):
        return np.linalg.inv(self.Minv(q))

    def C(self, q, qdot):
        return pin.computeCoriolisMatrix(self.pinModel, self.pinData,
                                         np.asarray(q).reshape([-1, 1]), np.asarray(qdot).reshape([-1, 1]))

    def g(self, q):
        return pin.computeGeneralizedGravity(self.pinModel, self.pinData,
                                             np.asarray(q).reshape([-1, 1])).reshape([-1, 1])