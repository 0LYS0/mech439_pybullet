"""
PybulletCore
~~~~~~~~~~~~~
"""

import os
import sys
import time
from math import *

import numpy as np
import matplotlib.pyplot as plt

from threading import Thread

import pybullet as p
import pybullet_data

from src.utils import *
from src.core.pybullet_robot import PybulletRobot

class PybulletCore:
    """Pybullet Simulator Core Class

    :param string scene_info: filename of scene configuration yaml file
    """
    def __init__(self):

        np.set_printoptions(precision=3)

        # Simulator configuration
        self.__filepath = os.path.dirname(os.path.abspath(__file__))
        self.__cfgpath  = self.__filepath + "/../configs"
        self.__urdfpath = self.__filepath + "/../assets/urdf"

        self.startPosition = [0, 0, 0] ## base position
        self.startOrientation = [0, 0, 0] ## base orientation

        self.g_vector = np.array([0, 0, -9.81]).reshape([3, 1])

        self.dt = 1. / 240  # Simulation Frequency




    def connect(self, robot_name = 'indyRP2', joint_limit = True, constraint_visualization = True):
        """
        Connect to Pybullet GUI
        """

        # Open GUI
        self.ClientId = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)


        # Set perspective camera
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])

        p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0.3, 0, 0], lineColorRGB=[1, 0, 0],
                           lineWidth=5, physicsClientId=self.ClientId)
        p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0.3, 0], lineColorRGB=[0, 1, 0],
                           lineWidth=5, physicsClientId=self.ClientId)
        p.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, 0.3], lineColorRGB=[0, 0, 1],
                           lineWidth=5, physicsClientId=self.ClientId)

        # Set physics scene
        p.setGravity(self.g_vector[0], self.g_vector[1], self.g_vector[2]) # set gravity
        p.performCollisionDetection()
        p.setTimeStep(self.dt)

        # Add plane
        self.planeId = p.loadURDF("plane.urdf")

        # Define robot's information
        robot_info = {"robot_name":None, "robot_position":None, "robot_orientation":None, "robot_properties":{}}
        robot_info["robot_name"] = robot_name
        robot_info["robot_position"] = self.startPosition
        robot_info["robot_orientation"] = self.startOrientation
        robot_info["robot_properties"]["joint_limit"] = joint_limit
        robot_info["robot_properties"]["constraint_visualization"] = constraint_visualization

        # Import robot
        self.my_robot = PybulletRobot(ClientId=self.ClientId, robot_info=robot_info, dt=self.dt)

        # Run core thread
        self.__isSimulation = False
        self._thread = Thread(target=self._thread_main)
        self._thread.start()

        # Start simulation
        self.__isSimulation = True

    def disconnect(self):
        """
        Disconnect to Pybullet GUI
        """

        self.__isSimulation = False
        time.sleep(1)
        p.disconnect(physicsClientId=self.ClientId)
        PRINT_BLUE("Disconnect Success!")

    def _thread_main(self):
        """
        Disconnect to Pybullet GUI
        """
        while True:
            ts = time.time()
            if self.__isSimulation:
                self._thread_pre()

                self.my_robot.robot_update()
                
                self._thread_post()

                p.stepSimulation()

            tf = time.time()
            if tf-ts < self.dt:
                time.sleep(self.dt-tf+ts)

    def _thread_pre(self):
        pass

    def _thread_post(self):
        pass
    
    # For jupyter notebook
    def MoveRobot(self, angle, degrees=True, verbose=False):
        
        if degrees:
            angle = deg2radlist(angle)

        self.my_robot.reset_joint_pos(angle)

        if (verbose == True):
            PRINT_BLUE("***** Set desired joint angle *****")
            print(np.asarray(angle).reshape(-1))
    