# Sensors Start
import numpy as np
import matplotlib.pyplot as plt
import ir_support
import swift
import mujoco
import mujoco.viewer
from spatialmath.base import *
from spatialmath import SE3
from roboticstoolbox import DHLink, DHRobot, models
from ir_support import CylindricalDHRobotPlot
from spatialgeometry import Sphere, Arrow, Mesh
from roboticstoolbox import DHLink, DHRobot, models
from math import pi
from roboticstoolbox import DHLink, DHRobot, models, jtraj
from ir_support_extra_robots.robots import Turtlebot3Waffle
from ir_support_extra_parts import part_mesh
from ir_support import UR3e
from spatialgeometry import Mesh
from pathlib import Path


# Student ID parameters
q = np.zeros([1, 6])  # changing position of the arm
armTravel = [0, -pi/2, 0, pi/2, 0, 0]
steps = 50


# Starting position situated ontop of the turtlebot
ur3StartPos = SE3(-0.9, 0, 0.5) * SE3.Rz(pi/2)

# Visualize each link as a cylinder

# Implementation of the bot and starting position
ur3Bot = UR3e()
ur3Bot.base = ur3StartPos


# Implementation of props
thisFolder = Path(__file__).resolve().parent
camOffset = transl(0, -0.05, 0)
camera = Mesh(filename=str(thisFolder / "d435.dae"),
              pose=(ur3Bot.fkine(q).A @ camOffset))
table = part_mesh("tableBrown2.1x1.4x0.5m", pose=SE3(0, 0, 0))
objectGoal_Location = transl(-0.8, -0.3, 0.5)
objectGoal = part_mesh("CardboardBox", pose=objectGoal_Location)
objectPickUP_Location = transl(-0.5, 0.1, 0.5)
objectPickUP = part_mesh("Apple", pose=objectPickUP_Location)
objectPickUP.base = SE3(objectPickUP_Location)

# repositioning
ur3Bot.q = [0, -pi/2, 0, pi/2, 0, 0]
Offset = trotx(pi) @ transl(0, 0, -0.05)
camera.T = ur3Bot.fkine(q).A @ camOffset

# Create Swift environment
env = swift.Swift()
env.launch(realtime=True)
env.add(table)
env.add(camera)
env.add(objectGoal)
env.add(objectPickUP)
ur3Bot.add_to_env(env)

env.set_camera_pose([-1, 1, 2.3], [0, 0, -0.2])
env.step()


class studentBotOperation():

    def __init__(self):
        global objectPickUP_Location
        self.ur3PickUP()

    def ur3PickUP(self):
        global objectPickUP_Location
        start_q = ur3Bot.q
        sol = ur3Bot.ikine_LM((objectPickUP_Location @ trotx(pi) @
                              transl(0, 0, -0.05)), q0=ur3Bot.q, mask=[1, 1, 1, 1, 1, 1])
        goal_q = sol.q
        if not sol.success or not check_joint_limits(self, goal_q, ur3Bot):
            print("Joints limits broken, can't pick up object properly")
        else:

            joint_trajectory = jtraj(
                start_q, goal_q, steps).q  # From lab solution 3

            for q in joint_trajectory:
                ur3Bot.q = q
                camera.T = ur3Bot.fkine(q).A @ camOffset
                env.step(0.05)

            # arm reset
            start_q = ur3Bot.q
            goal_q = [0, -pi/2, 0, pi/2, 0, 0]
            if not sol.success or not check_joint_limits(self, goal_q, ur3Bot):
                print("Joints limits broken, can't pick up object properly")
            else:
                joint_trajectory = jtraj(start_q, goal_q, steps).q

                for q in joint_trajectory:
                    ur3Bot.q = q
                    objectPickUP.T = ur3Bot.fkine(
                        q).A  @ trotx(pi) @ transl(0, 0, -0.05)
                    camera.T = ur3Bot.fkine(q).A @ camOffset
                    env.step(0.05)

                objectPickUP_Location = objectPickUP.T
                self.deposit()

    def deposit(self):
        start_q = ur3Bot.q
        objectDrop = objectGoal_Location
        sol = ur3Bot.ikine_LM(
            (objectDrop @ trotx(pi) @ transl(0, 0, -0.05)), q0=ur3Bot.q, mask=[1, 1, 1, 1, 1, 1])
        goal_q = sol.q
        if not sol.success or not check_joint_limits(self, goal_q, ur3Bot):
            print("Joints limits broken, can't pick up object properly")
        else:

            joint_trajectory = jtraj(
                start_q, goal_q, steps).q  # From lab solution 3

            for q in joint_trajectory:
                ur3Bot.q = q
                env.step(0.05)
                objectPickUP.T = ur3Bot.fkine(q).A  @ Offset
                camera.T = ur3Bot.fkine(q).A @ camOffset

            # arm reset
            start_q = ur3Bot.q
            goal_q = [0, -pi/2, 0, pi/2, 0, 0]
            if not sol.success or not check_joint_limits(self, goal_q, ur3Bot):
                print("Joints limits broken, can't pick up object properly")
            else:
                joint_trajectory = jtraj(start_q, goal_q, steps).q

                for q in joint_trajectory:
                    ur3Bot.q = q
                    camera.T = ur3Bot.fkine(q).A @ camOffset
                    env.step(0.05)


def check_joint_limits(self, Q, robot):  # extracted from lab solution 4
    success = True
    # Checks rows of q (joint configs) to ensure its within joint limits
    Q = np.atleast_2d(Q)
    qlim = robot.qlim  # shape (2, n), where row 0 = lower, row 1 = upper

    for i, q in enumerate(Q):
        # checks upper values of the qlim assigned earlier
        lower_violation = q < qlim[0]
        upper_violation = q > qlim[1]

        if np.any(lower_violation) or np.any(upper_violation):
            print(f"q[{i}] is out of joint limits:")
            for j in range(len(q)):
                if lower_violation[j]:
                    # prints out error through arrays
                    print(
                        f"  Joint {j}: {q[j]:.3f} < lower limit {qlim[0, j]:.3f}")
                elif upper_violation[j]:
                    print(
                        f"  Joint {j}: {q[j]:.3f} > upper limit {qlim[1, j]:.3f}")
            success = False

    return success  # boolean output


studentBotOperation()

env.hold()
