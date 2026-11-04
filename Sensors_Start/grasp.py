import numpy as np
import mujoco


class KinematicGrasp:
    """
    Temporary grasp substitute.

    The current UR3e model has no gripper or equality weld. This class
    attaches the free target to the attachment site kinematically.
    It is a demonstration placeholder, not a physical gripper model.
    """

    def __init__(self, model, data, target_joint_id, site_id):
        self.model = model
        self.data = data
        self.target_joint_id = target_joint_id
        self.site_id = site_id
        self.attached = False

        self.qadr = int(model.jnt_qposadr[target_joint_id])
        self.dadr = int(model.jnt_dofadr[target_joint_id])

    def attach(self):
        mujoco.mj_forward(self.model, self.data)

        site_pos = self.data.site_xpos[self.site_id].copy()
        site_mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()

        target_pos = self.data.qpos[self.qadr:self.qadr + 3].copy()
        target_quat = self.data.qpos[self.qadr + 3:self.qadr + 7].copy()

        self.relative_position = np.array([0, 0, 0.03]) # makes the cube 3cm below the EF

        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, site_mat.flatten()) #mat2 converts 3x3 rotation matrix into a quaternion

        inverse_site_quat = np.zeros(4)
        mujoco.mju_negQuat(inverse_site_quat, site_quat)

        self.relative_quat = np.zeros(4)
        mujoco.mju_mulQuat(
            self.relative_quat,
            inverse_site_quat,
            target_quat
        )

        self.attached = True
        print("Kinematic grasp activated (placeholder).")

    def update(self):
        if not self.attached:
            return

        mujoco.mj_forward(self.model, self.data)

        site_pos = self.data.site_xpos[self.site_id].copy()
        site_mat = self.data.site_xmat[self.site_id].reshape(3, 3).copy()

        target_pos = site_pos + site_mat @ self.relative_position

        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, site_mat.flatten())

        target_quat = np.zeros(4)
        mujoco.mju_mulQuat(
            target_quat,
            site_quat,
            self.relative_quat
        )

        self.data.qpos[self.qadr:self.qadr + 3] = target_pos
        self.data.qpos[self.qadr + 3:self.qadr + 7] = target_quat
        self.data.qvel[self.dadr:self.dadr + 6] = 0
