import logging

import numpy as np

from .skeleton_model import SkeletonModel, Joint

logger = logging.getLogger(__name__)


class JointAngleSolver:
    def __init__(self, skeleton: SkeletonModel) -> None:
        self._skeleton = skeleton

    def solve(
        self,
        keypoints_3d: np.ndarray,
        joint_names: list[str],
    ) -> dict[str, np.ndarray]:
        if keypoints_3d is None or len(keypoints_3d) == 0:
            return {}

        positions: dict[str, np.ndarray] = {}
        for i, name in enumerate(joint_names):
            if i < len(keypoints_3d):
                if not np.any(np.isnan(keypoints_3d[i])):
                    positions[name] = keypoints_3d[i]

        rotations: dict[str, np.ndarray] = {}

        for joint_name in self._skeleton.joint_names:
            joint = self._skeleton.joints[joint_name]

            if joint.parent is None:
                rotations[joint_name] = np.eye(3)
                continue

            if joint_name not in positions or joint.parent not in positions:
                rotations[joint_name] = np.eye(3)
                continue

            child_pos = positions[joint_name]
            parent_pos = positions[joint.parent]

            bone_vector = child_pos - parent_pos
            bone_length = np.linalg.norm(bone_vector)

            if bone_length < 1e-6:
                rotations[joint_name] = np.eye(3)
                continue

            bone_dir = bone_vector / bone_length
            rest_dir = self._get_rest_direction(joint_name)

            R = self._rotation_between_vectors(rest_dir, bone_dir)
            rotations[joint_name] = R

        root_name = self._skeleton.root
        if root_name and root_name in positions:
            root_pos = positions[root_name]
        else:
            root_pos = np.zeros(3)

        return self._euler_from_rotations(rotations, root_pos)

    @staticmethod
    def _get_rest_direction(joint_name: str) -> np.ndarray:
        if "Leg" in joint_name or "Foot" in joint_name:
            return np.array([0.0, -1.0, 0.0])
        if joint_name.startswith("Left") or ("Arm" in joint_name and "Left" in joint_name):
            return np.array([1.0, 0.0, 0.0])
        if joint_name.startswith("Right") or ("Arm" in joint_name and "Right" in joint_name):
            return np.array([-1.0, 0.0, 0.0])
        return np.array([0.0, 1.0, 0.0])

    def _rotation_between_vectors(
        self, v1: np.ndarray, v2: np.ndarray
    ) -> np.ndarray:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)

        if dot > 0.9999:
            return np.eye(3)

        if dot < -0.9999:
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(v1, perp)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            perp = perp - np.dot(perp, v1) * v1
            perp /= np.linalg.norm(perp)
            return self._rotation_matrix(perp, np.pi)

        axis = np.cross(v1, v2)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)

        return self._rotation_matrix(axis, angle)

    @staticmethod
    def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis

        return np.array([
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ])

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        if sy < 1e-6:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0.0
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])

        return np.degrees(np.array([x, y, z]))

    def _euler_from_rotations(
        self,
        rotations: dict[str, np.ndarray],
        root_position: np.ndarray,
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        joint_order = self._skeleton.get_joint_order()

        for name in joint_order:
            if name not in rotations:
                continue

            joint = self._skeleton.joints[name]
            R = rotations[name]

            euler = self._rotation_matrix_to_euler(R)

            channel_count = len(joint.channels)
            if channel_count == 6:
                result[name] = np.concatenate([root_position, euler])
            else:
                result[name] = euler

        return result
