import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from ..pose.pose3d_reconstructor import Pose3D
from .imu_preprocessor import ProcessedIMU

logger = logging.getLogger(__name__)


@dataclass
class FusedPose:
    keypoints_3d: np.ndarray       # shape (17, 3)
    velocities: np.ndarray         # shape (17, 3)
    confidence: np.ndarray         # shape (17,)
    timestamp_ns: int


class VisualIMUFusion:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        fusion_cfg = cfg["fusion"]
        self._process_noise_std = fusion_cfg["process_noise_std"]
        self._measurement_noise_std = fusion_cfg["measurement_noise_std"]
        self._imu_weight = fusion_cfg["imu_weight"]

        # State: 17 joints × (x, y, z, vx, vy, vz) = 102 dimensions
        # Using per-joint simplified UKF for tractability
        self._dim_x = 6   # per joint: [x, y, z, vx, vy, vz]
        self._dim_z = 3   # per joint: [x, y, z]

        self._joint_filters: dict[int, UnscentedKalmanFilter] = {}
        self._dt = 1.0 / 30.0  # default 30fps
        self._initialized = False

        self._create_filters()

    def _create_filters(self) -> None:
        from ..pose.pose2d_estimator import NUM_KEYPOINTS

        for j in range(NUM_KEYPOINTS):
            points = MerweScaledSigmaPoints(
                n=self._dim_x,
                alpha=0.1,
                beta=2.0,
                kappa=0.0,
            )

            ukf = UnscentedKalmanFilter(
                dim_x=self._dim_x,
                dim_z=self._dim_z,
                dt=self._dt,
                hx=self._measurement_function,
                fx=self._state_transition,
                points=points,
            )

            ukf.Q = np.eye(self._dim_x) * self._process_noise_std ** 2
            ukf.Q[3:, 3:] *= 10.0

            ukf.R = np.eye(self._dim_z) * self._measurement_noise_std ** 2

            ukf.x = np.zeros(self._dim_x)
            ukf.P = np.eye(self._dim_x) * 1.0

            self._joint_filters[j] = ukf

    @staticmethod
    def _state_transition(x: np.ndarray, dt: float) -> np.ndarray:
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        return F @ x

    @staticmethod
    def _measurement_function(x: np.ndarray) -> np.ndarray:
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])
        return H @ x

    def set_dt(self, dt: float) -> None:
        self._dt = dt
        for ukf in self._joint_filters.values():
            ukf.dt = dt

    def initialize(self, pose_3d: Pose3D) -> None:
        for j in range(len(self._joint_filters)):
            ukf = self._joint_filters[j]
            ukf.x[:3] = pose_3d.keypoints_3d[j]
            ukf.x[3:] = 0.0
            ukf.P = np.eye(self._dim_x) * 0.1

        self._initialized = True

    def predict(self) -> None:
        if not self._initialized:
            return

        for ukf in self._joint_filters.values():
            ukf.predict(fx_args=(self._dt,))

    def update_visual(self, pose_3d: Pose3D) -> None:
        if not self._initialized:
            self.initialize(pose_3d)
            return

        for j in range(len(self._joint_filters)):
            if pose_3d.confidence[j] > 0.3:
                ukf = self._joint_filters[j]
                noise_scale = 1.0 / max(pose_3d.confidence[j], 0.1)
                ukf.R = np.eye(self._dim_z) * self._measurement_noise_std ** 2 * noise_scale
                ukf.update(pose_3d.keypoints_3d[j])

    def update_imu(
        self,
        imu_data: dict[int, ProcessedIMU],
        joint_to_imu: dict[int, int],
    ) -> None:
        if not self._initialized:
            return

        for joint_idx, imu_idx in joint_to_imu.items():
            if joint_idx not in self._joint_filters:
                continue

            imu = imu_data.get(imu_idx)
            if imu is None:
                continue

            ukf = self._joint_filters[joint_idx]

            accel = imu.global_accel
            delta_v = accel * self._dt
            delta_pos = 0.5 * accel * self._dt ** 2

            ukf.x[:3] += delta_pos * self._imu_weight
            ukf.x[3:] += delta_v * self._imu_weight

    def get_fused_pose(self, timestamp_ns: int) -> FusedPose:
        num_joints = len(self._joint_filters)
        keypoints = np.zeros((num_joints, 3))
        velocities = np.zeros((num_joints, 3))
        confidence = np.zeros(num_joints)

        for j, ukf in self._joint_filters.items():
            keypoints[j] = ukf.x[:3]
            velocities[j] = ukf.x[3:]

            trace_P = np.trace(ukf.P[:3, :3])
            confidence[j] = max(0.0, 1.0 - trace_P / 0.1)

        return FusedPose(
            keypoints_3d=keypoints,
            velocities=velocities,
            confidence=confidence,
            timestamp_ns=timestamp_ns,
        )

    def fuse_step(
        self,
        pose_3d: Optional[Pose3D],
        imu_data: dict[int, ProcessedIMU],
        joint_to_imu: dict[int, int],
        timestamp_ns: int,
    ) -> FusedPose:
        self.predict()

        if pose_3d is not None:
            self.update_visual(pose_3d)

        self.update_imu(imu_data, joint_to_imu)

        return self.get_fused_pose(timestamp_ns)
