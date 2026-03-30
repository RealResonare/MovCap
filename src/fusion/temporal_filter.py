import logging

import numpy as np
import yaml
from scipy.signal import savgol_filter

from .ukf_fusion import FusedPose

logger = logging.getLogger(__name__)


class TemporalFilter:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        filter_cfg = cfg["temporal_filter"]
        self._type: str = filter_cfg["type"]
        self._window_length: int = int(filter_cfg["window_length"])
        self._polyorder: int = int(filter_cfg["polyorder"])

        if self._window_length % 2 == 0:
            self._window_length += 1
        if self._polyorder >= self._window_length:
            self._polyorder = self._window_length - 1

        self._buffer: list[FusedPose] = []
        self._max_buffer = self._window_length * 3

    def add(self, pose: FusedPose) -> None:
        self._buffer.append(pose)
        if len(self._buffer) > self._max_buffer:
            self._buffer = self._buffer[-self._max_buffer:]

    def get_filtered(self) -> FusedPose | None:
        if not self._buffer:
            return None

        if len(self._buffer) < self._window_length:
            return self._buffer[-1]

        window = self._buffer[-self._window_length :]
        latest = self._buffer[-1]

        n_frames = len(window)
        n_joints = latest.keypoints_3d.shape[0]

        positions = np.zeros((n_frames, n_joints, 3))
        for i, pose in enumerate(window):
            positions[i] = pose.keypoints_3d

        filtered_positions = np.zeros_like(positions)
        for j in range(n_joints):
            for dim in range(3):
                signal = positions[:, j, dim]
                filtered_positions[:, j, dim] = savgol_filter(
                    signal, self._window_length, self._polyorder
                )

        return FusedPose(
            keypoints_3d=filtered_positions[-1],
            velocities=latest.velocities,
            confidence=latest.confidence,
            timestamp_ns=latest.timestamp_ns,
        )

    def smooth_sequence(
        self, poses: list[FusedPose]
    ) -> list[FusedPose]:
        if len(poses) < self._window_length:
            return poses

        n_frames = len(poses)
        n_joints = poses[0].keypoints_3d.shape[0]

        positions = np.zeros((n_frames, n_joints, 3))
        for i, pose in enumerate(poses):
            positions[i] = pose.keypoints_3d

        filtered = np.zeros_like(positions)
        wl = min(self._window_length, n_frames if n_frames % 2 == 1 else n_frames - 1)

        for j in range(n_joints):
            for dim in range(3):
                filtered[:, j, dim] = savgol_filter(
                    positions[:, j, dim], wl, self._polyorder
                )

        result = []
        for i, pose in enumerate(poses):
            result.append(
                FusedPose(
                    keypoints_3d=filtered[i],
                    velocities=pose.velocities,
                    confidence=pose.confidence,
                    timestamp_ns=pose.timestamp_ns,
                )
            )

        return result

    def reset(self) -> None:
        self._buffer.clear()
