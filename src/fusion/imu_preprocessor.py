from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..acquisition.imu_manager import IMUData


@dataclass
class ProcessedIMU:
    sensor_id: int
    timestamp_ns: int
    quaternion: np.ndarray
    linear_accel: np.ndarray
    angular_vel: np.ndarray
    global_accel: np.ndarray


class IMUPreprocessor:
    def __init__(self, num_sensors: int = 8) -> None:
        self._num_sensors = num_sensors
        self._bias_offsets: dict[int, np.ndarray] = {}
        self._calibrated = False
        self._gravity = np.array([0.0, -9.81, 0.0])

    def calibrate_bias(
        self, samples: dict[int, list[IMUData]], static_duration_s: float = 2.0
    ) -> None:
        for sid, data_list in samples.items():
            if not data_list:
                continue

            accel_sum = np.zeros(3)
            count = 0

            for d in data_list:
                accel_sum += d.linear_accel
                count += 1

            if count > 0:
                self._bias_offsets[sid] = accel_sum / count - self._gravity

        self._calibrated = True

    def process(self, data: IMUData) -> Optional[ProcessedIMU]:
        if data.quaternion is None or data.linear_accel is None:
            return None

        q = data.quaternion.copy()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return None
        q /= q_norm

        accel = data.linear_accel.copy()
        gyro = data.angular_vel.copy() if data.angular_vel is not None else np.zeros(3)

        if self._calibrated and data.sensor_id in self._bias_offsets:
            accel -= self._bias_offsets[data.sensor_id]

        R = self._quaternion_to_rotation_matrix(q)
        global_accel = R @ accel

        return ProcessedIMU(
            sensor_id=data.sensor_id,
            timestamp_ns=data.timestamp_ns,
            quaternion=q,
            linear_accel=accel,
            angular_vel=gyro,
            global_accel=global_accel,
        )

    def process_batch(
        self, data_list: list[IMUData]
    ) -> list[ProcessedIMU]:
        results = []
        for d in data_list:
            processed = self.process(d)
            if processed is not None:
                results.append(processed)
        return results

    @staticmethod
    def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q

        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        dot = np.dot(q1, q2)

        if dot < 0.0:
            q2 = -q2
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)

        if theta < 1e-6:
            return q1

        sin_theta = np.sin(theta)
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        return w1 * q1 + w2 * q2
