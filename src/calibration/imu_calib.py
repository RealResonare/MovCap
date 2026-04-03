import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TPoseCalibration:
    sensor_id: int
    reference_quaternion: np.ndarray
    tpose_duration_s: float
    num_samples: int


class IMUTPoseCalibrator:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        imu_cfg = cfg["imu"]
        self._num_sensors: int = imu_cfg["count"]
        self._sample_rate: int = imu_cfg["sample_rate_hz"]
        self._hold_duration: float = 3.0
        self._calibrations: dict[int, TPoseCalibration] = {}

    @property
    def calibrations(self) -> dict[int, TPoseCalibration]:
        return self._calibrations

    @property
    def is_calibrated(self) -> bool:
        return len(self._calibrations) > 0

    @property
    def num_sensors(self) -> int:
        return self._num_sensors

    def collect_tpose(
        self,
        imu_read_fn,
        hold_duration: float = 3.0,
        progress_callback=None,
    ) -> dict[int, TPoseCalibration]:
        self._hold_duration = hold_duration
        self._calibrations.clear()

        expected_samples = int(self._sample_rate * hold_duration)
        collections: dict[int, list[np.ndarray]] = {i: [] for i in range(self._num_sensors)}

        if progress_callback:
            progress_callback("准备就绪", 0.0)

        start_time = time.time()

        while time.time() - start_time < hold_duration:
            elapsed = time.time() - start_time
            progress = elapsed / hold_duration

            imu_data = imu_read_fn()
            for sensor_id, data in imu_data.items():
                if data is not None and sensor_id in collections:
                    collections[sensor_id].append(data.quaternion.copy())

            if progress_callback:
                progress_callback(
                    f"T字形姿态采集中... {elapsed:.1f}/{hold_duration:.1f}s",
                    progress,
                )

            time.sleep(1.0 / self._sample_rate)

        for sensor_id, quats in collections.items():
            if len(quats) < expected_samples * 0.5:
                logger.warning(
                    "IMU %d: only %d/%d samples collected",
                    sensor_id, len(quats), expected_samples,
                )
                continue

            quat_array = np.array(quats)
            ref_quat = self._average_quaternions(quat_array)

            self._calibrations[sensor_id] = TPoseCalibration(
                sensor_id=sensor_id,
                reference_quaternion=ref_quat,
                tpose_duration_s=hold_duration,
                num_samples=len(quats),
            )

        if progress_callback:
            progress_callback(
                f"校准完成: {len(self._calibrations)}/{self._num_sensors} 传感器",
                1.0,
            )

        return self._calibrations

    @staticmethod
    def _average_quaternions(quaternions: np.ndarray) -> np.ndarray:
        if len(quaternions) == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])

        avg = np.mean(quaternions, axis=0)
        norm = np.linalg.norm(avg)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return avg / norm

    def get_correction_quaternion(self, sensor_id: int) -> Optional[np.ndarray]:
        cal = self._calibrations.get(sensor_id)
        if cal is None:
            return None
        return self._quaternion_inverse(cal.reference_quaternion)

    @staticmethod
    def _quaternion_inverse(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def apply_calibration(
        self, sensor_id: int, quaternion: np.ndarray
    ) -> np.ndarray:
        correction = self.get_correction_quaternion(sensor_id)
        if correction is None:
            return quaternion
        return self.quaternion_multiply(correction, quaternion)

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "num_sensors": self._num_sensors,
            "hold_duration": self._hold_duration,
            "calibrations": {},
        }

        for sensor_id, cal in self._calibrations.items():
            data["calibrations"][sensor_id] = {
                "sensor_id": cal.sensor_id,
                "reference_quaternion": cal.reference_quaternion.tolist(),
                "tpose_duration_s": cal.tpose_duration_s,
                "num_samples": cal.num_samples,
            }

        output_path = output_dir / "imu_tpose.yaml"
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info("IMU T-pose calibration saved to %s", output_path)

    def load(self, input_dir: str | Path) -> None:
        input_dir = Path(input_dir)
        path = input_dir / "imu_tpose.yaml"

        if not path.is_file():
            logger.warning("No IMU T-pose calibration found at %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._calibrations.clear()
        self._num_sensors = data.get("num_sensors", self._num_sensors)

        for sid_str, cal_data in data.get("calibrations", {}).items():
            sensor_id = int(sid_str)
            self._calibrations[sensor_id] = TPoseCalibration(
                sensor_id=sensor_id,
                reference_quaternion=np.array(
                    cal_data["reference_quaternion"], dtype=np.float64
                ),
                tpose_duration_s=cal_data["tpose_duration_s"],
                num_samples=cal_data["num_samples"],
            )

        logger.info(
            "IMU T-pose calibration loaded: %d sensors", len(self._calibrations)
        )
