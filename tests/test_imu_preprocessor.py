import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.imu_preprocessor import IMUPreprocessor, ProcessedIMU
from src.acquisition.imu_manager import IMUData


class TestIMUPreprocessor:
    def test_process_valid_data(self):
        preprocessor = IMUPreprocessor()
        data = IMUData(
            sensor_id=0,
            timestamp_ns=1_000_000,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_accel=np.array([0.0, 0.0, 9.81]),
            angular_vel=np.array([0.1, 0.2, 0.3]),
        )
        result = preprocessor.process(data)
        assert result is not None
        assert result.sensor_id == 0
        assert result.quaternion.shape == (4,)
        assert result.global_accel.shape == (3,)

    def test_process_none_quaternion(self):
        preprocessor = IMUPreprocessor()
        data = IMUData(
            sensor_id=0,
            timestamp_ns=0,
            quaternion=None,
            linear_accel=np.zeros(3),
            angular_vel=np.zeros(3),
        )
        result = preprocessor.process(data)
        assert result is None

    def test_process_none_accel(self):
        preprocessor = IMUPreprocessor()
        data = IMUData(
            sensor_id=0,
            timestamp_ns=0,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_accel=None,
            angular_vel=np.zeros(3),
        )
        result = preprocessor.process(data)
        assert result is None

    def test_process_zero_quaternion(self):
        preprocessor = IMUPreprocessor()
        data = IMUData(
            sensor_id=0,
            timestamp_ns=0,
            quaternion=np.array([0.0, 0.0, 0.0, 0.0]),
            linear_accel=np.zeros(3),
            angular_vel=np.zeros(3),
        )
        result = preprocessor.process(data)
        assert result is None

    def test_process_unnormalized_quaternion(self):
        preprocessor = IMUPreprocessor()
        data = IMUData(
            sensor_id=0,
            timestamp_ns=0,
            quaternion=np.array([2.0, 0.0, 0.0, 0.0]),
            linear_accel=np.array([0.0, 0.0, 9.81]),
            angular_vel=np.zeros(3),
        )
        result = preprocessor.process(data)
        assert result is not None
        assert np.allclose(np.linalg.norm(result.quaternion), 1.0, atol=1e-6)

    def test_process_none_angular_vel(self):
        preprocessor = IMUPreprocessor()
        data = IMUData(
            sensor_id=0,
            timestamp_ns=0,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_accel=np.array([0.0, 0.0, 9.81]),
            angular_vel=None,
        )
        result = preprocessor.process(data)
        assert result is not None

    def test_quaternion_slerp_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = IMUPreprocessor.quaternion_slerp(q, q, 0.5)
        assert np.allclose(result, q, atol=1e-6)

    def test_quaternion_slerp_endpoints(self):
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        r0 = IMUPreprocessor.quaternion_slerp(q1, q2, 0.0)
        r1 = IMUPreprocessor.quaternion_slerp(q1, q2, 1.0)
        assert np.allclose(r0, q1, atol=1e-6)
        assert np.allclose(r1, q2, atol=1e-6)

    def test_quaternion_slerp_unit_output(self):
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.707, 0.707, 0.0, 0.0])
        q2 /= np.linalg.norm(q2)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = IMUPreprocessor.quaternion_slerp(q1, q2, t)
            assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_quaternion_to_rotation_matrix(self):
        preprocessor = IMUPreprocessor()
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        R = preprocessor._quaternion_to_rotation_matrix(q_identity)
        assert np.allclose(R, np.eye(3), atol=1e-6)

    def test_calibration(self):
        preprocessor = IMUPreprocessor()
        samples = []
        for _ in range(50):
            data = IMUData(
                sensor_id=0,
                timestamp_ns=0,
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                linear_accel=np.array([0.1, -0.05, 9.81]),
                angular_vel=np.zeros(3),
            )
            samples.append(data)
        preprocessor.calibrate({0: samples})
        assert preprocessor._calibrated
