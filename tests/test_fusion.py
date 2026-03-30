import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.imu_preprocessor import IMUPreprocessor
from src.fusion.ukf_fusion import VisualIMUFusion
from src.fusion.temporal_filter import TemporalFilter, FusedPose
from src.acquisition.imu_manager import IMUData
from src.pose.pose3d_reconstructor import Pose3D


def test_imu_preprocessor_basic():
    preprocessor = IMUPreprocessor()

    data = IMUData(
        sensor_id=0,
        timestamp_ns=0,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_accel=np.array([0.0, 0.0, 9.81]),
        angular_vel=np.array([0.0, 0.0, 0.0]),
    )

    processed = preprocessor.process(data)
    assert processed is not None
    assert processed.sensor_id == 0
    assert processed.quaternion.shape == (4,)


def test_imu_quaternion_slerp():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.0, 1.0, 0.0, 0.0])

    q_mid = IMUPreprocessor.quaternion_slerp(q1, q2, 0.5)
    assert np.allclose(np.linalg.norm(q_mid), 1.0, atol=1e-6)


def test_ukf_fusion():
    fusion = VisualIMUFusion("config/default.yaml")

    pose = Pose3D(
        keypoints_3d=np.random.randn(17, 3) * 0.5,
        confidence=np.ones(17),
        reprojection_errors=np.zeros(17),
    )

    fusion.initialize(pose)

    fused = fusion.get_fused_pose(timestamp_ns=0)
    assert fused.keypoints_3d.shape == (17, 3)
    assert fused.confidence.shape == (17,)


def test_ukf_predict_update():
    fusion = VisualIMUFusion("config/default.yaml")

    initial = Pose3D(
        keypoints_3d=np.zeros((17, 3)),
        confidence=np.ones(17),
        reprojection_errors=np.zeros(17),
    )

    fusion.initialize(initial)

    for _ in range(10):
        fusion.predict()

        pose = Pose3D(
            keypoints_3d=np.random.randn(17, 3) * 0.01,
            confidence=np.ones(17),
            reprojection_errors=np.zeros(17),
        )
        fusion.update_visual(pose)

    fused = fusion.get_fused_pose(timestamp_ns=0)
    assert fused.keypoints_3d.shape == (17, 3)


def test_temporal_filter():
    filt = TemporalFilter("config/default.yaml")

    poses = []
    for i in range(20):
        pose = FusedPose(
            keypoints_3d=np.sin(np.linspace(0, np.pi, 17 * 3).reshape(17, 3) + i * 0.1),
            velocities=np.zeros((17, 3)),
            confidence=np.ones(17),
            timestamp_ns=i * 33_333_333,
        )
        poses.append(pose)

    smoothed = filt.smooth_sequence(poses)
    assert len(smoothed) == len(poses)

    for p in smoothed:
        assert p.keypoints_3d.shape == (17, 3)


def test_temporal_filter_add():
    filt = TemporalFilter("config/default.yaml")

    for i in range(15):
        pose = FusedPose(
            keypoints_3d=np.random.randn(17, 3),
            velocities=np.zeros((17, 3)),
            confidence=np.ones(17),
            timestamp_ns=i * 33_333_333,
        )
        filt.add(pose)

    filtered = filt.get_filtered()
    assert filtered is not None
    assert filtered.keypoints_3d.shape == (17, 3)
