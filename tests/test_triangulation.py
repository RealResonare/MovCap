import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose.pose3d_reconstructor import Pose3DReconstructor


@pytest.fixture
def reconstructor():
    return Pose3DReconstructor("config/default.yaml")


def test_triangulate_empty(reconstructor):
    result = reconstructor.triangulate({})
    assert result == []


def test_triangulate_insufficient_views(reconstructor):
    P1 = np.array([
        [1000, 0, 640, 0],
        [0, 1000, 360, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)

    K1 = np.array([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1],
    ], dtype=np.float64)

    reconstructor.set_camera_params(0, P1, K1, np.zeros(5))
    result = reconstructor.triangulate({0: []})
    assert result == []


def test_dlt_triangulation(reconstructor):
    P1 = np.array([
        [1000, 0, 640, 0],
        [0, 1000, 360, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)

    K1 = np.array([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1],
    ], dtype=np.float64)

    P2 = np.array([
        [1000, 0, 640, -500],
        [0, 1000, 360, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)

    K2 = K1.copy()

    reconstructor.set_camera_params(0, P1, K1, np.zeros(5))
    reconstructor.set_camera_params(1, P2, K2, np.zeros(5))

    pt_3d = np.array([0.5, 0.3, 2.0, 1.0])

    proj1 = P1 @ pt_3d
    proj2 = P2 @ pt_3d
    uv1 = proj1[:2] / proj1[2]
    uv2 = proj2[:2] / proj2[2]

    from src.pose.pose2d_estimator import Pose2D
    pose1 = Pose2D(
        keypoints=np.zeros((17, 2)),
        confidence=np.ones(17),
        bbox=np.zeros(4),
    )
    pose1.keypoints[0] = uv1

    pose2 = Pose2D(
        keypoints=np.zeros((17, 2)),
        confidence=np.ones(17),
        bbox=np.zeros(4),
    )
    pose2.keypoints[0] = uv2

    result = reconstructor.triangulate({0: [pose1], 1: [pose2]})
    assert len(result) == 1

    reconstructed = result[0].keypoints_3d[0]
    error = np.linalg.norm(reconstructed - pt_3d[:3])
    assert error < 0.1, f"Reconstruction error too large: {error}"
