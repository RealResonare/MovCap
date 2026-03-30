import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skeleton.skeleton_model import SkeletonModel
from src.skeleton.joint_angle_solver import JointAngleSolver
from src.skeleton.bvh_exporter import BVHExporter
from src.pose.pose2d_estimator import COCO_KEYPOINTS


@pytest.fixture
def skeleton():
    return SkeletonModel("config/skeleton_model.yaml")


@pytest.fixture
def angle_solver(skeleton):
    return JointAngleSolver(skeleton)


@pytest.fixture
def bvh_exporter(skeleton):
    return BVHExporter(skeleton, frame_time=1.0 / 30.0)


@pytest.fixture
def sample_keypoints_3d():
    return np.array([
        [0.00, 1.60, 0.10],
        [-0.03, 1.65, 0.10],
        [0.03, 1.65, 0.10],
        [-0.07, 1.63, 0.08],
        [0.07, 1.63, 0.08],
        [-0.18, 1.45, 0.00],
        [0.18, 1.45, 0.00],
        [-0.46, 1.45, 0.00],
        [0.46, 1.45, 0.00],
        [-0.72, 1.45, 0.00],
        [0.72, 1.45, 0.00],
        [-0.10, 0.95, 0.00],
        [0.10, 0.95, 0.00],
        [-0.10, 0.50, 0.00],
        [0.10, 0.50, 0.00],
        [-0.10, 0.08, 0.00],
        [0.10, 0.08, 0.00],
    ], dtype=np.float64)
