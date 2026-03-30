import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skeleton.skeleton_model import SkeletonModel
from src.skeleton.joint_angle_solver import JointAngleSolver
from src.skeleton.bvh_exporter import BVHExporter


@pytest.fixture
def skeleton():
    return SkeletonModel("config/skeleton_model.yaml")


def test_skeleton_model(skeleton):
    assert skeleton.root == "Hips"
    assert skeleton.num_joints > 0
    assert "Hips" in skeleton.joint_names
    assert "Head" in skeleton.joint_names
    assert "LeftArm" in skeleton.joint_names


def test_skeleton_hierarchy(skeleton):
    order = skeleton.get_joint_order()
    assert order[0] == "Hips"
    assert "Spine" in order
    assert "Head" in order


def test_skeleton_forward_kinematics(skeleton):
    rotations = {name: np.eye(3) for name in skeleton.joint_names}
    positions = skeleton.forward_kinematics(rotations)

    assert "Hips" in positions
    assert "Head" in positions
    assert positions["Hips"].shape == (3,)

    head_pos = positions["Head"]
    hips_pos = positions["Hips"]
    assert head_pos[1] > hips_pos[1]


def test_joint_angle_solver(skeleton):
    solver = JointAngleSolver(skeleton)

    keypoints = np.array([
        [0.0, 1.7, 0.0],    # nose
        [0.03, 1.72, 0.0],  # left_eye
        [-0.03, 1.72, 0.0], # right_eye
        [0.06, 1.7, 0.0],   # left_ear
        [-0.06, 1.7, 0.0],  # right_ear
        [0.2, 1.4, 0.0],    # left_shoulder
        [-0.2, 1.4, 0.0],   # right_shoulder
        [0.4, 1.2, 0.0],    # left_elbow
        [-0.4, 1.2, 0.0],   # right_elbow
        [0.55, 1.0, 0.0],   # left_wrist
        [-0.55, 1.0, 0.0],  # right_wrist
        [0.1, 0.9, 0.0],    # left_hip
        [-0.1, 0.9, 0.0],   # right_hip
        [0.1, 0.5, 0.0],    # left_knee
        [-0.1, 0.5, 0.0],   # right_knee
        [0.1, 0.05, 0.0],   # left_ankle
        [-0.1, 0.05, 0.0],  # right_ankle
    ])

    from src.pose.pose2d_estimator import COCO_KEYPOINTS
    angles = solver.solve(keypoints, COCO_KEYPOINTS)

    assert isinstance(angles, dict)
    assert len(angles) > 0


def test_bvh_export(skeleton, tmp_path):
    exporter = BVHExporter(skeleton, frame_time=1.0 / 30.0)

    for _ in range(30):
        rotations = {}
        for name in skeleton.joint_names:
            joint = skeleton.joints[name]
            n_channels = len(joint.channels)
            rotations[name] = np.zeros(n_channels)
        exporter.add_frame(rotations)

    output = tmp_path / "test_output.bvh"
    exporter.export_raw(output)

    assert output.exists()

    content = output.read_text()
    assert "HIERARCHY" in content or "ROOT" in content
    assert "MOTION" in content
    assert "Frames: 30" in content
    assert "Frame Time:" in content


def test_bvh_export_empty(skeleton, tmp_path):
    exporter = BVHExporter(skeleton)
    assert exporter.frame_count == 0

    output = tmp_path / "empty.bvh"
    exporter.export_raw(output)
    assert output.exists()
