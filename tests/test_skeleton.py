import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skeleton.skeleton_model import SkeletonModel, Joint
from src.skeleton.joint_angle_solver import JointAngleSolver
from src.skeleton.bvh_exporter import BVHExporter
from src.pose.pose2d_estimator import COCO_KEYPOINTS


class TestSkeletonModel:
    def test_load_config(self, skeleton):
        assert skeleton.root is not None
        assert skeleton.root == "Hips"
        assert skeleton.num_joints > 0

    def test_joint_names(self, skeleton):
        names = skeleton.joint_names
        assert isinstance(names, list)
        assert "Hips" in names
        assert "Head" in names

    def test_joint_order(self, skeleton):
        order = skeleton.get_joint_order()
        assert order[0] == skeleton.root
        assert len(order) == skeleton.num_joints

    def test_forward_kinematics_identity(self, skeleton):
        rotations = {name: np.eye(3) for name in skeleton.joint_names}
        positions = skeleton.forward_kinematics(rotations)
        assert len(positions) == skeleton.num_joints
        for name, pos in positions.items():
            assert pos.shape == (3,)

    def test_forward_kinematics_custom_root(self, skeleton):
        rotations = {name: np.eye(3) for name in skeleton.joint_names}
        root_pos = np.array([1.0, 2.0, 3.0])
        positions = skeleton.forward_kinematics(rotations, root_position=root_pos)
        assert np.allclose(positions[skeleton.root], root_pos + skeleton.joints[skeleton.root].offset)

    def test_coco_mapping(self, skeleton):
        mapping = skeleton.get_coco_to_joint_mapping()
        assert isinstance(mapping, dict)

    def test_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            SkeletonModel("nonexistent_config.yaml")

    def test_joint_children(self, skeleton):
        root = skeleton.joints[skeleton.root]
        assert len(root.children) > 0

    def test_joint_offset_shape(self, skeleton):
        for name, joint in skeleton.joints.items():
            assert joint.offset.shape == (3,)


class TestJointAngleSolver:
    def test_solve_basic(self, angle_solver, sample_keypoints_3d):
        angles = angle_solver.solve(sample_keypoints_3d, COCO_KEYPOINTS)
        assert isinstance(angles, dict)
        assert len(angles) > 0

    def test_solve_empty_keypoints(self, angle_solver):
        result = angle_solver.solve(np.array([]), COCO_KEYPOINTS)
        assert result == {}

    def test_solve_none_keypoints(self, angle_solver):
        result = angle_solver.solve(None, COCO_KEYPOINTS)
        assert result == {}

    def test_root_has_6_channels(self, angle_solver, skeleton, sample_keypoints_3d):
        angles = angle_solver.solve(sample_keypoints_3d, COCO_KEYPOINTS)
        root_name = skeleton.root
        if root_name in angles:
            root_joint = skeleton.joints[root_name]
            if len(root_joint.channels) == 6:
                assert angles[root_name].shape == (6,)

    def test_bvh_channel_order_position_before_rotation(self, angle_solver, skeleton, sample_keypoints_3d):
        angles = angle_solver.solve(sample_keypoints_3d, COCO_KEYPOINTS)
        root_name = skeleton.root
        if root_name in angles:
            root_joint = skeleton.joints[root_name]
            if len(root_joint.channels) == 6:
                root_data = angles[root_name]
                assert root_data[0] != 0.0 or root_data[1] != 0.0 or root_data[2] != 0.0

    def test_rotation_between_parallel_vectors(self, angle_solver):
        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        R = angle_solver._rotation_between_vectors(v1, v2)
        assert np.allclose(R, np.eye(3), atol=1e-6)

    def test_rotation_between_antiparallel_vectors(self, angle_solver):
        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.array([0.0, -1.0, 0.0])
        R = angle_solver._rotation_between_vectors(v1, v2)
        result = R @ v1
        assert np.allclose(result, v2, atol=1e-6)

    def test_rotation_matrix_is_orthogonal(self, angle_solver):
        axis = np.array([1.0, 0.0, 0.0])
        angle = np.pi / 4
        R = JointAngleSolver._rotation_matrix(axis, angle)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_euler_roundtrip(self, angle_solver):
        R = JointAngleSolver._rotation_matrix(
            np.array([0.0, 0.0, 1.0]), np.pi / 6
        )
        euler = JointAngleSolver._rotation_matrix_to_euler(R)
        assert euler.shape == (3,)

    def test_get_rest_direction(self):
        assert np.allclose(JointAngleSolver._get_rest_direction("LeftArm"), [1, 0, 0])
        assert np.allclose(JointAngleSolver._get_rest_direction("RightArm"), [-1, 0, 0])
        assert np.allclose(JointAngleSolver._get_rest_direction("LeftLeg"), [0, -1, 0])
        assert np.allclose(JointAngleSolver._get_rest_direction("Spine"), [0, 1, 0])


class TestBVHExporter:
    def test_init_valid(self, skeleton):
        exporter = BVHExporter(skeleton, frame_time=1.0 / 30.0)
        assert exporter.frame_count == 0

    def test_init_invalid_frame_time(self, skeleton):
        with pytest.raises(ValueError, match="positive"):
            BVHExporter(skeleton, frame_time=0)
        with pytest.raises(ValueError, match="positive"):
            BVHExporter(skeleton, frame_time=-1.0)

    def test_add_frame(self, bvh_exporter, skeleton):
        rotations = {name: np.zeros(len(skeleton.joints[name].channels))
                     for name in skeleton.joint_names}
        bvh_exporter.add_frame(rotations)
        assert bvh_exporter.frame_count == 1

    def test_add_frames_batch(self, bvh_exporter, skeleton):
        frames = []
        for _ in range(5):
            rotations = {name: np.zeros(len(skeleton.joints[name].channels))
                         for name in skeleton.joint_names}
            frames.append(rotations)
        bvh_exporter.add_frames(frames)
        assert bvh_exporter.frame_count == 5

    def test_clear(self, bvh_exporter, skeleton):
        rotations = {name: np.zeros(len(skeleton.joints[name].channels))
                     for name in skeleton.joint_names}
        bvh_exporter.add_frame(rotations)
        bvh_exporter.clear()
        assert bvh_exporter.frame_count == 0

    def test_export_raw_hierarchy_header(self, bvh_exporter, skeleton, tmp_path):
        rotations = {name: np.zeros(len(skeleton.joints[name].channels))
                     for name in skeleton.joint_names}
        bvh_exporter.add_frame(rotations)
        output = tmp_path / "test.bvh"
        bvh_exporter.export_raw(output)
        content = output.read_text()
        assert content.startswith("HIERARCHY\n")
        assert "ROOT Hips" in content
        assert "MOTION" in content
        assert "Frames: 1" in content
        assert "Frame Time:" in content

    def test_export_raw_creates_parent_dirs(self, bvh_exporter, skeleton, tmp_path):
        rotations = {name: np.zeros(len(skeleton.joints[name].channels))
                     for name in skeleton.joint_names}
        bvh_exporter.add_frame(rotations)
        output = tmp_path / "subdir" / "deep" / "test.bvh"
        bvh_exporter.export_raw(output)
        assert output.exists()

    def test_export_raw_empty(self, bvh_exporter, tmp_path):
        output = tmp_path / "empty.bvh"
        bvh_exporter.export_raw(output)
        content = output.read_text()
        assert "Frames: 0" in content

    def test_export_raw_end_site(self, bvh_exporter, skeleton, tmp_path):
        rotations = {name: np.zeros(len(skeleton.joints[name].channels))
                     for name in skeleton.joint_names}
        bvh_exporter.add_frame(rotations)
        output = tmp_path / "test.bvh"
        bvh_exporter.export_raw(output)
        content = output.read_text()
        assert "End Site" in content

    def test_collect_frame_values_missing_joint(self, bvh_exporter, skeleton):
        values = bvh_exporter._collect_frame_values({})
        assert isinstance(values, list)
        assert all(v == 0.0 for v in values)
