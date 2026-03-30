import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MoCapPipeline, DEFAULT_JOINT_TO_IMU


class TestMoCapPipeline:
    def test_init_valid_config(self):
        pipeline = MoCapPipeline("config/default.yaml")
        assert pipeline._fps > 0
        assert pipeline._frame_count == 0
        assert not pipeline._running

    def test_init_missing_config(self):
        with pytest.raises(FileNotFoundError):
            MoCapPipeline("nonexistent.yaml")

    def test_init_invalid_config(self, tmp_path):
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("just a string")
        with pytest.raises(ValueError, match="Invalid config"):
            MoCapPipeline(str(bad_config))

    def test_init_missing_section(self, tmp_path):
        bad_config = tmp_path / "missing.yaml"
        bad_config.write_text("cameras:\n  fps: 30\n")
        with pytest.raises(KeyError, match="Missing required config section"):
            MoCapPipeline(str(bad_config))

    def test_init_invalid_fps(self, tmp_path):
        bad_config = tmp_path / "badfps.yaml"
        bad_config.write_text(
            "cameras:\n  fps: 0\n  ids: [0]\n  resolution: [640, 480]\n"
            "imu:\n  port: COM3\n  baud_rate: 115200\n  num_sensors: 8\n"
            "fusion:\n  process_noise: 0.01\n  measurement_noise: 0.1\n"
        )
        with pytest.raises(ValueError, match="FPS must be positive"):
            MoCapPipeline(str(bad_config))

    def test_frame_count_initial(self):
        pipeline = MoCapPipeline("config/default.yaml")
        assert pipeline.frame_count == 0

    def test_bvh_frame_count_initial(self):
        pipeline = MoCapPipeline("config/default.yaml")
        assert pipeline.bvh_frame_count == 0

    def test_process_frame_not_running(self):
        pipeline = MoCapPipeline("config/default.yaml")
        result = pipeline.process_frame()
        assert result is None

    def test_process_frame_not_initialized(self):
        pipeline = MoCapPipeline("config/default.yaml")
        pipeline._running = True
        result = pipeline.process_frame()
        assert result is None

    def test_context_manager(self):
        pipeline = MoCapPipeline("config/default.yaml")
        assert not pipeline._running

    def test_default_joint_to_imu_mapping(self):
        assert isinstance(DEFAULT_JOINT_TO_IMU, dict)
        assert 0 in DEFAULT_JOINT_TO_IMU
        assert 5 in DEFAULT_JOINT_TO_IMU
        for coco_idx, imu_idx in DEFAULT_JOINT_TO_IMU.items():
            assert isinstance(coco_idx, int)
            assert isinstance(imu_idx, int)
            assert 0 <= imu_idx < 8

    def test_skeleton_property_before_init(self):
        pipeline = MoCapPipeline("config/default.yaml")
        assert pipeline.skeleton is None

    def test_solve_and_add_bvh_frame_not_initialized(self):
        pipeline = MoCapPipeline("config/default.yaml")
        from src.fusion.ukf_fusion import FusedPose
        fused = FusedPose(
            keypoints_3d=np.zeros((17, 3)),
            velocities=np.zeros((17, 3)),
            confidence=np.ones(17),
            timestamp_ns=0,
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            pipeline.solve_and_add_bvh_frame(fused)

    def test_export_bvh_not_initialized(self):
        pipeline = MoCapPipeline("config/default.yaml")
        with pytest.raises(RuntimeError, match="not initialized"):
            pipeline.export_bvh("test.bvh")
