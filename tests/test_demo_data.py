import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.demo_data import DemoDataGenerator
from src.pose.pose2d_estimator import NUM_KEYPOINTS


class TestDemoDataGenerator:
    @pytest.fixture
    def generator(self):
        return DemoDataGenerator()

    def test_init(self, generator):
        assert generator._dt > 0

    def test_generate_walk_cycle(self, generator):
        frames = generator.generate_walk_cycle(60)
        assert len(frames) == 60
        for frame in frames:
            assert frame.shape == (NUM_KEYPOINTS, 3)

    def test_generate_wave(self, generator):
        frames = generator.generate_wave(30)
        assert len(frames) == 30
        for frame in frames:
            assert frame.shape == (NUM_KEYPOINTS, 3)

    def test_generate_squat(self, generator):
        frames = generator.generate_squat(30)
        assert len(frames) == 30
        for frame in frames:
            assert frame.shape == (NUM_KEYPOINTS, 3)

    def test_get_demo_sequence_walk(self, generator):
        frames = generator.get_demo_sequence("walk", 10)
        assert len(frames) == 10

    def test_get_demo_sequence_wave(self, generator):
        frames = generator.get_demo_sequence("wave", 10)
        assert len(frames) == 10

    def test_get_demo_sequence_squat(self, generator):
        frames = generator.get_demo_sequence("squat", 10)
        assert len(frames) == 10

    def test_get_demo_sequence_unknown(self, generator):
        frames = generator.get_demo_sequence("unknown_motion", 10)
        assert len(frames) == 10

    def test_generate_to_bvh(self, generator, tmp_path):
        output = tmp_path / "demo.bvh"
        result = generator.generate_to_bvh("walk", 30, output)
        assert result.exists()
        content = result.read_text()
        assert "HIERARCHY" in content
        assert "MOTION" in content
        assert "Frames: 30" in content

    def test_generate_to_bvh_creates_dirs(self, generator, tmp_path):
        output = tmp_path / "sub" / "dir" / "demo.bvh"
        result = generator.generate_to_bvh("walk", 10, output)
        assert result.exists()

    def test_walk_cycle_has_motion(self, generator):
        frames = generator.generate_walk_cycle(60)
        first = frames[0]
        last = frames[-1]
        assert not np.allclose(first, last)

    def test_wave_has_arm_motion(self, generator):
        frames = generator.generate_wave(60)
        right_wrist_positions = [f[10] for f in frames]
        max_y = max(p[1] for p in right_wrist_positions)
        min_y = min(p[1] for p in right_wrist_positions)
        assert max_y - min_y > 0.01

    def test_squat_has_vertical_motion(self, generator):
        frames = generator.generate_squat(60)
        hip_heights = [f[11][1] for f in frames]
        assert max(hip_heights) - min(hip_heights) > 0.01
