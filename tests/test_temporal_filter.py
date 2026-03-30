import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.temporal_filter import TemporalFilter
from src.fusion.ukf_fusion import FusedPose


def _make_fused(kpts: np.ndarray, ts: int = 0) -> FusedPose:
    return FusedPose(
        keypoints_3d=kpts,
        velocities=np.zeros_like(kpts),
        confidence=np.ones(kpts.shape[0]),
        timestamp_ns=ts,
    )


class TestTemporalFilter:
    def test_init(self):
        tf = TemporalFilter("config/default.yaml")
        assert tf._window_length % 2 == 1
        assert tf._polyorder < tf._window_length

    def test_add_single(self):
        tf = TemporalFilter("config/default.yaml")
        kpts = np.random.randn(17, 3)
        pose = _make_fused(kpts)
        tf.add(pose)
        assert len(tf._buffer) == 1

    def test_get_filtered_insufficient_data(self):
        tf = TemporalFilter("config/default.yaml")
        kpts = np.random.randn(17, 3)
        tf.add(_make_fused(kpts))
        result = tf.get_filtered()
        assert result is None

    def test_get_filtered_sufficient_data(self):
        tf = TemporalFilter("config/default.yaml")
        for i in range(tf._window_length + 5):
            kpts = np.random.randn(17, 3) * 0.01
            kpts += np.array([0.0, 1.0, 0.0])
            tf.add(_make_fused(kpts, ts=i * 33_000_000))
        result = tf.get_filtered()
        assert result is not None
        assert result.keypoints_3d.shape == (17, 3)

    def test_smooth_sequence(self):
        tf = TemporalFilter("config/default.yaml")
        poses = []
        for i in range(30):
            kpts = np.random.randn(17, 3) * 0.01
            kpts += np.array([0.0, 1.0, 0.0])
            poses.append(_make_fused(kpts, ts=i * 33_000_000))
        smoothed = tf.smooth_sequence(poses)
        assert len(smoothed) == 30
        for p in smoothed:
            assert p.keypoints_3d.shape == (17, 3)

    def test_smooth_sequence_empty(self):
        tf = TemporalFilter("config/default.yaml")
        smoothed = tf.smooth_sequence([])
        assert smoothed == []

    def test_smooth_sequence_short(self):
        tf = TemporalFilter("config/default.yaml")
        poses = [_make_fused(np.random.randn(17, 3))]
        smoothed = tf.smooth_sequence(poses)
        assert len(smoothed) == 1

    def test_buffer_max_size(self):
        tf = TemporalFilter("config/default.yaml")
        for i in range(tf._max_buffer + 50):
            kpts = np.random.randn(17, 3)
            tf.add(_make_fused(kpts))
        assert len(tf._buffer) <= tf._max_buffer

    def test_smoothing_reduces_noise(self):
        tf = TemporalFilter("config/default.yaml")
        base = np.zeros((17, 3))
        base[:, 1] = 1.0
        poses = []
        for i in range(50):
            noise = np.random.randn(17, 3) * 0.1
            poses.append(_make_fused(base + noise))
        smoothed = tf.smooth_sequence(poses)
        raw_var = np.var([p.keypoints_3d for p in poses], axis=0).mean()
        smooth_var = np.var([p.keypoints_3d for p in smoothed], axis=0).mean()
        assert smooth_var < raw_var
