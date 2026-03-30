import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.acquisition.synchronizer import StreamSynchronizer, _MAX_BUFFER_SIZE
from src.acquisition.camera_manager import Frame
from src.acquisition.imu_manager import IMUData


def _make_frame(cam_id: int, ts_ns: int) -> Frame:
    return Frame(
        camera_id=cam_id,
        frame_id=0,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp_ns=ts_ns,
    )


def _make_imu(sensor_id: int, ts_ns: int) -> IMUData:
    return IMUData(
        sensor_id=sensor_id,
        timestamp_ns=ts_ns,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_accel=np.zeros(3),
        angular_vel=np.zeros(3),
    )


class TestStreamSynchronizer:
    def test_init_default(self):
        sync = StreamSynchronizer()
        assert sync._tolerance_ns == int(16.0 * 1e6)

    def test_init_custom_tolerance(self):
        sync = StreamSynchronizer(tolerance_ms=10.0)
        assert sync._tolerance_ns == int(10.0 * 1e6)

    def test_init_invalid_tolerance(self):
        with pytest.raises(ValueError, match="positive"):
            StreamSynchronizer(tolerance_ms=0)
        with pytest.raises(ValueError, match="positive"):
            StreamSynchronizer(tolerance_ms=-5.0)

    def test_add_camera_frame(self):
        sync = StreamSynchronizer()
        frame = _make_frame(0, 1_000_000)
        sync.add_camera_frame(frame)
        assert 0 in sync._camera_buffers
        assert len(sync._camera_buffers[0]) == 1

    def test_add_imu_data(self):
        sync = StreamSynchronizer()
        data = _make_imu(0, 1_000_000)
        sync.add_imu_data(data)
        assert 0 in sync._imu_buffers
        assert len(sync._imu_buffers[0]) == 1

    def test_get_synced_sample_no_data(self):
        sync = StreamSynchronizer()
        result = sync.get_synced_sample()
        assert result is None

    def test_get_synced_sample_single_camera(self):
        sync = StreamSynchronizer()
        ts = 1_000_000_000
        sync.add_camera_frame(_make_frame(0, ts))
        result = sync.get_synced_sample()
        assert result is not None
        assert 0 in result.frames

    def test_get_synced_sample_multi_camera(self):
        sync = StreamSynchronizer(tolerance_ms=50.0)
        ts = 1_000_000_000
        sync.add_camera_frame(_make_frame(0, ts))
        sync.add_camera_frame(_make_frame(1, ts + 5_000_000))
        result = sync.get_synced_sample()
        assert result is not None
        assert 0 in result.frames
        assert 1 in result.frames

    def test_get_synced_sample_with_imu(self):
        sync = StreamSynchronizer(tolerance_ms=50.0)
        ts = 1_000_000_000
        sync.add_camera_frame(_make_frame(0, ts))
        sync.add_imu_data(_make_imu(0, ts + 1_000_000))
        result = sync.get_synced_sample()
        assert result is not None
        assert 0 in result.imu_data

    def test_buffer_pruning_camera(self):
        sync = StreamSynchronizer(tolerance_ms=16.0)
        for i in range(_MAX_BUFFER_SIZE + 50):
            sync.add_camera_frame(_make_frame(0, i * 1_000_000))
        assert len(sync._camera_buffers[0]) <= _MAX_BUFFER_SIZE

    def test_buffer_pruning_imu(self):
        sync = StreamSynchronizer(tolerance_ms=16.0)
        for i in range(_MAX_BUFFER_SIZE + 50):
            sync.add_imu_data(_make_imu(0, i * 1_000_000))
        assert len(sync._imu_buffers[0]) <= _MAX_BUFFER_SIZE

    def test_multiple_sensors(self):
        sync = StreamSynchronizer(tolerance_ms=50.0)
        ts = 1_000_000_000
        sync.add_camera_frame(_make_frame(0, ts))
        sync.add_camera_frame(_make_frame(1, ts + 2_000_000))
        sync.add_imu_data(_make_imu(0, ts))
        sync.add_imu_data(_make_imu(1, ts + 1_000_000))
        result = sync.get_synced_sample()
        assert result is not None
