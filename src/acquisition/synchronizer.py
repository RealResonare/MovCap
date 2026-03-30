from dataclasses import dataclass
from typing import Optional

import numpy as np

from .camera_manager import Frame
from .imu_manager import IMUData


@dataclass
class SyncedFrame:
    camera_id: int
    frame_id: int
    image: np.ndarray
    timestamp_ns: int


@dataclass
class SyncedSample:
    timestamp_ns: int
    frames: dict[int, Optional[SyncedFrame]]
    imu_data: dict[int, Optional[IMUData]]


_MAX_BUFFER_SIZE = 300


class StreamSynchronizer:
    def __init__(self, tolerance_ms: float = 16.0) -> None:
        if tolerance_ms <= 0:
            raise ValueError(f"tolerance_ms must be positive, got {tolerance_ms}")
        self._tolerance_ns = int(tolerance_ms * 1e6)
        self._camera_buffers: dict[int, list[Frame]] = {}
        self._imu_buffers: dict[int, list[IMUData]] = {}

    def add_camera_frame(self, frame: Frame) -> None:
        if frame.camera_id not in self._camera_buffers:
            self._camera_buffers[frame.camera_id] = []
        buf = self._camera_buffers[frame.camera_id]
        buf.append(frame)
        if len(buf) > _MAX_BUFFER_SIZE:
            cutoff = frame.timestamp_ns - self._tolerance_ns * 10
            self._camera_buffers[frame.camera_id] = [
                f for f in buf if f.timestamp_ns > cutoff
            ]

    def add_imu_data(self, data: IMUData) -> None:
        if data.sensor_id not in self._imu_buffers:
            self._imu_buffers[data.sensor_id] = []
        buf = self._imu_buffers[data.sensor_id]
        buf.append(data)
        if len(buf) > _MAX_BUFFER_SIZE:
            cutoff = data.timestamp_ns - self._tolerance_ns * 10
            self._imu_buffers[data.sensor_id] = [
                d for d in buf if d.timestamp_ns > cutoff
            ]

    def get_synced_sample(
        self, reference_camera_id: Optional[int] = None
    ) -> Optional[SyncedSample]:
        if not self._camera_buffers:
            return None

        if reference_camera_id is None:
            reference_camera_id = min(self._camera_buffers.keys())

        ref_buf = self._camera_buffers.get(reference_camera_id, [])
        if not ref_buf:
            return None

        ref_frame = ref_buf[-1]
        ref_ts = ref_frame.timestamp_ns

        frames: dict[int, Optional[SyncedFrame]] = {}
        for cam_id, buf in self._camera_buffers.items():
            if not buf:
                frames[cam_id] = None
                continue

            best = min(buf, key=lambda f: abs(f.timestamp_ns - ref_ts))
            dt = abs(best.timestamp_ns - ref_ts)

            if dt <= self._tolerance_ns:
                frames[cam_id] = SyncedFrame(
                    camera_id=best.camera_id,
                    frame_id=best.frame_id,
                    image=best.image,
                    timestamp_ns=best.timestamp_ns,
                )
            else:
                frames[cam_id] = None

        imu_data: dict[int, Optional[IMUData]] = {}
        for sid, buf in self._imu_buffers.items():
            if not buf:
                imu_data[sid] = None
                continue

            best = min(buf, key=lambda d: abs(d.timestamp_ns - ref_ts))
            imu_data[sid] = best

        return SyncedSample(
            timestamp_ns=ref_ts,
            frames=frames,
            imu_data=imu_data,
        )

    def clear(self) -> None:
        self._camera_buffers.clear()
        self._imu_buffers.clear()
