import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .acquisition.camera_manager import CameraManager
from .acquisition.imu_manager import IMUManager, IMUData
from .acquisition.synchronizer import StreamSynchronizer, SyncedSample
from .calibration.intrinsic_calib import IntrinsicCalibrator
from .calibration.extrinsic_calib import ExtrinsicCalibrator
from .pose.pose2d_estimator import Pose2DEstimator, COCO_KEYPOINTS, NUM_KEYPOINTS
from .pose.pose3d_reconstructor import Pose3DReconstructor, Pose3D
from .fusion.imu_preprocessor import IMUPreprocessor, ProcessedIMU
from .fusion.ukf_fusion import VisualIMUFusion, FusedPose
from .fusion.temporal_filter import TemporalFilter
from .skeleton.skeleton_model import SkeletonModel
from .skeleton.joint_angle_solver import JointAngleSolver
from .skeleton.bvh_exporter import BVHExporter

logger = logging.getLogger(__name__)

DEFAULT_JOINT_TO_IMU = {
    0: 0,
    5: 1,
    6: 1,
    7: 3,
    8: 4,
    9: 5,
    10: 6,
    11: 2,
    12: 2,
    13: 7,
}


class MoCapPipeline:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        self._config_path = config_path
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

        if not isinstance(self._cfg, dict):
            raise ValueError(f"Invalid config format in {config_path}")

        for key in ("cameras", "imu", "fusion"):
            if key not in self._cfg:
                raise KeyError(f"Missing required config section: '{key}'")

        self._fps: int = int(self._cfg["cameras"]["fps"])
        if self._fps <= 0:
            raise ValueError(f"FPS must be positive, got {self._fps}")
        self._frame_time = 1.0 / self._fps

        self._camera_manager: Optional[CameraManager] = None
        self._imu_manager: Optional[IMUManager] = None
        self._synchronizer: Optional[StreamSynchronizer] = None
        self._pose2d: Optional[Pose2DEstimator] = None
        self._pose3d: Optional[Pose3DReconstructor] = None
        self._imu_preprocessor: Optional[IMUPreprocessor] = None
        self._fusion: Optional[VisualIMUFusion] = None
        self._temporal_filter: Optional[TemporalFilter] = None
        self._skeleton: Optional[SkeletonModel] = None
        self._angle_solver: Optional[JointAngleSolver] = None
        self._bvh_exporter: Optional[BVHExporter] = None

        self._intrinsics: Optional[IntrinsicCalibrator] = None
        self._extrinsics: Optional[ExtrinsicCalibrator] = None

        self._running = False
        self._frame_count = 0

    def load_calibration(self, calibration_dir: str | Path) -> None:
        cal_dir = Path(calibration_dir)

        self._intrinsics = IntrinsicCalibrator(self._config_path)
        self._intrinsics.load(cal_dir)

        self._extrinsics = ExtrinsicCalibrator(self._config_path)
        self._extrinsics.load(cal_dir)

        self._pose3d = Pose3DReconstructor(self._config_path)

        camera_ids = sorted(self._intrinsics.calibrations.keys())
        ref_id = camera_ids[0]

        projs = self._extrinsics.get_projection_matrices(
            ref_id, self._intrinsics.calibrations
        )

        for cam_id in camera_ids:
            if cam_id in projs:
                cal = self._intrinsics.calibrations[cam_id]
                self._pose3d.set_camera_params(
                    cam_id, projs[cam_id], cal.camera_matrix, cal.dist_coeffs
                )

    def initialize(self, calibration_dir: Optional[str | Path] = None) -> None:
        self._camera_manager = CameraManager(self._config_path)
        self._imu_manager = IMUManager(self._config_path)
        self._synchronizer = StreamSynchronizer()
        self._pose2d = Pose2DEstimator(self._config_path)
        self._imu_preprocessor = IMUPreprocessor(
            num_sensors=self._cfg["imu"]["count"]
        )
        self._fusion = VisualIMUFusion(self._config_path)
        self._temporal_filter = TemporalFilter(self._config_path)
        self._skeleton = SkeletonModel()
        self._angle_solver = JointAngleSolver(self._skeleton)
        self._bvh_exporter = BVHExporter(
            self._skeleton, frame_time=self._frame_time
        )

        if calibration_dir is not None:
            self.load_calibration(calibration_dir)

    def start(self) -> None:
        if self._camera_manager is not None:
            self._camera_manager.start()

        if self._imu_manager is not None:
            self._imu_manager.start()

        self._running = True
        self._frame_count = 0

    def process_frame(self) -> Optional[FusedPose]:
        if not self._running:
            return None

        if self._camera_manager is None or self._synchronizer is None:
            logger.warning("Pipeline not initialized: camera_manager or synchronizer is None")
            return None

        if self._fusion is None or self._temporal_filter is None:
            logger.warning("Pipeline not initialized: fusion or temporal_filter is None")
            return None

        frames = self._camera_manager.read(timeout_ms=2000)

        imu_data: dict[int, Optional[IMUData]] = {}
        if self._imu_manager is not None:
            imu_data = self._imu_manager.read_all(timeout_ms=100)

        for cam_id, frame in frames.items():
            if frame is not None:
                self._synchronizer.add_camera_frame(frame)

        for sid, data in imu_data.items():
            if data is not None:
                self._synchronizer.add_imu_data(data)

        sample = self._synchronizer.get_synced_sample()
        if sample is None:
            return None

        poses_2d: dict[int, list] = {}

        for cam_id, frame in sample.frames.items():
            if frame is not None and self._pose2d is not None:
                poses = self._pose2d.estimate(frame.image)
                poses_2d[cam_id] = poses

        pose_3d = None
        if self._pose3d is not None and len(poses_2d) >= 2:
            pose_3d_list = self._pose3d.triangulate(poses_2d)
            if pose_3d_list:
                pose_3d = pose_3d_list[0]

        processed_imu: dict[int, ProcessedIMU] = {}
        if self._imu_preprocessor is not None:
            for sid, data in sample.imu_data.items():
                if data is not None:
                    p = self._imu_preprocessor.process(data)
                    if p is not None:
                        processed_imu[sid] = p

        self._fusion.set_dt(self._frame_time)

        fused = self._fusion.fuse_step(
            pose_3d=pose_3d,
            imu_data=processed_imu,
            joint_to_imu=DEFAULT_JOINT_TO_IMU,
            timestamp_ns=sample.timestamp_ns,
        )

        self._temporal_filter.add(fused)
        filtered = self._temporal_filter.get_filtered()

        if filtered is not None:
            fused = filtered

        self._frame_count += 1
        return fused

    def process_to_bvh(
        self,
        num_frames: int = 0,
        output_path: Optional[str | Path] = None,
    ) -> None:
        if self._angle_solver is None or self._bvh_exporter is None:
            raise RuntimeError("Pipeline not initialized: call initialize() first")

        self.start()

        try:
            count = 0
            while num_frames <= 0 or count < num_frames:
                fused = self.process_frame()
                if fused is None:
                    continue

                angles = self._angle_solver.solve(
                    fused.keypoints_3d,
                    COCO_KEYPOINTS,
                )

                self._bvh_exporter.add_frame(angles)
                count += 1

                if num_frames > 0 and count >= num_frames:
                    break

        finally:
            self.stop()

        if output_path is not None:
            self._bvh_exporter.export_raw(output_path)

    def stop(self) -> None:
        self._running = False

        if self._camera_manager is not None:
            self._camera_manager.stop()

        if self._imu_manager is not None:
            self._imu_manager.stop()

    def solve_and_add_bvh_frame(self, fused: FusedPose) -> dict[str, np.ndarray]:
        if self._angle_solver is None or self._bvh_exporter is None:
            raise RuntimeError("Pipeline not initialized: call initialize() first")
        angles = self._angle_solver.solve(fused.keypoints_3d, COCO_KEYPOINTS)
        self._bvh_exporter.add_frame(angles)
        return angles

    def export_bvh(self, output_path: str | Path) -> None:
        if self._bvh_exporter is None:
            raise RuntimeError("Pipeline not initialized: call initialize() first")
        self._bvh_exporter.export_raw(output_path)

    @property
    def skeleton(self) -> Optional[SkeletonModel]:
        return self._skeleton

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def bvh_frame_count(self) -> int:
        return self._bvh_exporter.frame_count if self._bvh_exporter else 0

    def __enter__(self) -> "MoCapPipeline":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
