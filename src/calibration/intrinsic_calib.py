import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .charuco_detector import CharucoDetector, CharucoDetection

logger = logging.getLogger(__name__)


@dataclass
class IntrinsicResult:
    camera_id: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_error: float
    image_size: tuple[int, int]


class IntrinsicCalibrator:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._detector = CharucoDetector(config_path)
        self._min_frames = cfg["calibration"]["min_calibration_frames"]
        self._max_error = cfg["calibration"]["max_reprojection_error_px"]
        self._calibrations: dict[int, IntrinsicResult] = {}

    @property
    def calibrations(self) -> dict[int, IntrinsicResult]:
        return self._calibrations

    def collect_frames(
        self, images: list[np.ndarray], camera_id: int
    ) -> list[CharucoDetection]:
        detections: list[CharucoDetection] = []
        for img in images:
            det = self._detector.detect(img)
            if det is not None:
                detections.append(det)

        if len(detections) < self._min_frames:
            raise ValueError(
                f"Camera {camera_id}: only {len(detections)} valid frames, "
                f"need at least {self._min_frames}"
            )

        return detections

    def calibrate(
        self, detections: list[CharucoDetection], camera_id: int, image_size: tuple[int, int]
    ) -> IntrinsicResult:
        all_corners: list[np.ndarray] = []
        all_ids: list[np.ndarray] = []

        for det in detections:
            all_corners.append(det.image_points)
            all_ids.append(det.ids)

        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=self._detector.board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )

        if ret > self._max_error:
            logger.warning(
                "Camera %d reproj error %.3fpx > threshold %.3fpx",
                camera_id, ret, self._max_error,
            )

        result = IntrinsicResult(
            camera_id=camera_id,
            camera_matrix=mtx,
            dist_coeffs=dist,
            reprojection_error=ret,
            image_size=image_size,
        )
        self._calibrations[camera_id] = result
        return result

    def calibrate_from_images(
        self, images: list[np.ndarray], camera_id: int
    ) -> IntrinsicResult:
        detections = self.collect_frames(images, camera_id)
        h, w = images[0].shape[:2]
        return self.calibrate(detections, camera_id, (w, h))

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for cam_id, cal in self._calibrations.items():
            data = {
                "camera_id": cam_id,
                "camera_matrix": cal.camera_matrix.tolist(),
                "dist_coeffs": cal.dist_coeffs.tolist(),
                "reprojection_error": cal.reprojection_error,
                "image_size": list(cal.image_size),
            }
            with open(output_dir / f"intrinsic_cam{cam_id}.yaml", "w") as f:
                yaml.dump(data, f)

    def load(self, input_dir: str | Path) -> None:
        input_dir = Path(input_dir)
        self._calibrations.clear()

        for path in sorted(input_dir.glob("intrinsic_cam*.yaml")):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            result = IntrinsicResult(
                camera_id=data["camera_id"],
                camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
                reprojection_error=data["reprojection_error"],
                image_size=tuple(data["image_size"]),
            )
            self._calibrations[result.camera_id] = result

    def undistort(
        self, image: np.ndarray, camera_id: int
    ) -> np.ndarray:
        cal = self._calibrations[camera_id]
        return cv2.undistort(image, cal.camera_matrix, cal.dist_coeffs)
