import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from .charuco_detector import CharucoDetector, CharucoDetection
from .intrinsic_calib import IntrinsicResult

logger = logging.getLogger(__name__)


@dataclass
class StereoResult:
    camera_id_1: int
    camera_id_2: int
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    reprojection_error: float


@dataclass
class CameraPair:
    R: np.ndarray
    T: np.ndarray
    projection_matrix: np.ndarray


class ExtrinsicCalibrator:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._detector = CharucoDetector(config_path)
        self._min_frames = cfg["calibration"]["min_calibration_frames"]
        self._stereo_results: dict[tuple[int, int], StereoResult] = {}
        self._pair_transforms: dict[tuple[int, int], CameraPair] = {}

    @property
    def stereo_results(self) -> dict[tuple[int, int], StereoResult]:
        return self._stereo_results

    def calibrate_pair(
        self,
        detections_1: list[CharucoDetection],
        detections_2: list[CharucoDetection],
        intrinsic_1: IntrinsicResult,
        intrinsic_2: IntrinsicResult,
        camera_id_1: int,
        camera_id_2: int,
    ) -> StereoResult:
        matched_1: list[np.ndarray] = []
        matched_2: list[np.ndarray] = []

        for det1 in detections_1:
            for det2 in detections_2:
                common_ids = np.intersect1d(det1.ids.flatten(), det2.ids.flatten())
                if len(common_ids) < 4:
                    continue

                pts1 = []
                pts2 = []
                obj_pts = []

                for cid in common_ids:
                    idx1 = np.where(det1.ids.flatten() == cid)[0][0]
                    idx2 = np.where(det2.ids.flatten() == cid)[0][0]
                    pts1.append(det1.image_points[idx1])
                    pts2.append(det2.image_points[idx2])
                    obj_pts.append(self._detector.board.getChessboardCorners()[cid])

                matched_1.append(np.array(pts1, dtype=np.float32))
                matched_2.append(np.array(pts2, dtype=np.float32))

        if len(matched_1) < self._min_frames:
            raise ValueError(
                f"Stereo pair ({camera_id_1}, {camera_id_2}): "
                f"only {len(matched_1)} matched frames, need {self._min_frames}"
            )

        all_pts_1 = np.concatenate(matched_1, axis=0)
        all_pts_2 = np.concatenate(matched_2, axis=0)

        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            [self._detector.board.getChessboardCorners()],
            all_pts_1,
            all_pts_2,
            intrinsic_1.camera_matrix,
            intrinsic_1.dist_coeffs,
            intrinsic_2.camera_matrix,
            intrinsic_2.dist_coeffs,
            intrinsic_1.image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )

        result = StereoResult(
            camera_id_1=camera_id_1,
            camera_id_2=camera_id_2,
            R=R,
            T=T,
            E=E,
            F=F,
            reprojection_error=ret,
        )

        key = (camera_id_1, camera_id_2)
        self._stereo_results[key] = result

        P1 = intrinsic_1.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = intrinsic_2.camera_matrix @ np.hstack([R, T])

        self._pair_transforms[key] = CameraPair(R=R, T=T, projection_matrix=P2)

        return result

    def get_projection_matrices(
        self, reference_camera_id: int, intrinsics: dict[int, IntrinsicResult]
    ) -> dict[int, np.ndarray]:
        projs: dict[int, np.ndarray] = {}
        ref_int = intrinsics[reference_camera_id]
        projs[reference_camera_id] = ref_int.camera_matrix @ np.hstack(
            [np.eye(3), np.zeros((3, 1))]
        )

        for (cam1, cam2), pair in self._pair_transforms.items():
            if cam1 == reference_camera_id:
                other_int = intrinsics[cam2]
                projs[cam2] = other_int.camera_matrix @ np.hstack([pair.R, pair.T])
            elif cam2 == reference_camera_id:
                other_int = intrinsics[cam1]
                R_inv = pair.R.T
                T_inv = -R_inv @ pair.T
                projs[cam1] = other_int.camera_matrix @ np.hstack([R_inv, T_inv])

        return projs

    def get_extrinsics_to_reference(
        self, reference_camera_id: int
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        result[reference_camera_id] = (np.eye(3), np.zeros((3, 1)))

        for (cam1, cam2), sr in self._stereo_results.items():
            if cam1 == reference_camera_id:
                result[cam2] = (sr.R, sr.T)
            elif cam2 == reference_camera_id:
                R_inv = sr.R.T
                T_inv = -R_inv @ sr.T
                result[cam1] = (R_inv, T_inv)

        return result

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for (cam1, cam2), sr in self._stereo_results.items():
            data = {
                "camera_id_1": cam1,
                "camera_id_2": cam2,
                "R": sr.R.tolist(),
                "T": sr.T.tolist(),
                "E": sr.E.tolist(),
                "F": sr.F.tolist(),
                "reprojection_error": sr.reprojection_error,
            }
            with open(output_dir / f"extrinsic_{cam1}_{cam2}.yaml", "w") as f:
                yaml.dump(data, f)

    def load(self, input_dir: str | Path) -> None:
        input_dir = Path(input_dir)
        self._stereo_results.clear()
        self._pair_transforms.clear()

        for path in sorted(input_dir.glob("extrinsic_*_*.yaml")):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            cam1 = data["camera_id_1"]
            cam2 = data["camera_id_2"]

            sr = StereoResult(
                camera_id_1=cam1,
                camera_id_2=cam2,
                R=np.array(data["R"], dtype=np.float64),
                T=np.array(data["T"], dtype=np.float64),
                E=np.array(data["E"], dtype=np.float64),
                F=np.array(data["F"], dtype=np.float64),
                reprojection_error=data["reprojection_error"],
            )

            key = (cam1, cam2)
            self._stereo_results[key] = sr

            P2 = np.hstack([sr.R, sr.T])
            self._pair_transforms[key] = CameraPair(R=sr.R, T=sr.T, projection_matrix=P2)
