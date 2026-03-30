import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import yaml

from .pose2d_estimator import Pose2D, NUM_KEYPOINTS, COCO_KEYPOINTS

logger = logging.getLogger(__name__)


@dataclass
class Pose3D:
    keypoints_3d: np.ndarray       # shape (17, 3) — x, y, z in world coordinates
    confidence: np.ndarray         # shape (17,) — fusion confidence
    reprojection_errors: np.ndarray  # shape (17,) — per-joint reprojection error px


class Pose3DReconstructor:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        tri_cfg = cfg["triangulation"]
        self._min_views: int = tri_cfg["min_views"]
        self._reproj_threshold: float = tri_cfg["reprojection_threshold_px"]

        self._projection_matrices: dict[int, np.ndarray] = {}
        self._camera_matrices: dict[int, np.ndarray] = {}
        self._dist_coeffs: dict[int, np.ndarray] = {}

    @property
    def keypoint_names(self) -> list[str]:
        return COCO_KEYPOINTS

    def set_camera_params(
        self,
        camera_id: int,
        projection_matrix: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        self._projection_matrices[camera_id] = projection_matrix
        self._camera_matrices[camera_id] = camera_matrix
        self._dist_coeffs[camera_id] = dist_coeffs

    def triangulate(
        self,
        poses_2d: dict[int, list[Pose2D]],
    ) -> list[Pose3D]:
        if not self._projection_matrices:
            return []

        camera_ids = sorted(set(poses_2d.keys()) & set(self._projection_matrices.keys()))

        if len(camera_ids) < self._min_views:
            return []

        # Match persons across views using simple greedy matching
        person_groups = self._match_persons(poses_2d, camera_ids)

        all_poses_3d: list[Pose3D] = []

        for group in person_groups:
            kpts_3d = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
            conf = np.zeros(NUM_KEYPOINTS, dtype=np.float64)
            reproj_errors = np.full(NUM_KEYPOINTS, np.inf)

            for j in range(NUM_KEYPOINTS):
                pts_2d = []
                proj_mats = []
                view_confs = []

                for cam_id in camera_ids:
                    pose = group.get(cam_id)
                    if pose is None:
                        continue

                    pt = pose.keypoints[j]
                    c = pose.confidence[j]

                    if c < 0.3:
                        continue

                    pts_2d.append(pt)
                    proj_mats.append(self._projection_matrices[cam_id])
                    view_confs.append(c)

                if len(pts_2d) < self._min_views:
                    continue

                pt_3d, err = self._triangulate_joint(pts_2d, proj_mats)

                if pt_3d is not None and err < self._reproj_threshold:
                    kpts_3d[j] = pt_3d
                    conf[j] = np.mean(view_confs)
                    reproj_errors[j] = err

            all_poses_3d.append(
                Pose3D(
                    keypoints_3d=kpts_3d,
                    confidence=conf,
                    reprojection_errors=reproj_errors,
                )
            )

        return all_poses_3d

    def _triangulate_joint(
        self,
        pts_2d: list[np.ndarray],
        proj_mats: list[np.ndarray],
    ) -> tuple[Optional[np.ndarray], float]:
        if len(pts_2d) == 2:
            p1 = pts_2d[0].reshape(2, 1).astype(np.float64)
            p2 = pts_2d[1].reshape(2, 1).astype(np.float64)

            point_4d = cv2.triangulatePoints(proj_mats[0], proj_mats[1], p1, p2)

            if abs(point_4d[3, 0]) < 1e-10:
                return None, float("inf")

            point_3d = (point_4d[:3, 0] / point_4d[3, 0]).astype(np.float64)

            err = self._compute_reprojection_error(point_3d, pts_2d, proj_mats)
            return point_3d, err

        else:
            return self._triangulate_dlt(pts_2d, proj_mats)

    def _triangulate_dlt(
        self,
        pts_2d: list[np.ndarray],
        proj_mats: list[np.ndarray],
    ) -> tuple[Optional[np.ndarray], float]:
        n = len(pts_2d)
        A = np.zeros((2 * n, 4), dtype=np.float64)

        for i, (pt, P) in enumerate(zip(pts_2d, proj_mats)):
            x, y = pt[0], pt[1]
            A[2 * i] = x * P[2] - P[0]
            A[2 * i + 1] = y * P[2] - P[1]

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]

        if abs(X[3]) < 1e-10:
            return None, float("inf")

        point_3d = X[:3] / X[3]

        err = self._compute_reprojection_error(point_3d, pts_2d, proj_mats)
        return point_3d, err

    def _compute_reprojection_error(
        self,
        point_3d: np.ndarray,
        pts_2d: list[np.ndarray],
        proj_mats: list[np.ndarray],
    ) -> float:
        total_err = 0.0
        count = 0

        for pt_2d, P in zip(pts_2d, proj_mats):
            proj = P @ np.append(point_3d, 1.0)
            if abs(proj[2]) < 1e-10:
                continue
            proj_2d = proj[:2] / proj[2]

            err = np.linalg.norm(proj_2d - pt_2d)
            total_err += err
            count += 1

        return total_err / count if count > 0 else float("inf")

    def _match_persons(
        self,
        poses_2d: dict[int, list[Pose2D]],
        camera_ids: list[int],
    ) -> list[dict[int, Pose2D]]:
        max_persons = max((len(p) for p in poses_2d.values()), default=0)
        groups: list[dict[int, Pose2D]] = []

        for p_idx in range(max_persons):
            group: dict[int, Pose2D] = {}
            for cam_id in camera_ids:
                poses = poses_2d.get(cam_id, [])
                if p_idx < len(poses):
                    group[cam_id] = poses[p_idx]

            if len(group) >= self._min_views:
                groups.append(group)

        return groups
