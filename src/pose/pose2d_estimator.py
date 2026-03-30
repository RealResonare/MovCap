import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)

COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

NUM_KEYPOINTS = len(COCO_KEYPOINTS)


@dataclass
class Pose2D:
    keypoints: np.ndarray          # shape (17, 2) — x, y pixel coordinates
    confidence: np.ndarray         # shape (17,) — per-keypoint confidence
    bbox: np.ndarray               # shape (4,) — x1, y1, x2, y2
    person_id: int = 0


class Pose2DEstimator:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        if YOLO is None:
            raise ImportError("ultralytics is required: pip install ultralytics")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        pose_cfg = cfg["pose2d"]
        self._confidence_threshold = pose_cfg["confidence_threshold"]
        self._device = pose_cfg["device"]
        self._model = YOLO(pose_cfg["model"])

    @property
    def keypoint_names(self) -> list[str]:
        return COCO_KEYPOINTS

    @property
    def num_keypoints(self) -> int:
        return NUM_KEYPOINTS

    def estimate(self, image: np.ndarray) -> list[Pose2D]:
        if image is None or image.size == 0:
            return []

        try:
            results = self._model(
                image,
                conf=self._confidence_threshold,
                device=self._device,
                verbose=False,
            )
        except Exception as e:
            logger.error("YOLO inference failed: %s", e)
            return []

        poses: list[Pose2D] = []

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            kpts = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for i, (kpts_person, box) in enumerate(zip(kpts, boxes)):
                if kpts_person.shape[0] < NUM_KEYPOINTS:
                    continue
                xy = kpts_person[:, :2]
                conf = kpts_person[:, 2]
                bbox = box[:4]

                poses.append(
                    Pose2D(
                        keypoints=xy,
                        confidence=conf,
                        bbox=bbox,
                        person_id=i,
                    )
                )

        poses.sort(key=lambda p: -p.confidence.mean())
        return poses

    def estimate_batch(
        self, images: list[np.ndarray]
    ) -> list[list[Pose2D]]:
        results = self._model(
            images,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,
        )

        batch_poses: list[list[Pose2D]] = []

        for result in results:
            poses: list[Pose2D] = []

            if result.keypoints is not None:
                kpts = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()

                for i, (kpts_person, box) in enumerate(zip(kpts, boxes)):
                    xy = kpts_person[:, :2]
                    conf = kpts_person[:, 2]
                    bbox = box[:4]

                    poses.append(
                        Pose2D(
                            keypoints=xy,
                            confidence=conf,
                            bbox=bbox,
                            person_id=i,
                        )
                    )

            poses.sort(key=lambda p: -p.confidence.mean())
            batch_poses.append(poses)

        return batch_poses
