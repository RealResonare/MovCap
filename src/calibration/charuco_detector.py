from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import yaml


_ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
}


@dataclass
class CharucoDetection:
    corners: np.ndarray
    ids: np.ndarray
    image_points: np.ndarray
    object_points: np.ndarray
    rejected: np.ndarray


class CharucoDetector:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cal_cfg = cfg["calibration"]
        dict_name = cal_cfg["charuco_dict"]
        dict_id = _ARUCO_DICTS[dict_name]

        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self._detector_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._detector_params)

        self._board = cv2.aruco.CharucoBoard(
            size=(cal_cfg["charuco_squares_x"], cal_cfg["charuco_squares_y"]),
            squareLength=cal_cfg["charuco_square_length_m"],
            markerLength=cal_cfg["charuco_marker_length_m"],
            dictionary=self._aruco_dict,
        )

        self._square_length = cal_cfg["charuco_square_length_m"]

    @property
    def board(self) -> cv2.aruco.CharucoBoard:
        return self._board

    def detect(self, image: np.ndarray) -> Optional[CharucoDetection]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        corners, ids, rejected = self._detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            return None

        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self._board
        )

        if charuco_corners is None or charuco_ids is None:
            return None

        if len(charuco_corners) < 4:
            return None

        obj_pts = self._board.getChessboardCorners()[charuco_ids.flatten()]

        return CharucoDetection(
            corners=corners,
            ids=ids,
            image_points=charuco_corners,
            object_points=obj_pts,
            rejected=rejected,
        )

    def draw_detection(
        self, image: np.ndarray, detection: CharucoDetection
    ) -> np.ndarray:
        img = image.copy()
        cv2.aruco.drawDetectedMarkers(img, detection.corners, detection.ids)
        cv2.aruco.drawDetectedCornersCharuco(
            img, detection.image_points, detection.ids
        )
        return img

    def generate_board_image(
        self, width: int = 1200, margin: int = 20
    ) -> np.ndarray:
        return self._board.generateImage(
            (width, int(width * 5 / 7)),
            marginSize=margin,
        )
