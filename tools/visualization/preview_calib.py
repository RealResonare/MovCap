import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.calibration import IntrinsicCalibrator, ExtrinsicCalibrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize camera calibration results"
    )
    parser.add_argument(
        "--calibration",
        default="config/calibration/",
        help="Calibration directory",
    )

    args = parser.parse_args()
    cal_dir = Path(args.calibration)

    intrinsics = IntrinsicCalibrator()
    intrinsics.load(cal_dir)

    print("Loaded intrinsic calibrations:")
    for cam_id, cal in intrinsics.calibrations.items():
        print(f"  Camera {cam_id}:")
        print(f"    Focal length: ({cal.camera_matrix[0,0]:.1f}, {cal.camera_matrix[1,1]:.1f})")
        print(f"    Principal point: ({cal.camera_matrix[0,2]:.1f}, {cal.camera_matrix[1,2]:.1f})")
        print(f"    Reprojection error: {cal.reprojection_error:.4f} px")

    extrinsics = ExtrinsicCalibrator()
    extrinsics.load(cal_dir)

    print("\nLoaded extrinsic calibrations:")
    for (cam1, cam2), sr in extrinsics.stereo_results.items():
        baseline = np.linalg.norm(sr.T)
        print(f"  Pair ({cam1}, {cam2}):")
        print(f"    Baseline: {baseline * 1000:.1f} mm")
        print(f"    Reprojection error: {sr.reprojection_error:.4f}")


if __name__ == "__main__":
    main()
