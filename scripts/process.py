import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from src.fusion.temporal_filter import TemporalFilter, FusedPose
from src.skeleton.skeleton_model import SkeletonModel
from src.skeleton.joint_angle_solver import JointAngleSolver
from src.skeleton.bvh_exporter import BVHExporter
from src.pose.pose2d_estimator import COCO_KEYPOINTS

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Process recorded MoCap data")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Configuration file"
    )
    parser.add_argument(
        "--input", required=True, help="Input raw 3D data JSON file"
    )
    parser.add_argument(
        "--output", required=True, help="Output BVH file path"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        logger.error("Input file not found: %s", args.input)
        return

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    with open(input_path, "r") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list) or len(raw_data) == 0:
        logger.error("Invalid input data: expected non-empty list of frames")
        return

    logger.info("Loaded %d frames from %s", len(raw_data), args.input)

    skeleton = SkeletonModel()
    angle_solver = JointAngleSolver(skeleton)
    bvh_exporter = BVHExporter(skeleton, frame_time=1.0 / cfg["cameras"]["fps"])
    temporal_filter = TemporalFilter(args.config)

    fused_poses: list[FusedPose] = []
    for frame_data in raw_data:
        kpts = np.array(frame_data, dtype=np.float64)
        pose = FusedPose(
            keypoints_3d=kpts,
            velocities=np.zeros_like(kpts),
            confidence=np.ones(kpts.shape[0]),
            timestamp_ns=0,
        )
        fused_poses.append(pose)

    smoothed = temporal_filter.smooth_sequence(fused_poses)
    logger.info("Smoothed %d frames", len(smoothed))

    for pose in smoothed:
        angles = angle_solver.solve(pose.keypoints_3d, COCO_KEYPOINTS)
        bvh_exporter.add_frame(angles)

    output_path = Path(args.output)
    bvh_exporter.export_raw(output_path)
    logger.info("BVH saved: %s (%d frames)", output_path, bvh_exporter.frame_count)


if __name__ == "__main__":
    main()
