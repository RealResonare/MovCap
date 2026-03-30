import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.pipeline import MoCapPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Record a MoCap session")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Configuration file"
    )
    parser.add_argument(
        "--calibration",
        default="config/calibration/",
        help="Calibration directory",
    )
    parser.add_argument(
        "--output", required=True, help="Output BVH file path"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds (0 = until stopped)",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable real-time visualization",
    )

    args = parser.parse_args()

    pipeline = MoCapPipeline(args.config)
    pipeline.initialize(calibration_dir=args.calibration)

    num_frames = int(args.duration * 30) if args.duration > 0 else 0

    logger.info("Starting recording...")
    logger.info("  Output: %s", args.output)
    logger.info("  Duration: %s", "continuous" if args.duration <= 0 else f"{args.duration}s")
    logger.info("Press Ctrl+C to stop.\n")

    pipeline.start()
    poses_3d: list[list] = []

    try:
        count = 0
        while num_frames <= 0 or count < num_frames:
            fused = pipeline.process_frame()
            if fused is None:
                time.sleep(0.001)
                continue

            pipeline.solve_and_add_bvh_frame(fused)

            poses_3d.append(fused.keypoints_3d.tolist())
            count += 1

            if count % 30 == 0:
                elapsed = count / 30.0
                logger.info("  Frame %d (%.1fs)", count, elapsed)

            if num_frames > 0 and count >= num_frames:
                break

    except KeyboardInterrupt:
        logger.info("Recording stopped by user.")

    finally:
        pipeline.stop()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.export_bvh(output_path)
    logger.info("BVH saved: %s (%d frames)", output_path, pipeline.bvh_frame_count)

    raw_path = output_path.with_suffix(".json")
    with open(raw_path, "w") as f:
        json.dump(poses_3d, f)
    logger.info("Raw 3D data saved: %s", raw_path)


if __name__ == "__main__":
    main()
