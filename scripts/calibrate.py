import argparse
import logging
import time
from pathlib import Path

import cv2
import yaml

from src.calibration import CharucoDetector, IntrinsicCalibrator, ExtrinsicCalibrator
from src.acquisition import CameraManager

logger = logging.getLogger(__name__)


def collect_calibration_frames(
    cam_manager: CameraManager,
    detector: CharucoDetector,
    duration_s: float = 30.0,
    min_detections: int = 30,
) -> dict[int, list]:
    camera_frames: dict[int, list] = {}
    start_time = time.time()

    print("=" * 50)
    print("摄像头标定 - ChArUco棋盘格标定法")
    print("=" * 50)
    print("\n标定步骤：")
    print("  1. 将 ChArUco 标定板放在摄像头前方")
    print("  2. 缓慢移动标定板，覆盖画面各个区域")
    print("  3. 倾斜、旋转标定板以捕捉不同角度")
    print("  4. 对于外参标定，确保标定板同时被")
    print("     多个摄像头看到")
    print("  5. 按 'q' 可提前结束\n")
    print(f"Collecting calibration frames for {duration_s}s...")
    print("Move the ChArUco board in front of all cameras.")
    print("Press 'q' to stop early.\n")

    while time.time() - start_time < duration_s:
        frames = cam_manager.read(timeout_ms=500)

        for cam_id, frame in frames.items():
            if frame is None:
                continue

            det = detector.detect(frame.image)
            if det is not None:
                if cam_id not in camera_frames:
                    camera_frames[cam_id] = []
                camera_frames[cam_id].append(det)

                vis = detector.draw_detection(frame.image, det)
                cv2.imshow(f"Camera {cam_id}", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    for cam_id, dets in camera_frames.items():
        print(f"  Camera {cam_id}: {len(dets)} valid detections")

    return camera_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate cameras for MovCap")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Configuration file"
    )
    parser.add_argument(
        "--output", default="config/calibration/", help="Output directory"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Collection duration in seconds"
    )
    parser.add_argument(
        "--generate-board",
        action="store_true",
        help="Generate and save a ChArUco board image",
    )

    args = parser.parse_args()

    if args.generate_board:
        detector = CharucoDetector(args.config)
        board_img = detector.generate_board_image()
        output_path = Path(args.output) / "charuco_board.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), board_img)
        print(f"Board image saved to {output_path}")
        return

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    detector = CharucoDetector(args.config)
    intrinsic_cal = IntrinsicCalibrator(args.config)
    extrinsic_cal = ExtrinsicCalibrator(args.config)

    with CameraManager(args.config) as cam_manager:
        print(f"Cameras started: {cam_manager.camera_count}")

        camera_frames = collect_calibration_frames(
            cam_manager, detector, duration_s=args.duration
        )

        for cam_id, detections in camera_frames.items():
            if not detections:
                print(f"Camera {cam_id}: no detections, skipping")
                continue

            img_size = (
                detections[0].image_points.shape[1],
                detections[0].image_points.shape[0],
            )

            print(f"\nCalibrating camera {cam_id}...")
            result = intrinsic_cal.calibrate(detections, cam_id, img_size)
            print(
                f"  Reprojection error: {result.reprojection_error:.4f} px"
            )

        intrinsic_cal.save(args.output)
        print(f"\nIntrinsic calibrations saved to {args.output}")

        camera_ids = sorted(camera_frames.keys())
        for i in range(len(camera_ids)):
            for j in range(i + 1, len(camera_ids)):
                cam1, cam2 = camera_ids[i], camera_ids[j]
                dets1 = camera_frames[cam1]
                dets2 = camera_frames[cam2]

                print(f"\nCalibrating stereo pair ({cam1}, {cam2})...")
                try:
                    sr = extrinsic_cal.calibrate_pair(
                        dets1,
                        dets2,
                        intrinsic_cal.calibrations[cam1],
                        intrinsic_cal.calibrations[cam2],
                        cam1,
                        cam2,
                    )
                    print(f"  Reprojection error: {sr.reprojection_error:.4f}")
                except ValueError as e:
                    print(f"  Skipped: {e}")

        extrinsic_cal.save(args.output)
        print(f"\nExtrinsic calibrations saved to {args.output}")
        print("Calibration complete!")


if __name__ == "__main__":
    main()
