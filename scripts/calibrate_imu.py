import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from src.calibration.imu_calib import IMUTPoseCalibrator
from src.acquisition.imu_manager import IMUManager

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="IMU T-pose calibration for MovCap"
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Configuration file"
    )
    parser.add_argument(
        "--output", default="config/calibration/", help="Output directory"
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=3.0,
        help="T-pose hold duration in seconds (default: 3.0)",
    )

    args = parser.parse_args()

    calibrator = IMUTPoseCalibrator(args.config)

    print("=" * 50)
    print("IMU T-Pose 标定")
    print("=" * 50)
    print("\n步骤：")
    print("  1. 站立，双臂向两侧平伸（T字形姿态）")
    print("  2. 保持静止，等待数据采集完成")
    print("  3. 采集过程中不要移动")
    print(f"\n采集时长: {args.hold_duration} 秒\n")

    input("准备好后按 Enter 开始采集...")

    try:
        with IMUManager(args.config) as imu_manager:
            print(f"\nIMU 管理器已启动 ({imu_manager.sensor_count} 传感器)\n")

            def imu_read_fn():
                return imu_manager.read_all(timeout_ms=50)

            def progress_callback(message, progress):
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "=" * filled + "-" * (bar_len - filled)
                sys.stdout.write(f"\r  [{bar}] {progress*100:.0f}% {message}")
                sys.stdout.flush()

            calibrations = calibrator.collect_tpose(
                imu_read_fn,
                hold_duration=args.hold_duration,
                progress_callback=progress_callback,
            )

            print()

            if not calibrations:
                logger.error("No IMU sensors calibrated. Check connections.")
                sys.exit(1)

            for sid, cal in calibrations.items():
                print(
                    f"  IMU {sid}: {cal.num_samples} samples, "
                    f"quat=[{cal.reference_quaternion[0]:.4f}, "
                    f"{cal.reference_quaternion[1]:.4f}, "
                    f"{cal.reference_quaternion[2]:.4f}, "
                    f"{cal.reference_quaternion[3]:.4f}]"
                )

            calibrator.save(args.output)
            print(f"\n校准数据已保存到: {args.output}/imu_tpose.yaml")
            print("标定完成!")

    except KeyboardInterrupt:
        print("\n标定已取消")
        sys.exit(1)
    except Exception as e:
        logger.error("标定失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
