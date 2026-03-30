import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class CheckStatus(Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    INFO = "info"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    detail: str = ""


@dataclass
class EnvironmentReport:
    checks: list[CheckResult] = field(default_factory=list)
    has_cameras: bool = False
    has_imus: bool = False
    has_calibration: bool = False

    @property
    def all_ok(self) -> bool:
        return all(c.status != CheckStatus.ERROR for c in self.checks)

    @property
    def can_record(self) -> bool:
        return self.has_cameras and self.has_calibration

    @property
    def can_demo(self) -> bool:
        critical = [c for c in self.checks if c.name in (
            "Python Version", "numpy", "scipy", "PyYAML"
        )]
        return all(c.status != CheckStatus.ERROR for c in critical)


class EnvironmentChecker:
    REQUIRED_PACKAGES = {
        "numpy": "numpy",
        "scipy": "scipy",
        "cv2": "opencv-python",
        "yaml": "pyyaml",
        "serial": "pyserial",
        "filterpy": "filterpy",
        "ultralytics": "ultralytics",
        "bvhsdk": "bvhsdk",
        "torch": "torch",
    }

    OPTIONAL_PACKAGES = {
        "matplotlib": "matplotlib",
        "open3d": "open3d",
    }

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        self._config_path = config_path

    def run_full_check(self) -> EnvironmentReport:
        report = EnvironmentReport()

        self._check_python(report)
        self._check_packages(report)
        self._check_cuda(report)
        self._check_gpu_memory(report)
        self._check_disk_space(report)
        self._check_cameras(report)
        self._check_imu_ports(report)
        self._check_calibration(report)
        self._check_config(report)
        self._check_directories(report)

        return report

    def _check_python(self, report: EnvironmentReport) -> None:
        ver = sys.version_info
        ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
        if ver.major == 3 and ver.minor >= 10:
            report.checks.append(CheckResult(
                "Python Version", CheckStatus.OK,
                f"Python {ver_str}", platform.python_implementation()
            ))
        elif ver.major == 3 and ver.minor >= 8:
            report.checks.append(CheckResult(
                "Python Version", CheckStatus.WARN,
                f"Python {ver_str} (3.10+ recommended)",
                f"Platform: {platform.system()} {platform.machine()}"
            ))
        else:
            report.checks.append(CheckResult(
                "Python Version", CheckStatus.ERROR,
                f"Python {ver_str} (3.10+ required)",
                "Please install Python 3.10 or later"
            ))

    def _check_packages(self, report: EnvironmentReport) -> None:
        for module_name, pip_name in self.REQUIRED_PACKAGES.items():
            try:
                mod = __import__(module_name)
                version = getattr(mod, "__version__", "installed")
                report.checks.append(CheckResult(
                    pip_name, CheckStatus.OK, version
                ))
            except ImportError:
                report.checks.append(CheckResult(
                    pip_name, CheckStatus.ERROR,
                    f"Missing: pip install {pip_name}"
                ))

        for module_name, pip_name in self.OPTIONAL_PACKAGES.items():
            try:
                mod = __import__(module_name)
                version = getattr(mod, "__version__", "installed")
                report.checks.append(CheckResult(
                    pip_name, CheckStatus.OK, version, "(optional)"
                ))
            except ImportError:
                report.checks.append(CheckResult(
                    pip_name, CheckStatus.WARN,
                    f"Not installed: pip install {pip_name}",
                    "(optional - needed for 2D skeleton preview)"
                ))

    def _check_cuda(self, report: EnvironmentReport) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                report.checks.append(CheckResult(
                    "CUDA", CheckStatus.OK,
                    f"{name} ({mem:.1f} GB)",
                    f"CUDA {torch.version.cuda}"
                ))
            else:
                report.checks.append(CheckResult(
                    "CUDA", CheckStatus.WARN,
                    "CUDA not available, using CPU",
                    "Pose estimation will be slower"
                ))
        except Exception:
            report.checks.append(CheckResult(
                "CUDA", CheckStatus.WARN, "Could not check CUDA"
            ))

    def _check_gpu_memory(self, report: EnvironmentReport) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                mem_free, mem_total = torch.cuda.mem_get_info(0)
                mem_free_gb = mem_free / (1024**3)
                if mem_free_gb < 1.0:
                    report.checks.append(CheckResult(
                        "GPU Memory", CheckStatus.WARN,
                        f"{mem_free_gb:.1f} GB free (low)",
                        "Close other GPU applications"
                    ))
                else:
                    report.checks.append(CheckResult(
                        "GPU Memory", CheckStatus.OK,
                        f"{mem_free_gb:.1f} GB free"
                    ))
        except Exception:
            pass

    def _check_disk_space(self, report: EnvironmentReport) -> None:
        try:
            usage = shutil.disk_usage(".")
            free_gb = usage.free / (1024**3)
            if free_gb < 1.0:
                report.checks.append(CheckResult(
                    "Disk Space", CheckStatus.WARN,
                    f"{free_gb:.1f} GB free (low)"
                ))
            else:
                report.checks.append(CheckResult(
                    "Disk Space", CheckStatus.OK,
                    f"{free_gb:.1f} GB free"
                ))
        except Exception:
            pass

    def _check_cameras(self, report: EnvironmentReport) -> None:
        try:
            import cv2
            import yaml

            with open(self._config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            cam_ids = cfg["cameras"]["devices"]
            found = []
            missing = []

            for cam_id in cam_ids:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        found.append(cam_id)
                    else:
                        missing.append(cam_id)
                    cap.release()
                else:
                    missing.append(cam_id)
                    cap.release()

            if found:
                report.has_cameras = True
                report.checks.append(CheckResult(
                    "Cameras", CheckStatus.OK,
                    f"Found: {found}"
                ))
            else:
                report.checks.append(CheckResult(
                    "Cameras", CheckStatus.WARN,
                    "No cameras detected",
                    "Connect USB cameras or use demo mode"
                ))

            if missing:
                report.checks.append(CheckResult(
                    "Camera IDs", CheckStatus.WARN,
                    f"Not found: {missing}",
                    "Update config/default.yaml camera device IDs"
                ))
        except Exception as e:
            report.checks.append(CheckResult(
                "Cameras", CheckStatus.WARN,
                f"Could not check: {e}"
            ))

    def _check_imu_ports(self, report: EnvironmentReport) -> None:
        try:
            import yaml

            with open(self._config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            ports = cfg["imu"]["ports"]
            available = []
            unavailable = []

            for port in ports:
                try:
                    import serial.tools.list_ports
                    port_info = [
                        p for p in serial.tools.list_ports.comports()
                        if p.device == port
                    ]
                    if port_info:
                        available.append(port)
                    else:
                        unavailable.append(port)
                except Exception:
                    unavailable.append(port)

            if available:
                report.has_imus = True
                report.checks.append(CheckResult(
                    "IMU Ports", CheckStatus.OK,
                    f"Available: {available}"
                ))
            else:
                report.checks.append(CheckResult(
                    "IMU Ports", CheckStatus.WARN,
                    "No configured IMU ports found",
                    "Connect BNO055 modules or use demo mode"
                ))

            if unavailable:
                report.checks.append(CheckResult(
                    "IMU Missing", CheckStatus.WARN,
                    f"Ports not found: {unavailable}"
                ))
        except Exception as e:
            report.checks.append(CheckResult(
                "IMU Ports", CheckStatus.WARN,
                f"Could not check: {e}"
            ))

    def _check_calibration(self, report: EnvironmentReport) -> None:
        cal_dir = Path("config/calibration")
        intrinsic = cal_dir / "intrinsic.yaml"
        extrinsic = cal_dir / "extrinsic.yaml"

        if intrinsic.exists() and extrinsic.exists():
            report.has_calibration = True
            report.checks.append(CheckResult(
                "Calibration", CheckStatus.OK,
                "Intrinsic + Extrinsic found"
            ))
        elif intrinsic.exists():
            report.checks.append(CheckResult(
                "Calibration", CheckStatus.WARN,
                "Only intrinsic calibration found",
                "Run extrinsic calibration"
            ))
        else:
            report.checks.append(CheckResult(
                "Calibration", CheckStatus.WARN,
                "No calibration data",
                "Run calibration first or use demo mode"
            ))

    def _check_config(self, report: EnvironmentReport) -> None:
        config = Path(self._config_path)
        if config.exists():
            try:
                import yaml
                with open(config, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                required_keys = ["cameras", "imu", "pose2d", "fusion", "bvh"]
                missing = [k for k in required_keys if k not in cfg]
                if missing:
                    report.checks.append(CheckResult(
                        "Config", CheckStatus.ERROR,
                        f"Missing keys: {missing}"
                    ))
                else:
                    report.checks.append(CheckResult(
                        "Config", CheckStatus.OK,
                        f"{self._config_path}"
                    ))
            except Exception as e:
                report.checks.append(CheckResult(
                    "Config", CheckStatus.ERROR,
                    f"Cannot parse: {e}"
                ))
        else:
            report.checks.append(CheckResult(
                "Config", CheckStatus.ERROR,
                f"Missing: {self._config_path}"
            ))

        skel_config = Path("config/skeleton_model.yaml")
        if skel_config.exists():
            report.checks.append(CheckResult(
                "Skeleton Config", CheckStatus.OK,
                "skeleton_model.yaml"
            ))
        else:
            report.checks.append(CheckResult(
                "Skeleton Config", CheckStatus.ERROR,
                "Missing: config/skeleton_model.yaml"
            ))

    def _check_directories(self, report: EnvironmentReport) -> None:
        dirs = ["recordings", "config/calibration"]
        for d in dirs:
            p = Path(d)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                report.checks.append(CheckResult(
                    f"Dir: {d}", CheckStatus.OK, "Created"
                ))
