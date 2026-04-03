import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Suppress noisy OpenCV MSMF/MSM warnings
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


@dataclass
class Frame:
    camera_id: int
    frame_id: int
    image: np.ndarray
    timestamp_ns: int


# ── Device Backend Abstraction ──────────────────────────────────────────

class DeviceBackend(ABC):
    """Abstract interface for camera data sources.

    Implement this class to add new hardware backends (FPGA, network, etc.).
    """

    @abstractmethod
    def open(self, device_id: int, width: int, height: int, fps: int) -> bool:
        """Open device. Returns True on success."""

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame. Returns None on failure."""

    @abstractmethod
    def release(self) -> None:
        """Release device resources."""

    @abstractmethod
    def get_property(self, name: str) -> float:
        """Get device property by name."""


class USBCameraBackend(DeviceBackend):
    """OpenCV VideoCapture backend for USB cameras."""

    def __init__(self, backend_name: str = "dshow") -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._backend = getattr(cv2, f"CAP_{backend_name.upper()}", cv2.CAP_ANY)

    def open(self, device_id: int, width: int, height: int, fps: int) -> bool:
        self._cap = cv2.VideoCapture(device_id, self._backend)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        # Warmup: discard first few frames to let camera stabilize
        for _ in range(5):
            self._cap.grab()

        # Verify we can actually read a frame
        ret, _ = self._cap.read()
        return ret

    def read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None

        # grab() advances the pipeline without decoding; faster and more reliable
        if not self._cap.grab():
            return None

        ret, frame = self._cap.retrieve()
        return frame if ret else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_property(self, name: str) -> float:
        prop_map = {
            "width": cv2.CAP_PROP_FRAME_WIDTH,
            "height": cv2.CAP_PROP_FRAME_HEIGHT,
            "fps": cv2.CAP_PROP_FPS,
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
        }
        if self._cap is None:
            return 0.0
        return self._cap.get(prop_map.get(name, 0))


class FPGACameraBackend(DeviceBackend):
    """FPGA frame grabber backend (placeholder for future expansion).

    Expected config:
        cameras:
          device_type: "fpga"
          fpga:
            host: "192.168.1.100"
            port: 4660
            protocol: "tcp"       # "tcp" | "udp" | "pcie"
            channels: [0, 1, 2]   # FPGA channel indices
            pixel_format: "bgr24"
    """

    def __init__(self, fpga_config: Optional[dict] = None) -> None:
        self._cfg = fpga_config or {}
        self._host: str = self._cfg.get("host", "192.168.1.100")
        self._port: int = self._cfg.get("port", 4660)
        self._protocol: str = self._cfg.get("protocol", "tcp")
        self._channels: list[int] = self._cfg.get("channels", [0])
        self._pixel_format: str = self._cfg.get("pixel_format", "bgr24")
        self._active_channel: Optional[int] = None
        self._connected = False

    def open(self, device_id: int, width: int, height: int, fps: int) -> bool:
        self._active_channel = device_id
        logger.info(
            "FPGA backend: connecting to %s:%d (channel %d, %dx%d@%dfps, %s)",
            self._host, self._port, device_id, width, height, fps, self._protocol
        )
        # TODO: implement actual FPGA connection
        # e.g. socket.connect((self._host, self._port)) for TCP/UDP
        # or  open("/dev/fpga0", ...) for PCIe
        self._connected = False
        return False

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._connected:
            return None
        # TODO: read frame from FPGA buffer
        return None

    def release(self) -> None:
        self._connected = False
        self._active_channel = None

    def get_property(self, name: str) -> float:
        return 0.0


def create_camera_backend(config_path: str) -> DeviceBackend:
    """Factory: create the appropriate backend from config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam_cfg = cfg.get("cameras", {})
    device_type = cam_cfg.get("device_type", "usb")

    if device_type == "fpga":
        return FPGACameraBackend(cam_cfg.get("fpga", {}))

    backend_name = cam_cfg.get("backend", "any")
    return USBCameraBackend(backend_name)


# ── Camera Manager ─────────────────────────────────────────────────────

class CameraManager:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cam_cfg = cfg["cameras"]
        self._device_ids: list[int] = list(cam_cfg["devices"])
        self._resolution: tuple[int, int] = tuple(cam_cfg["resolution"])
        self._fps: int = cam_cfg["fps"]
        self._config_path = config_path

        self._backends: dict[int, DeviceBackend] = {}
        self._frame_queues: dict[int, Queue[Frame]] = {}
        self._threads: list[threading.Thread] = []
        self._running = False
        self._frame_counters: dict[int, int] = {}
        self._lock = threading.Lock()
        self._connected_ids: list[int] = []

    @property
    def camera_count(self) -> int:
        return len(self._device_ids)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def connected_devices(self) -> list[int]:
        return list(self._connected_ids)

    def start(self) -> None:
        if self._running:
            return

        self._connected_ids = []

        for dev_id in self._device_ids:
            backend = create_camera_backend(self._config_path)
            try:
                ok = backend.open(
                    dev_id, self._resolution[0], self._resolution[1], self._fps
                )
            except Exception as e:
                logger.warning("Camera %d open exception: %s", dev_id, e)
                ok = False

            if not ok:
                logger.warning("Camera %d not available, skipping", dev_id)
                backend.release()
                continue

            self._backends[dev_id] = backend
            self._frame_queues[dev_id] = Queue(maxsize=self._fps * 2)
            self._frame_counters[dev_id] = 0
            self._connected_ids.append(dev_id)
            logger.info("Camera %d opened successfully", dev_id)

        self._running = True

        for dev_id in self._connected_ids:
            t = threading.Thread(
                target=self._capture_loop,
                args=(dev_id,),
                daemon=True,
                name=f"cam-{dev_id}",
            )
            self._threads.append(t)
            t.start()

        if not self._connected_ids:
            logger.warning("No cameras available")
        else:
            logger.info(
                "Cameras started: %s (of %s configured)",
                self._connected_ids, self._device_ids
            )

    def _capture_loop(self, dev_id: int) -> None:
        backend = self._backends[dev_id]
        queue = self._frame_queues[dev_id]

        while self._running:
            image = backend.read_frame()
            if image is None:
                time.sleep(0.001)
                continue

            timestamp_ns = time.perf_counter_ns()

            with self._lock:
                fid = self._frame_counters[dev_id]
                self._frame_counters[dev_id] += 1

            frame = Frame(
                camera_id=dev_id,
                frame_id=fid,
                image=image,
                timestamp_ns=timestamp_ns,
            )

            if queue.full():
                try:
                    queue.get_nowait()
                except Empty:
                    pass
            queue.put_nowait(frame)

    def read(self, timeout_ms: int = 1000) -> dict[int, Optional[Frame]]:
        result: dict[int, Optional[Frame]] = {}
        for dev_id in self._connected_ids:
            try:
                result[dev_id] = self._frame_queues[dev_id].get(timeout=timeout_ms / 1000)
            except Empty:
                result[dev_id] = None
        return result

    def read_batch(self, timeout_ms: int = 1000) -> list[dict[int, Optional[Frame]]]:
        return [self.read(timeout_ms)]

    def read_single(self, camera_id: int, timeout_ms: int = 1000) -> Optional[Frame]:
        if camera_id not in self._connected_ids:
            return None
        try:
            return self._frame_queues[camera_id].get(timeout=timeout_ms / 1000)
        except Empty:
            return None

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

        for backend in self._backends.values():
            backend.release()
        self._backends.clear()
        self._frame_queues.clear()
        self._connected_ids = []

    def get_properties(self, camera_id: int) -> dict[str, float]:
        if camera_id not in self._backends:
            return {}
        backend = self._backends[camera_id]
        return {
            "width": backend.get_property("width"),
            "height": backend.get_property("height"),
            "fps": backend.get_property("fps"),
            "brightness": backend.get_property("brightness"),
            "contrast": backend.get_property("contrast"),
        }

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
