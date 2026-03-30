import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    camera_id: int
    frame_id: int
    image: np.ndarray
    timestamp_ns: int


class CameraManager:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cam_cfg = cfg["cameras"]
        self._device_ids: list[int] = cam_cfg["devices"]
        self._resolution: tuple[int, int] = tuple(cam_cfg["resolution"])
        self._fps: int = cam_cfg["fps"]
        self._backend = getattr(cv2, f"CAP_{cam_cfg['backend'].upper()}", cv2.CAP_ANY)

        self._captures: list[cv2.VideoCapture] = []
        self._frame_queues: dict[int, Queue[Frame]] = {}
        self._threads: list[threading.Thread] = []
        self._running = False
        self._frame_counters: dict[int, int] = {}
        self._lock = threading.Lock()

    @property
    def camera_count(self) -> int:
        return len(self._device_ids)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return

        for dev_id in self._device_ids:
            cap = cv2.VideoCapture(dev_id, self._backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self._fps)

            if not cap.isOpened():
                self.stop()
                raise RuntimeError(f"Cannot open camera {dev_id}")

            self._captures.append(cap)
            self._frame_queues[dev_id] = Queue(maxsize=self._fps * 2)
            self._frame_counters[dev_id] = 0

        self._running = True

        for i, dev_id in enumerate(self._device_ids):
            t = threading.Thread(
                target=self._capture_loop,
                args=(i, dev_id),
                daemon=True,
                name=f"cam-{dev_id}",
            )
            self._threads.append(t)
            t.start()

    def _capture_loop(self, index: int, dev_id: int) -> None:
        cap = self._captures[index]
        queue = self._frame_queues[dev_id]

        while self._running:
            ret, image = cap.read()
            if not ret:
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
        for dev_id in self._device_ids:
            try:
                result[dev_id] = self._frame_queues[dev_id].get(timeout=timeout_ms / 1000)
            except Empty:
                result[dev_id] = None
        return result

    def read_batch(self, timeout_ms: int = 1000) -> list[dict[int, Optional[Frame]]]:
        frames = self.read(timeout_ms)
        return [frames]

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

        for cap in self._captures:
            cap.release()
        self._captures.clear()
        self._frame_queues.clear()

    def get_properties(self, camera_id: int) -> dict[str, float]:
        index = self._device_ids.index(camera_id)
        cap = self._captures[index]
        return {
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
        }

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
