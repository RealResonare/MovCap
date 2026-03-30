import logging
import struct
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional

import numpy as np
import serial
import yaml

logger = logging.getLogger(__name__)


@dataclass
class IMUData:
    sensor_id: int
    timestamp_ns: int
    quaternion: np.ndarray      # [w, x, y, z]
    linear_accel: np.ndarray    # [ax, ay, az] m/s²
    angular_vel: np.ndarray     # [gx, gy, gz] rad/s


# BNO055 register addresses
_BNO055_ADDR = 0x28
_QUA_W_LSB = 0x20
_LIA_DATA_X_LSB = 0x28
_GYR_DATA_X_LSB = 0x14


class BNO055Serial:
    HEADER = 0xAA
    RESPONSE = 0xBB
    ERROR = 0xEE
    READ = 0x01
    WRITE = 0x00

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0) -> None:
        self._ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(0.5)
        self._set_mode(0x08)

    def _send_packet(self, cmd: int, reg: int, length: int = 0, data: bytes = b"") -> bytes:
        packet = bytes([self.HEADER, cmd, reg, length]) + data
        self._ser.write(packet)
        resp = self._ser.read(2 + length if cmd == self.READ else 2)
        if len(resp) < 2:
            raise TimeoutError(f"BNO055 response timeout on {self._ser.port}")
        if resp[0] == self.ERROR:
            raise IOError(f"BNO055 error: register=0x{reg:02X} code={resp[1]}")
        if cmd == self.READ and len(resp) >= 2 + length:
            return resp[2 : 2 + length]
        return b""

    def _read_bytes(self, reg: int, length: int) -> bytes:
        return self._send_packet(self.READ, reg, length)

    def _write_byte(self, reg: int, value: int) -> None:
        self._send_packet(self.WRITE, reg, 1, bytes([value]))

    def _set_mode(self, mode: int) -> None:
        self._write_byte(0x3D, mode)
        time.sleep(0.03)

    def read_quaternion(self) -> np.ndarray:
        raw = self._read_bytes(_QUA_W_LSB, 8)
        vals = struct.unpack_from("<hhhh", raw)
        scale = 1.0 / (1 << 14)
        return np.array([v * scale for v in vals], dtype=np.float64)

    def read_linear_accel(self) -> np.ndarray:
        raw = self._read_bytes(_LIA_DATA_X_LSB, 6)
        vals = struct.unpack_from("<hhh", raw)
        scale = 1.0 / 100.0
        return np.array([v * scale for v in vals], dtype=np.float64)

    def read_gyro(self) -> np.ndarray:
        raw = self._read_bytes(_GYR_DATA_X_LSB, 6)
        vals = struct.unpack_from("<hhh", raw)
        scale = 1.0 / 16.0 * (np.pi / 180.0)
        return np.array([v * scale for v in vals], dtype=np.float64)

    def read_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.read_quaternion(), self.read_linear_accel(), self.read_gyro()

    def close(self) -> None:
        self._set_mode(0x00)
        self._ser.close()


class IMUManager:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        imu_cfg = cfg["imu"]
        self._ports: list[str] = imu_cfg["ports"][: imu_cfg["count"]]
        self._baud: int = imu_cfg["baud_rate"]
        self._sample_rate: int = imu_cfg["sample_rate_hz"]
        self._segment_map: dict[int, str] = imu_cfg.get("segment_map", {})

        self._devices: list[BNO055Serial | None] = []
        self._queues: dict[int, Queue[IMUData]] = {}
        self._threads: list[threading.Thread] = []
        self._running = False
        self._sensor_count = len(self._ports)

    @property
    def sensor_count(self) -> int:
        return self._sensor_count

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def segment_map(self) -> dict[int, str]:
        return self._segment_map

    def start(self) -> None:
        if self._running:
            return

        for i, port in enumerate(self._ports):
            try:
                dev = BNO055Serial(port, self._baud)
                self._devices.append(dev)
                self._queues[i] = Queue(maxsize=self._sample_rate * 2)
            except Exception as e:
                logger.warning("IMU %d on %s failed to connect: %s", i, port, e)
                self._devices.append(None)
                self._queues[i] = Queue(maxsize=self._sample_rate * 2)

        self._running = True

        for i, dev in enumerate(self._devices):
            if dev is not None:
                t = threading.Thread(
                    target=self._read_loop,
                    args=(i, dev),
                    daemon=True,
                    name=f"imu-{i}",
                )
                self._threads.append(t)
                t.start()

    def _read_loop(self, sensor_id: int, device: BNO055Serial) -> None:
        interval = 1.0 / self._sample_rate
        queue = self._queues[sensor_id]

        while self._running:
            t0 = time.perf_counter()

            try:
                quat, accel, gyro = device.read_all()
                ts = time.perf_counter_ns()

                data = IMUData(
                    sensor_id=sensor_id,
                    timestamp_ns=ts,
                    quaternion=quat,
                    linear_accel=accel,
                    angular_vel=gyro,
                )

                if queue.full():
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass
                queue.put_nowait(data)

            except Exception:
                pass

            elapsed = time.perf_counter() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def read_all(self, timeout_ms: int = 50) -> dict[int, Optional[IMUData]]:
        result: dict[int, Optional[IMUData]] = {}
        for sid in range(self._sensor_count):
            try:
                result[sid] = self._queues[sid].get(timeout=timeout_ms / 1000)
            except Empty:
                result[sid] = None
        return result

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

        for dev in self._devices:
            if dev is not None:
                try:
                    dev.close()
                except Exception:
                    pass
        self._devices.clear()
        self._queues.clear()

    def __enter__(self) -> "IMUManager":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
