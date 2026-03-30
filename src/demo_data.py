import json
import logging
from pathlib import Path

import numpy as np

from .skeleton.skeleton_model import SkeletonModel
from .skeleton.joint_angle_solver import JointAngleSolver
from .skeleton.bvh_exporter import BVHExporter
from .pose.pose2d_estimator import COCO_KEYPOINTS

logger = logging.getLogger(__name__)

_TPOSE_OFFSETS = np.array([
    [ 0.00,  1.60,  0.10],   # 0  nose
    [-0.03,  1.65,  0.10],   # 1  left_eye
    [ 0.03,  1.65,  0.10],   # 2  right_eye
    [-0.07,  1.63,  0.08],   # 3  left_ear
    [ 0.07,  1.63,  0.08],   # 4  right_ear
    [-0.18,  1.45,  0.00],   # 5  left_shoulder
    [ 0.18,  1.45,  0.00],   # 6  right_shoulder
    [-0.46,  1.45,  0.00],   # 7  left_elbow
    [ 0.46,  1.45,  0.00],   # 8  right_elbow
    [-0.72,  1.45,  0.00],   # 9  left_wrist
    [ 0.72,  1.45,  0.00],   # 10 right_wrist
    [-0.10,  0.95,  0.00],   # 11 left_hip
    [ 0.10,  0.95,  0.00],   # 12 right_hip
    [-0.10,  0.50,  0.00],   # 13 left_knee
    [ 0.10,  0.50,  0.00],   # 14 right_knee
    [-0.10,  0.08,  0.00],   # 15 left_ankle
    [ 0.10,  0.08,  0.00],   # 16 right_ankle
], dtype=np.float64)


class DemoDataGenerator:
    def __init__(self, fps: int = 30) -> None:
        self._fps = fps
        self._dt = 1.0 / fps
        self._skeleton = SkeletonModel()
        self._angle_solver = JointAngleSolver(self._skeleton)

    def generate_walk_cycle(
        self, num_frames: int = 120, speed: float = 1.0
    ) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for i in range(num_frames):
            t = i * self._dt * speed
            kpts = self._walk_pose(t)
            frames.append(kpts)
        return frames

    def _walk_pose(self, t: float) -> np.ndarray:
        kpts = _TPOSE_OFFSETS.copy()
        phase = t * 2.0 * np.pi

        hip_sway = 0.02 * np.sin(phase)
        vertical_bob = 0.02 * abs(np.sin(phase))
        kpts[:, 0] += hip_sway * 0.3
        kpts[:, 1] += vertical_bob * 0.5

        step = 0.25 * np.sin(phase)
        kpts[11, 2] += step * 0.5
        kpts[12, 2] -= step * 0.5

        l_knee_bend = max(0, -np.sin(phase)) * 0.15
        r_knee_bend = max(0, np.sin(phase)) * 0.15
        kpts[13, 1] -= l_knee_bend
        kpts[14, 1] -= r_knee_bend
        kpts[13, 2] += step * 0.3
        kpts[14, 2] -= step * 0.3

        l_ankle = max(0, -np.sin(phase + 0.3)) * 0.08
        r_ankle = max(0, np.sin(phase + 0.3)) * 0.08
        kpts[15, 1] -= l_ankle
        kpts[16, 1] -= r_ankle

        arm_swing = 0.15 * np.sin(phase + np.pi)
        kpts[7, 2] += arm_swing
        kpts[8, 2] -= arm_swing
        kpts[9, 2] += arm_swing * 1.3
        kpts[10, 2] -= arm_swing * 1.3

        elbow_bend = 0.08 * (1 + np.sin(phase * 2)) * 0.5
        kpts[7, 1] -= elbow_bend
        kpts[8, 1] -= elbow_bend
        kpts[9, 1] -= elbow_bend * 1.5
        kpts[10, 1] -= elbow_bend * 1.5

        spine_bend = 0.01 * np.sin(phase)
        kpts[0:5, 2] += spine_bend
        kpts[5:7, 2] += spine_bend * 0.5

        return kpts

    def generate_wave(self, num_frames: int = 90) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for i in range(num_frames):
            t = i * self._dt
            kpts = _TPOSE_OFFSETS.copy()

            wave_phase = t * 4.0 * np.pi
            kpts[10, 1] = 1.45 + 0.3 * np.sin(wave_phase)
            kpts[10, 0] = 0.72 + 0.05 * np.sin(wave_phase * 2)
            kpts[10, 2] = 0.1 * np.cos(wave_phase)

            kpts[8, 1] = 1.45 + 0.1 * np.sin(wave_phase * 0.5)
            kpts[8, 0] = 0.46 + 0.03 * np.sin(wave_phase)

            if t > 0.5:
                progress = min(1.0, (t - 0.5) * 2)
                kpts[0, 1] += 0.03 * np.sin(wave_phase * 0.5) * progress

            frames.append(kpts)
        return frames

    def generate_squat(self, num_frames: int = 90) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for i in range(num_frames):
            t = i * self._dt
            kpts = _TPOSE_OFFSETS.copy()

            squat = 0.5 * (1 - np.cos(t * np.pi * 1.5))
            squat = np.clip(squat, 0, 1)

            hip_drop = 0.4 * squat
            kpts[:, 1] -= hip_drop

            knee_forward = 0.15 * squat
            kpts[13, 2] += knee_forward
            kpts[14, 2] += knee_forward

            lean = 0.1 * squat
            kpts[0:5, 2] += lean
            kpts[5:7, 2] += lean * 0.5

            arm_raise = 0.8 * squat
            kpts[9, 1] += arm_raise
            kpts[10, 1] += arm_raise
            kpts[7, 1] += arm_raise * 0.6
            kpts[8, 1] += arm_raise * 0.6
            kpts[9, 2] -= 0.2 * squat
            kpts[10, 2] -= 0.2 * squat

            frames.append(kpts)
        return frames

    def generate_to_bvh(
        self,
        motion: str = "walk",
        num_frames: int = 120,
        output_path: str | Path = "recordings/demo.bvh",
    ) -> Path:
        output_path = Path(output_path)

        frames = self.get_demo_sequence(motion, num_frames)

        exporter = BVHExporter(self._skeleton, frame_time=self._dt)
        for kpts in frames:
            angles = self._angle_solver.solve(kpts, COCO_KEYPOINTS)
            exporter.add_frame(angles)

        exporter.export_raw(output_path)
        return output_path

    def get_demo_sequence(
        self, motion: str = "walk", num_frames: int = 120
    ) -> list[np.ndarray]:
        if motion == "walk":
            return self.generate_walk_cycle(num_frames)
        elif motion == "wave":
            return self.generate_wave(num_frames)
        elif motion == "squat":
            return self.generate_squat(num_frames)
        return self.generate_walk_cycle(num_frames)

    def save_raw_json(
        self,
        motion: str = "walk",
        num_frames: int = 120,
        output_path: str | Path = "recordings/demo_raw.json",
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frames = self.get_demo_sequence(motion, num_frames)
        data = [f.tolist() for f in frames]

        with open(output_path, "w") as f:
            json.dump(data, f)

        return output_path
