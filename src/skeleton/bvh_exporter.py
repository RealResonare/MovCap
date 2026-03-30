import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from bvhsdk import BVH, Joint as BVHJoint
except ImportError:
    BVH = None
    BVHJoint = None

from .skeleton_model import SkeletonModel

logger = logging.getLogger(__name__)


class BVHExporter:
    def __init__(self, skeleton: SkeletonModel, frame_time: float = 1.0 / 30.0) -> None:
        if frame_time <= 0:
            raise ValueError(f"frame_time must be positive, got {frame_time}")

        self._skeleton = skeleton
        self._frame_time = frame_time
        self._motion_data: list[dict[str, np.ndarray]] = []

    @property
    def frame_count(self) -> int:
        return len(self._motion_data)

    def add_frame(self, rotations: dict[str, np.ndarray]) -> None:
        self._motion_data.append(rotations)

    def add_frames(self, rotations_list: list[dict[str, np.ndarray]]) -> None:
        self._motion_data.extend(rotations_list)

    def clear(self) -> None:
        self._motion_data.clear()

    def export(self, output_path: str | Path) -> None:
        if BVH is None:
            raise ImportError("bvhsdk is required: pip install bvhsdk")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bvh = BVH()

        root_name = self._skeleton.root
        root_joint = self._skeleton.joints[root_name]

        self._build_hierarchy(bvh, root_name, root_joint, is_root=True)

        bvh.frames = len(self._motion_data)
        bvh.frametime = self._frame_time

        bvh.motion = []
        for frame_data in self._motion_data:
            frame_values = self._collect_frame_values(frame_data)
            bvh.motion.append(frame_values)

        bvh.write(str(output_path))

    def _build_hierarchy(
        self,
        bvh,
        joint_name: str,
        joint,
        is_root: bool = False,
    ) -> None:
        bvh_joint = BVHJoint(joint_name)

        bvh_joint.offset = list(joint.offset)
        bvh_joint.channels = joint.channels

        if is_root:
            bvh.joints = bvh_joint

        for child_name in joint.children:
            child_joint = self._skeleton.joints[child_name]
            child_bvh = BVHJoint(child_name)
            child_bvh.offset = list(child_joint.offset)
            child_bvh.channels = child_joint.channels

            for grandchild_name in child_joint.children:
                grandchild_joint = self._skeleton.joints[grandchild_name]
                self._build_child_hierarchy(bvh, grandchild_name, grandchild_joint, child_bvh)

            bvh_joint.children.append(child_bvh)

    def _build_child_hierarchy(
        self,
        bvh,
        joint_name: str,
        joint,
        parent_bvh_joint,
    ) -> None:
        bvh_joint = BVHJoint(joint_name)
        bvh_joint.offset = list(joint.offset)
        bvh_joint.channels = joint.channels

        parent_bvh_joint.children.append(bvh_joint)

        for child_name in joint.children:
            child_joint = self._skeleton.joints[child_name]
            self._build_child_hierarchy(bvh, child_name, child_joint, bvh_joint)

    def _collect_frame_values(self, frame_data: dict[str, np.ndarray]) -> list[float]:
        values: list[float] = []
        joint_order = self._skeleton.get_joint_order()

        for name in joint_order:
            if name not in frame_data:
                joint = self._skeleton.joints[name]
                if len(joint.channels) == 6:
                    values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    values.extend([0.0, 0.0, 0.0])
                continue

            data = frame_data[name]
            values.extend(data.tolist())

        return values

    def export_raw(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._skeleton.root is None:
            raise RuntimeError("Skeleton has no root joint")

        joint_order = self._skeleton.get_joint_order()

        with open(output_path, "w") as f:
            f.write("HIERARCHY\n")
            self._write_hierarchy(f, self._skeleton.root, 0)
            f.write("MOTION\n")
            f.write(f"Frames: {len(self._motion_data)}\n")
            f.write(f"Frame Time: {self._frame_time:.6f}\n")

            for frame_data in self._motion_data:
                line_values = []
                for name in joint_order:
                    if name in frame_data:
                        vals = frame_data[name]
                        line_values.extend(f"{v:.6f}" for v in vals)
                    else:
                        joint = self._skeleton.joints[name]
                        n = len(joint.channels)
                        line_values.extend(["0.000000"] * n)
                f.write(" ".join(line_values) + "\n")

    def _write_hierarchy(self, f, joint_name: str, depth: int) -> None:
        joint = self._skeleton.joints[joint_name]
        indent = "\t" * depth

        if depth == 0:
            f.write("ROOT " + joint_name + "\n")
        else:
            f.write(indent + "JOINT " + joint_name + "\n")

        f.write(indent + "{\n")
        f.write(indent + f"\tOFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
        f.write(indent + f"\tCHANNELS {len(joint.channels)} {' '.join(joint.channels)}\n")

        for child_name in joint.children:
            self._write_hierarchy(f, child_name, depth + 1)

        if not joint.children:
            f.write(indent + "\tEnd Site\n")
            f.write(indent + "\t{\n")
            f.write(indent + "\t\tOFFSET 0.000000 0.000000 0.000000\n")
            f.write(indent + "\t}\n")

        f.write(indent + "}\n")
