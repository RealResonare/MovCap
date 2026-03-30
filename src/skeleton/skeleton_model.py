import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Joint:
    name: str
    parent: Optional[str]
    offset: np.ndarray
    channels: list[str]
    children: list[str] = field(default_factory=list)
    coco_indices: Optional[list[int]] = None


class SkeletonModel:
    def __init__(self, config_path: str = "config/skeleton_model.yaml") -> None:
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Skeleton config not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict) or "skeleton" not in cfg:
            raise ValueError(f"Invalid skeleton config: missing 'skeleton' section")

        self._joints: dict[str, Joint] = {}
        self._root: Optional[str] = None
        self._coco_keypoints: dict[int, str] = {
            int(k): v for k, v in cfg["coco_keypoints"].items()
        }
        self._segment_constraints: dict[str, list[float]] = cfg.get(
            "segment_constraints", {}
        )

        skel_cfg = cfg["skeleton"]
        for name, joint_cfg in skel_cfg.items():
            parent = joint_cfg["parent"]
            offset = np.array(joint_cfg["offset"], dtype=np.float64)
            channels = joint_cfg["channels"]
            coco_map = joint_cfg.get("coco_mapping")

            joint = Joint(
                name=name,
                parent=parent,
                offset=offset,
                channels=channels,
                coco_indices=coco_map,
            )
            self._joints[name] = joint

            if parent is not None and parent in self._joints:
                self._joints[parent].children.append(name)

        for name, j in self._joints.items():
            if j.parent is None:
                self._root = name
                break

    @property
    def joints(self) -> dict[str, Joint]:
        return self._joints

    @property
    def root(self) -> Optional[str]:
        return self._root

    @property
    def joint_names(self) -> list[str]:
        return list(self._joints.keys())

    @property
    def num_joints(self) -> int:
        return len(self._joints)

    def get_joint_order(self) -> list[str]:
        order: list[str] = []
        self._traverse(self._root, order)
        return order

    def _traverse(self, name: Optional[str], order: list[str]) -> None:
        if name is None:
            return
        order.append(name)
        for child in self._joints[name].children:
            self._traverse(child, order)

    def forward_kinematics(
        self,
        joint_rotations: dict[str, np.ndarray],
        root_position: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        positions: dict[str, np.ndarray] = {}

        if root_position is None:
            root_pos = self._joints[self._root].offset.copy()
        else:
            root_pos = root_position.copy()

        self._fk_recursive(
            self._root, root_pos, np.eye(3), joint_rotations, positions
        )

        return positions

    def _fk_recursive(
        self,
        joint_name: str,
        parent_pos: np.ndarray,
        parent_rot: np.ndarray,
        rotations: dict[str, np.ndarray],
        positions: dict[str, np.ndarray],
    ) -> None:
        joint = self._joints[joint_name]

        if joint_name in rotations:
            local_rot = rotations[joint_name]
        else:
            local_rot = np.eye(3)

        world_rot = parent_rot @ local_rot
        world_pos = parent_pos + parent_rot @ joint.offset

        positions[joint_name] = world_pos

        for child_name in joint.children:
            self._fk_recursive(
                child_name, world_pos, world_rot, rotations, positions
            )

    def get_coco_to_joint_mapping(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for joint in self._joints.values():
            if joint.coco_indices is not None:
                for idx in joint.coco_indices:
                    mapping[idx] = joint.name
        return mapping
