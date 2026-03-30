import logging
from typing import Optional

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

logger = logging.getLogger(__name__)

SKELETON_EDGES = [
    ("Hips", "Spine"),
    ("Spine", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
    ("Chest", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Chest", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToe"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToe"),
]

JOINT_COLORS = {
    "Hips": [1, 0, 0],
    "Spine": [1, 0.5, 0],
    "Chest": [1, 1, 0],
    "Neck": [0, 1, 0],
    "Head": [0, 1, 1],
    "LeftShoulder": [0.5, 0, 1],
    "LeftArm": [0.5, 0, 1],
    "LeftForeArm": [0.5, 0, 1],
    "LeftHand": [0.5, 0, 1],
    "RightShoulder": [1, 0, 0.5],
    "RightArm": [1, 0, 0.5],
    "RightForeArm": [1, 0, 0.5],
    "RightHand": [1, 0, 0.5],
    "LeftUpLeg": [0, 0.5, 1],
    "LeftLeg": [0, 0.5, 1],
    "LeftFoot": [0, 0.5, 1],
    "LeftToe": [0, 0.5, 1],
    "RightUpLeg": [1, 0.5, 0],
    "RightLeg": [1, 0.5, 0],
    "RightFoot": [1, 0.5, 0],
    "RightToe": [1, 0.5, 0],
}


class LiveViewer:
    def __init__(self, window_name: str = "MovCap Live") -> None:
        if o3d is None:
            raise ImportError("open3d is required: pip install open3d")

        self._window_name = window_name
        self._vis: Optional[o3d.visualization.Visualizer] = None
        self._geometry: dict[str, o3d.geometry.LineSet] = {}
        self._spheres: dict[str, o3d.geometry.TriangleMesh] = {}
        self._initialized = False

    def initialize(self) -> None:
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(self._window_name, width=1280, height=720)

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self._vis.add_geometry(coord)

        for name, _ in SKELETON_EDGES:
            if name not in self._spheres:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                color = JOINT_COLORS.get(name, [1, 1, 1])
                sphere.paint_uniform_color(color)
                self._spheres[name] = sphere
                self._vis.add_geometry(sphere)

        for i, (p, c) in enumerate(SKELETON_EDGES):
            line = o3d.geometry.LineSet()
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            color = JOINT_COLORS.get(p, [1, 1, 1])
            line.colors = o3d.utility.Vector3dVector([color])
            self._geometry[f"{p}_{c}"] = line
            self._vis.add_geometry(line)

        self._initialized = True

    def update(self, joint_positions: dict[str, np.ndarray]) -> None:
        if not self._initialized:
            self.initialize()

        if self._vis is None:
            return

        for name, sphere in self._spheres.items():
            if name in joint_positions:
                pos = joint_positions[name]
                sphere.translate(pos - sphere.get_center(), relative=False)
                self._vis.update_geometry(sphere)

        for parent, child in SKELETON_EDGES:
            key = f"{parent}_{child}"
            if key not in self._geometry:
                continue
            line = self._geometry[key]
            if parent in joint_positions and child in joint_positions:
                p_pos = joint_positions[parent]
                c_pos = joint_positions[child]
                points = o3d.utility.Vector3dVector([p_pos, c_pos])
                line.points = points
                self._vis.update_geometry(line)

        self._vis.poll_events()
        self._vis.update_renderer()

    def close(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None
        self._initialized = False

    def __enter__(self) -> "LiveViewer":
        self.initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
