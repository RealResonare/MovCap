import tkinter as tk

import numpy as np

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

LIMB_GROUPS = {
    "head": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "torso": [(5, 11), (6, 12), (11, 12), (0, 5), (0, 6)],
    "left_arm": [(5, 7), (7, 9)],
    "right_arm": [(6, 8), (8, 10)],
    "left_leg": [(11, 13), (13, 15)],
    "right_leg": [(12, 14), (14, 16)],
}

LIMB_COLORS = {
    "head": "#00CCFF",
    "torso": "#FFCC00",
    "left_arm": "#FF6600",
    "right_arm": "#FF0066",
    "left_leg": "#0066FF",
    "right_leg": "#00FF66",
}

JOINT_RADIUS = 5


class SkeletonCanvas:
    def __init__(self, parent: tk.Widget, width: int = 480, height: int = 480) -> None:
        self._width = width
        self._height = height

        self._frame = tk.Frame(parent, bg="#1a1a2e", relief=tk.RIDGE, bd=2)
        self._frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._canvas = tk.Canvas(
            self._frame, width=width, height=height,
            bg="#1a1a2e", highlightthickness=0
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        self._label = tk.Label(
            self._frame, text="无数据", bg="#1a1a2e", fg="#666688",
            font=("Consolas", 9)
        )
        self._label.pack(side=tk.BOTTOM, fill=tk.X)

        self._draw_grid()
        self._joint_items: list = []
        self._bone_items: dict = {}
        self._init_skeleton()

    @property
    def frame(self) -> tk.Frame:
        return self._frame

    def _draw_grid(self) -> None:
        cx, cy = self._width // 2, self._height // 2
        self._canvas.create_line(
            cx, 0, cx, self._height, fill="#222244", dash=(2, 4)
        )
        self._canvas.create_line(
            0, cy, self._width, cy, fill="#222244", dash=(2, 4)
        )

    def _init_skeleton(self) -> None:
        for group_name, pairs in LIMB_GROUPS.items():
            color = LIMB_COLORS.get(group_name, "#FFFFFF")
            for pair in pairs:
                item = self._canvas.create_line(
                    0, 0, 0, 0, fill=color, width=3, capstyle=tk.ROUND
                )
                self._bone_items[pair] = item

        for i in range(17):
            item = self._canvas.create_oval(
                0, 0, 0, 0, fill="#FFFFFF", outline="#AAAAAA", width=1
            )
            self._joint_items.append(item)

    def update(self, keypoints_3d: np.ndarray, frame_num: int = 0) -> None:
        if keypoints_3d is None or len(keypoints_3d) == 0:
            return

        kpts = keypoints_3d.copy()

        if kpts.ndim == 2 and kpts.shape[1] >= 2:
            xy = kpts[:, :2]
        else:
            return

        valid_mask = ~np.isnan(xy[:, 0]) & ~np.isnan(xy[:, 1])

        if not np.any(valid_mask):
            return

        valid_xy = xy[valid_mask]
        x_range = np.ptp(valid_xy[:, 0]) if np.ptp(valid_xy[:, 0]) > 0.01 else 1.0
        y_range = np.ptp(valid_xy[:, 1]) if np.ptp(valid_xy[:, 1]) > 0.01 else 1.0

        margin = 60
        scale = min(
            (self._width - 2 * margin) / x_range,
            (self._height - 2 * margin) / y_range
        ) * 0.8

        cx = (np.min(valid_xy[:, 0]) + np.max(valid_xy[:, 0])) / 2
        cy = (np.min(valid_xy[:, 1]) + np.max(valid_xy[:, 1])) / 2

        sx = self._width / 2 + (xy[:, 0] - cx) * scale
        sy = self._height / 2 - (xy[:, 1] - cy) * scale

        for pair, item in self._bone_items.items():
            i, j = pair
            if i < len(sx) and j < len(sx) and valid_mask[i] and valid_mask[j]:
                self._canvas.coords(item, sx[i], sy[i], sx[j], sy[j])
                self._canvas.itemconfig(item, state="normal")
            else:
                self._canvas.itemconfig(item, state="hidden")

        for i, item in enumerate(self._joint_items):
            if i < len(sx) and valid_mask[i]:
                r = JOINT_RADIUS
                self._canvas.coords(item, sx[i] - r, sy[i] - r, sx[i] + r, sy[i] + r)
                self._canvas.itemconfig(item, state="normal")
            else:
                self._canvas.itemconfig(item, state="hidden")

        detected = int(np.sum(valid_mask))
        self._label.config(text=f"帧: {frame_num} | 关键点: {detected}/17")

    def clear(self) -> None:
        for item in self._bone_items.values():
            self._canvas.coords(item, 0, 0, 0, 0)
        for item in self._joint_items:
            self._canvas.coords(item, 0, 0, 0, 0)
        self._label.config(text="无数据")
