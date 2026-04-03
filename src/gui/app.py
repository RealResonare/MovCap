import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk

from .skeleton_canvas import SkeletonCanvas

BG_DARK = "#0d1117"
BG_PANEL = "#161b22"
BG_CARD = "#1c2333"
BG_INPUT = "#0d1117"
BG_BUTTON = "#21262d"
BG_BUTTON_HOVER = "#30363d"
FG_TEXT = "#c9d1d9"
FG_DIM = "#6e7681"
FG_ACCENT = "#58a6ff"
FG_OK = "#3fb950"
FG_WARN = "#d29922"
FG_ERROR = "#f85149"
FG_BORDER = "#30363d"

PREVIEW_CAM_W = 320
PREVIEW_CAM_H = 180
POSE_PREVIEW_FPS = 8

KP_CN = [
    "鼻", "左眼", "右眼", "左耳", "右耳",
    "左肩", "右肩", "左肘", "右肘",
    "左腕", "右腕", "左髋", "右髋",
    "左膝", "右膝", "左踝", "右踝",
]

SEGMENT_CN = {
    0: "头部",
    1: "胸部",
    2: "腰部",
    3: "左上臂",
    4: "右上臂",
    5: "左前臂",
    6: "右前臂",
    7: "左大腿",
}


def _section_frame(parent: tk.Widget, title: str) -> tk.LabelFrame:
    f = tk.LabelFrame(
        parent, text=f" {title} ", bg=BG_PANEL, fg=FG_ACCENT,
        font=("Segoe UI", 10, "bold"), relief=tk.FLAT, bd=1,
        highlightbackground=FG_BORDER, highlightthickness=1
    )
    f.pack(fill=tk.X, padx=8, pady=4)
    return f


class MovCapApp:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        self._config_path = config_path
        self._project_dir = Path(__file__).parent.parent.parent
        os.chdir(self._project_dir)

        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

        self._root = tk.Tk()
        self._root.title("MovCap")
        self._root.geometry("1340x920")
        self._root.minsize(1100, 750)
        self._root.configure(bg=BG_DARK)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._demo_frames: list[np.ndarray] = []
        self._demo_playing = False
        self._demo_index = 0
        self._recording = False
        self._running_task: Optional[threading.Thread] = None

        self._mode_var = tk.StringVar(value="hybrid")

        self._cam_manager = None
        self._cam_preview_running = False
        self._cam_preview_labels: dict[int, tk.Label] = {}
        self._cam_preview_indicators: dict[int, tk.Label] = {}
        self._cam_preview_photo_refs: dict[int, ImageTk.PhotoImage] = {}
        self._last_cam_frames: dict[int, np.ndarray] = {}
        self._cam_enabled: dict[int, tk.BooleanVar] = {}
        self._cam_threads: list[threading.Thread] = []
        self._cam_latency_labels: dict[int, tk.Label] = {}
        self._cam_latency_warned: set[int] = set()
        self._latency_warn_shown = False

        self._imu_enabled: dict[int, tk.BooleanVar] = {}

        self._pose_estimator = None
        self._pose_running = False
        self._skeleton_enabled = tk.BooleanVar(value=True)
        self._pose_frame_num = 0

        self._setup_styles()
        self._build_ui()
        self._log("MovCap 界面已初始化")
        self._start_camera_preview()

    # ── Styles ─────────────────────────────────────────────────────────
    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure(
            "Dark.TLabel", background=BG_DARK, foreground=FG_TEXT,
            font=("Segoe UI", 10)
        )
        style.configure(
            "Dark.TButton", background=BG_BUTTON, foreground=FG_TEXT,
            font=("Segoe UI", 10), padding=(12, 6), borderwidth=1,
            relief=tk.SOLID
        )
        style.map(
            "Dark.TButton",
            background=[("active", BG_BUTTON_HOVER)],
            foreground=[("active", FG_ACCENT)],
            relief=[("pressed", tk.SUNKEN)]
        )
        style.configure(
            "Accent.TButton", background="#1f6feb", foreground="#ffffff",
            font=("Segoe UI", 10, "bold"), padding=(12, 6), borderwidth=0
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#388bfd")],
            relief=[("pressed", tk.SUNKEN)]
        )
        style.configure(
            "Danger.TButton", background="#da3633", foreground="#ffffff",
            font=("Segoe UI", 10, "bold"), padding=(12, 6), borderwidth=0
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#f85149")],
            relief=[("pressed", tk.SUNKEN)]
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=BG_PANEL, background=FG_ACCENT
        )

    # ── Main Layout ────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        header = tk.Frame(self._root, bg=BG_PANEL, height=48,
                          highlightbackground=FG_BORDER, highlightthickness=1)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="  ●  MovCap", bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 16, "bold")
        ).pack(side=tk.LEFT, padx=(12, 0))

        tk.Label(
            header, text="视觉惯性动作捕捉系统",
            bg=BG_PANEL, fg=FG_DIM, font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=(12, 0))

        self._status_label = tk.Label(
            header, text="● 就绪 ", bg=BG_PANEL, fg=FG_OK,
            font=("Segoe UI", 10), anchor="e"
        )
        self._status_label.pack(side=tk.RIGHT, padx=(0, 12))

        body = tk.Frame(self._root, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=BG_DARK, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=8)
        left.pack_propagate(False)

        center = tk.Frame(body, bg=BG_DARK)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)

        right = tk.Frame(body, bg=BG_DARK, width=250)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        right.pack_propagate(False)

        self._build_env_panel(left)
        self._build_mode_panel(left)
        self._build_action_panel(left)
        self._build_demo_panel(left)

        self._build_camera_panel(center)
        self._build_skeleton_panel(center)
        self._build_log_panel(center)

        self._build_imu_panel(right)
        self._build_info_panel(right)
        self._build_kp_panel(right)

    # ── Environment Check ──────────────────────────────────────────────
    def _build_env_panel(self, parent: tk.Widget) -> None:
        frame = _section_frame(parent, "🔍 环境检测")

        self._env_tree = ttk.Treeview(
            frame, columns=("status", "value"), show="tree",
            height=5, selectmode="browse"
        )
        self._env_tree.column("#0", width=110, minwidth=80)
        self._env_tree.column("status", width=30, minwidth=25, anchor="center")
        self._env_tree.column("value", width=130, minwidth=80)
        self._env_tree.heading("#0", text="组件", anchor="w")
        self._env_tree.heading("status", text="", anchor="center")
        self._env_tree.heading("value", text="状态", anchor="w")
        self._env_tree.tag_configure("ok", foreground=FG_OK)
        self._env_tree.tag_configure("warn", foreground=FG_WARN)
        self._env_tree.tag_configure("error", foreground=FG_ERROR)
        self._env_tree.tag_configure("info", foreground=FG_DIM)

        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._env_tree.yview)
        self._env_tree.configure(yscrollcommand=sb.set)
        self._env_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

        btn_f = tk.Frame(frame, bg=BG_PANEL)
        btn_f.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(
            btn_f, text="刷新", style="Dark.TButton", command=self._run_env_check
        ).pack(side=tk.LEFT)
        self._env_summary = tk.Label(
            btn_f, text="", bg=BG_PANEL, fg=FG_DIM, font=("Consolas", 8)
        )
        self._env_summary.pack(side=tk.RIGHT)

        self._root.after(500, self._run_env_check)

    # ── Fusion Mode ────────────────────────────────────────────────────
    def _build_mode_panel(self, parent: tk.Widget) -> None:
        frame = _section_frame(parent, "⚙ 融合模式")

        inner = tk.Frame(frame, bg=BG_PANEL)
        inner.pack(fill=tk.X, padx=8, pady=6)

        for text, val in [("仅视觉", "visual"), ("仅IMU", "imu"), ("混合", "hybrid")]:
            tk.Radiobutton(
                inner, text=text, variable=self._mode_var, value=val,
                bg=BG_PANEL, fg=FG_TEXT, selectcolor=BG_CARD,
                activebackground=BG_PANEL, activeforeground=FG_ACCENT,
                font=("Segoe UI", 9), highlightthickness=0, borderwidth=0
            ).pack(side=tk.LEFT, padx=(0, 14))

    # ── Action Buttons ─────────────────────────────────────────────────
    def _build_action_panel(self, parent: tk.Widget) -> None:
        frame = _section_frame(parent, "▶ 操作")

        for text, cmd, sty in [
            ("◎ 标定摄像头", self._on_calibrate, "Accent.TButton"),
            ("◎ 标定IMU (T字形)", self._on_calibrate_imu, "Accent.TButton"),
            ("● 录制动作", self._on_record, "Accent.TButton"),
            ("◐ 处理数据", self._on_process, "Dark.TButton"),
            ("▷ 运行测试", self._on_run_tests, "Dark.TButton"),
        ]:
            ttk.Button(frame, text=text, style=sty, command=cmd).pack(
                fill=tk.X, padx=8, pady=2
            )

    # ── Demo Panel ─────────────────────────────────────────────────────
    def _build_demo_panel(self, parent: tk.Widget) -> None:
        frame = _section_frame(parent, "▶ 演示模式")

        tk.Label(
            frame, text="无需硬件，直接预览骨骼动画",
            bg=BG_PANEL, fg=FG_DIM, font=("Segoe UI", 8)
        ).pack(padx=8, pady=(4, 2), anchor="w")

        motion_f = tk.Frame(frame, bg=BG_PANEL)
        motion_f.pack(fill=tk.X, padx=8, pady=2)

        self._demo_var = tk.StringVar(value="walk")
        for motion in ["walk", "wave", "squat"]:
            cn = {"walk": "行走", "wave": "挥手", "squat": "下蹲"}[motion]
            tk.Radiobutton(
                motion_f, text=cn, variable=self._demo_var, value=motion,
                bg=BG_PANEL, fg=FG_TEXT, selectcolor=BG_CARD,
                activebackground=BG_PANEL, activeforeground=FG_ACCENT,
                font=("Segoe UI", 9), highlightthickness=0
            ).pack(side=tk.LEFT, padx=(0, 12))

        btn_f = tk.Frame(frame, bg=BG_PANEL)
        btn_f.pack(fill=tk.X, padx=8, pady=2)

        ttk.Button(
            btn_f, text="▶ 播放", style="Accent.TButton", command=self._on_play_demo
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self._stop_demo_btn = ttk.Button(
            btn_f, text="■ 停止", style="Danger.TButton",
            command=self._on_stop_demo, state=tk.DISABLED
        )
        self._stop_demo_btn.pack(side=tk.LEFT, padx=(3, 0))

        speed_f = tk.Frame(frame, bg=BG_PANEL)
        speed_f.pack(fill=tk.X, padx=8, pady=2)

        tk.Label(
            speed_f, text="速度:", bg=BG_PANEL, fg=FG_DIM, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT)

        self._speed_var = tk.DoubleVar(value=1.0)
        tk.Scale(
            speed_f, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self._speed_var, bg=BG_PANEL, fg=FG_TEXT,
            highlightthickness=0, troughcolor=BG_CARD,
            activebackground=FG_ACCENT, font=("Segoe UI", 8),
            length=130, showvalue=True, digits=2
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        ttk.Button(
            frame, text="⤓ 导出 BVH", style="Dark.TButton",
            command=self._on_export_demo_bvh
        ).pack(fill=tk.X, padx=8, pady=(2, 6))

    # ── Camera Preview Panel ───────────────────────────────────────────
    def _build_camera_panel(self, parent: tk.Widget) -> None:
        frame = _section_frame(parent, "📷 摄像头预览")

        top_bar = tk.Frame(frame, bg=BG_PANEL)
        top_bar.pack(fill=tk.X, padx=6, pady=(6, 0))

        tk.Checkbutton(
            top_bar, text="🦴 实时骨骼", variable=self._skeleton_enabled,
            bg=BG_PANEL, fg=FG_TEXT, selectcolor=BG_INPUT,
            activebackground=BG_PANEL, activeforeground=FG_ACCENT,
            font=("Segoe UI", 9), highlightthickness=0
        ).pack(side=tk.RIGHT)

        device_ids = self._cfg["cameras"]["devices"]
        for dev_id in device_ids:
            self._cam_enabled[dev_id] = tk.BooleanVar(value=True)

        container = tk.Frame(frame, bg=BG_PANEL)
        container.pack(fill=tk.X, padx=6, pady=6)

        for i, cam_id in enumerate(device_ids):
            card = tk.Frame(
                container, bg=BG_CARD, highlightbackground=FG_ACCENT,
                highlightthickness=1
            )
            card.pack(
                side=tk.LEFT, fill=tk.BOTH, expand=True,
                padx=(0 if i == 0 else 3, 0)
            )

            hdr = tk.Frame(card, bg=BG_CARD)
            hdr.pack(fill=tk.X, padx=4, pady=(4, 2))

            cb = tk.Checkbutton(
                hdr, text=f"摄像头 {cam_id}", variable=self._cam_enabled[cam_id],
                bg=BG_CARD, fg=FG_TEXT, selectcolor=BG_INPUT,
                activebackground=BG_CARD, activeforeground=FG_ACCENT,
                font=("Segoe UI", 9, "bold"), highlightthickness=0,
                command=lambda cid=cam_id: self._on_cam_toggle(cid)
            )
            cb.pack(side=tk.LEFT)

            lat_lbl = tk.Label(
                hdr, text="", bg=BG_CARD, fg=FG_DIM,
                font=("Consolas", 8)
            )
            lat_lbl.pack(side=tk.RIGHT, padx=(4, 2))
            self._cam_latency_labels[cam_id] = lat_lbl

            indicator = tk.Label(
                hdr, text="◌", bg=BG_CARD, fg=FG_DIM, font=("Segoe UI", 8)
            )
            indicator.pack(side=tk.RIGHT)
            self._cam_preview_indicators[cam_id] = indicator

            lbl = tk.Label(
                card, text="等待连接...", bg="#080c14", fg=FG_DIM,
                font=("Consolas", 9), width=30, height=10,
                compound=tk.CENTER
            )
            lbl.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))
            self._cam_preview_labels[cam_id] = lbl

    def _on_cam_toggle(self, cam_id: int) -> None:
        enabled = self._cam_enabled[cam_id].get()
        if not enabled and cam_id in self._cam_preview_labels:
            self._cam_preview_labels[cam_id].config(image="")
            self._cam_preview_photo_refs.pop(cam_id, None)
        state = "启用" if enabled else "禁用"
        self._log(f"摄像头 {cam_id} {state}")

    def _start_camera_preview(self) -> None:
        try:
            from src.acquisition.camera_manager import CameraManager
            self._cam_manager = CameraManager(self._config_path)
            self._cam_manager.start()

            connected = self._cam_manager.connected_devices
            configured = self._cfg["cameras"]["devices"]

            for cam_id in configured:
                if cam_id in connected:
                    self._root.after(0, self._set_cam_status, cam_id, True, FG_OK)
                else:
                    self._root.after(
                        0, self._set_cam_status, cam_id, False, FG_ERROR,
                        f"摄像头 {cam_id}\n设备不可用"
                    )

            if connected:
                self._cam_preview_running = True
                for cam_id in connected:
                    t = threading.Thread(
                        target=self._cam_single_loop, args=(cam_id,), daemon=True,
                        name=f"preview-cam-{cam_id}"
                    )
                    self._cam_threads.append(t)
                    t.start()

                self._pose_running = True
                threading.Thread(target=self._pose_preview_loop, daemon=True).start()

                self._log(f"摄像头预览已启动: {connected}")
            else:
                self._log("无可用摄像头，请检查设备连接", "warn")

        except Exception as e:
            self._log(f"摄像头初始化异常: {e}", "error")
            for cam_id in self._cam_preview_labels:
                self._root.after(
                    0, self._set_cam_status, cam_id, False, FG_ERROR,
                    f"摄像头 {cam_id}\n初始化异常"
                )

    def _set_cam_status(self, cam_id: int, connected: bool, color: str,
                        text: Optional[str] = None) -> None:
        if cam_id in self._cam_preview_indicators:
            self._cam_preview_indicators[cam_id].config(
                text="●" if connected else "◌", fg=color
            )
        if text and cam_id in self._cam_preview_labels:
            self._cam_preview_labels[cam_id].config(text=text, fg=FG_DIM)

    def _stop_camera_preview(self) -> None:
        self._cam_preview_running = False
        self._pose_running = False
        if self._cam_manager is not None:
            try:
                self._cam_manager.stop()
            except Exception:
                pass
            self._cam_manager = None
        self._cam_threads.clear()
        for cam_id in self._cam_preview_labels:
            self._root.after(
                0, self._set_cam_status, cam_id, False, FG_DIM,
                f"摄像头 {cam_id}\n已暂停"
            )

    def _restart_camera_preview(self) -> None:
        self._stop_camera_preview()
        self._cam_latency_warned.clear()
        self._latency_warn_shown = False
        time.sleep(0.3)
        self._start_camera_preview()

    def _cam_single_loop(self, cam_id: int) -> None:
        while self._cam_preview_running and self._cam_manager is not None:
            t0 = time.perf_counter()
            frame = self._cam_manager.read_single(cam_id, timeout_ms=100)
            latency_ms = (time.perf_counter() - t0) * 1000

            if frame is None:
                self._root.after(0, self._update_cam_latency, cam_id, 9999.0)
                time.sleep(0.01)
                continue

            self._root.after(0, self._update_cam_latency, cam_id, latency_ms)

            img = frame.image
            self._last_cam_frames[cam_id] = img

            try:
                enabled = self._cam_enabled[cam_id].get()
            except Exception:
                enabled = True

            if enabled:
                photo = self._format_cam_frame(img)
                self._root.after(0, self._update_cam_label, cam_id, photo)

    def _update_cam_latency(self, cam_id: int, latency_ms: float) -> None:
        if cam_id not in self._cam_latency_labels:
            return

        lbl = self._cam_latency_labels[cam_id]

        if latency_ms > 5000:
            lbl.config(text="超时", fg=FG_ERROR)
            return

        if latency_ms < 20:
            color = FG_OK
        elif latency_ms < 60:
            color = FG_WARN
        else:
            color = FG_ERROR

        lbl.config(text=f"{latency_ms:.0f}ms", fg=color)

        if latency_ms >= 100 and cam_id not in self._cam_latency_warned:
            self._cam_latency_warned.add(cam_id)
            if not self._latency_warn_shown:
                self._latency_warn_shown = True
                self._root.after(
                    0, lambda: messagebox.showwarning(
                        "通道延迟过高",
                        f"摄像头 {cam_id} 延迟: {latency_ms:.0f}ms\n\n"
                        f"建议:\n"
                        f"• 检查 USB 连接是否松动\n"
                        f"• 尝试更换 USB 端口 (优先使用 USB 3.0)\n"
                        f"• 降低分辨率或帧率\n\n"
                        f"高延迟会影响动作捕捉精度"
                    )
                )

    def _format_cam_frame(self, frame: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(PREVIEW_CAM_W / w, PREVIEW_CAM_H / h)
        resized = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(resized))

    def _update_cam_label(self, cam_id: int, photo: ImageTk.PhotoImage) -> None:
        if cam_id in self._cam_preview_labels:
            self._cam_preview_photo_refs[cam_id] = photo
            self._cam_preview_labels[cam_id].config(image=photo, text="")

    # ── Pose Preview (skeleton from live camera) ────────────────────────
    def _init_pose_estimator(self) -> None:
        if self._pose_estimator is not None:
            return
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        try:
            from src.pose.pose2d_estimator import Pose2DEstimator
            self._pose_estimator = Pose2DEstimator(self._config_path, device=device)
            self._log(f"姿态估计模型已加载 (device={device})")
        except Exception as e:
            self._log(f"姿态估计模型加载失败: {e}", "warn")
            self._pose_estimator = None

    def _pose_preview_loop(self) -> None:
        self._init_pose_estimator()

        while self._pose_running and self._cam_manager is not None:
            try:
                enabled = self._skeleton_enabled.get()
            except Exception:
                enabled = True

            if not enabled or self._pose_estimator is None:
                time.sleep(0.2)
                continue

            primary_cam = None
            if self._cam_manager.connected_devices:
                primary_cam = self._cam_manager.connected_devices[0]
            else:
                time.sleep(0.1)
                continue

            frame = self._cam_manager.read_single(primary_cam, timeout_ms=100)
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                poses = self._pose_estimator.estimate(frame.image)
            except Exception:
                time.sleep(0.1)
                continue

            if poses:
                best = poses[0]
                kpts = best.keypoints
                if kpts is not None and len(kpts) >= 17:
                    self._pose_frame_num += 1
                    self._root.after(
                        0, self._skeleton_canvas.update,
                        kpts, self._pose_frame_num
                    )
                    self._root.after(
                        0, self._update_info, "mode", "实时预览", FG_ACCENT
                    )
                    self._root.after(0, self._update_kp_indicators, kpts, best.confidence)

            time.sleep(1.0 / POSE_PREVIEW_FPS)

    def _update_kp_indicators(self, kpts: np.ndarray, confidence: Optional[np.ndarray] = None) -> None:
        for i, dot in enumerate(self._kp_indicators):
            if i >= len(kpts):
                dot.config(fg=FG_DIM)
            elif np.any(np.isnan(kpts[i, :2])):
                dot.config(fg=FG_DIM)
            elif confidence is not None and confidence[i] > 0.5:
                dot.config(fg=FG_OK)
            elif confidence is not None and confidence[i] > 0.3:
                dot.config(fg=FG_WARN)
            else:
                dot.config(fg=FG_DIM)

    # ── Skeleton Panel ─────────────────────────────────────────────────
    def _build_skeleton_panel(self, parent: tk.Widget) -> None:
        frame = tk.LabelFrame(
            parent, text=" 🦴 骨骼预览 ", bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT, bd=1,
            highlightbackground=FG_BORDER, highlightthickness=1
        )
        frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self._skeleton_canvas = SkeletonCanvas(frame, width=480, height=360)

    # ── IMU Selection Panel ────────────────────────────────────────────
    def _build_imu_panel(self, parent: tk.Widget) -> None:
        frame = _section_frame(parent, "📡 IMU 传感器")

        inner = tk.Frame(frame, bg=BG_PANEL)
        inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        canvas = tk.Canvas(inner, bg=BG_PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(inner, orient=tk.VERTICAL, command=canvas.yview)
        scroll_f = tk.Frame(canvas, bg=BG_PANEL)
        scroll_f.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_f, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)

        segment_map = self._cfg.get("imu", {}).get("segment_map", {})
        ports = self._cfg.get("imu", {}).get("ports", [])

        for idx in range(self._cfg.get("imu", {}).get("count", 8)):
            self._imu_enabled[idx] = tk.BooleanVar(value=True)
            seg_name = SEGMENT_CN.get(idx, segment_map.get(idx, f"传感器{idx}"))
            port = ports[idx] if idx < len(ports) else "---"

            row = tk.Frame(scroll_f, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=1)

            tk.Checkbutton(
                row, text=f"IMU {idx}",
                variable=self._imu_enabled[idx],
                bg=BG_PANEL, fg=FG_TEXT, selectcolor=BG_INPUT,
                activebackground=BG_PANEL, activeforeground=FG_ACCENT,
                font=("Segoe UI", 9), highlightthickness=0
            ).pack(side=tk.LEFT)

            tk.Label(
                row, text=f"{seg_name}  {port}",
                bg=BG_PANEL, fg=FG_DIM, font=("Consolas", 8)
            ).pack(side=tk.RIGHT)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ── Status Info Panel ──────────────────────────────────────────────
    def _build_info_panel(self, parent: tk.Widget) -> None:
        frame = _section_frame(parent, "📊 状态信息")

        self._info_labels: dict[str, tk.Label] = {}
        for key, label, default in [
            ("mode", "模式", "空闲"), ("frames", "帧数", "0"),
            ("fps", "帧率", "--"), ("motion", "动作", "--"),
        ]:
            row = tk.Frame(frame, bg=BG_PANEL)
            row.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(
                row, text=f"{label}:", bg=BG_PANEL, fg=FG_DIM,
                font=("Segoe UI", 9), anchor="w", width=6
            ).pack(side=tk.LEFT)
            lbl = tk.Label(
                row, text=default, bg=BG_PANEL, fg=FG_TEXT,
                font=("Consolas", 9, "bold"), anchor="w"
            )
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._info_labels[key] = lbl

        sel_f = tk.Frame(frame, bg=BG_PANEL)
        sel_f.pack(fill=tk.X, padx=8, pady=(6, 2))

        self._sel_cam_label = tk.Label(
            sel_f, text="摄像头: 0, 1, 2", bg=BG_PANEL, fg=FG_OK,
            font=("Consolas", 8), anchor="w"
        )
        self._sel_cam_label.pack(fill=tk.X)

        self._sel_imu_label = tk.Label(
            sel_f, text="IMU: 0-7", bg=BG_PANEL, fg=FG_OK,
            font=("Consolas", 8), anchor="w"
        )
        self._sel_imu_label.pack(fill=tk.X)

    # ── Keypoints Panel ────────────────────────────────────────────────
    def _build_kp_panel(self, parent: tk.Widget) -> None:
        frame = _section_frame(parent, "🎯 关键点 COCO-17")

        inner = tk.Frame(frame, bg=BG_PANEL)
        inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        canvas = tk.Canvas(inner, bg=BG_PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(inner, orient=tk.VERTICAL, command=canvas.yview)
        kp_f = tk.Frame(canvas, bg=BG_PANEL)
        kp_f.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=kp_f, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)

        self._kp_indicators = []
        for i, name in enumerate(KP_CN):
            row = tk.Frame(kp_f, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=1)
            dot = tk.Label(
                row, text="●", bg=BG_PANEL, fg=FG_DIM,
                font=("Segoe UI", 7), width=2
            )
            dot.pack(side=tk.LEFT)
            self._kp_indicators.append(dot)
            tk.Label(
                row, text=f"{i}: {name}", bg=BG_PANEL, fg=FG_DIM,
                font=("Consolas", 8), anchor="w"
            ).pack(side=tk.LEFT)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ── Log Panel ──────────────────────────────────────────────────────
    def _build_log_panel(self, parent: tk.Widget) -> None:
        frame = tk.LabelFrame(
            parent, text=" 📋 日志 ", bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT, bd=1,
            highlightbackground=FG_BORDER, highlightthickness=1
        )
        frame.pack(fill=tk.X, pady=(6, 0))

        self._log_text = scrolledtext.ScrolledText(
            frame, height=5, bg="#080c14", fg=FG_TEXT,
            font=("Consolas", 9), insertbackground=FG_ACCENT,
            selectbackground="#1c2333", relief=tk.FLAT, wrap=tk.WORD,
            state=tk.DISABLED
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        for tag, color in [
            ("info", FG_TEXT), ("ok", FG_OK), ("warn", FG_WARN),
            ("error", FG_ERROR), ("time", FG_DIM)
        ]:
            self._log_text.tag_configure(tag, foreground=color)

    # ── Helpers ─────────────────────────────────────────────────────────
    def _log(self, message: str, level: str = "info") -> None:
        ts = datetime.now().strftime("%H:%M:%S")

        def _ins() -> None:
            self._log_text.configure(state=tk.NORMAL)
            self._log_text.insert(tk.END, f"[{ts}] ", "time")
            self._log_text.insert(tk.END, f"{message}\n", level)
            self._log_text.see(tk.END)
            self._log_text.configure(state=tk.DISABLED)

        self._root.after(0, _ins)

    def _set_status(self, text: str, color: str = FG_OK) -> None:
        self._root.after(
            0, lambda: self._status_label.config(text=f"● {text} ", fg=color)
        )

    def _update_info(self, key: str, value: str, color: str = FG_TEXT) -> None:
        def _upd() -> None:
            if key in self._info_labels:
                self._info_labels[key].config(text=value, fg=color)
        self._root.after(0, _upd)

    def _get_mode_cn(self) -> str:
        return {"visual": "仅视觉", "imu": "仅IMU", "hybrid": "混合模式"}.get(
            self._mode_var.get(), "混合模式"
        )

    def _get_selected_cameras(self) -> list[int]:
        return sorted(cid for cid, var in self._cam_enabled.items() if var.get())

    def _get_selected_imus(self) -> list[int]:
        return sorted(sid for sid, var in self._imu_enabled.items() if var.get())

    def _run_in_thread(self, func, *args) -> None:
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("忙碌中", "已有任务正在运行。")
            return
        self._running_task = threading.Thread(target=func, args=args, daemon=True)
        self._running_task.start()

    # ── Environment Check ──────────────────────────────────────────────
    def _run_env_check(self) -> None:
        self._log("正在运行环境检测...")
        self._set_status("检测环境...", FG_WARN)

        def _check() -> None:
            try:
                from src.environment import EnvironmentChecker
                checker = EnvironmentChecker(self._config_path)
                report = checker.run_full_check()
                self._root.after(0, lambda: self._populate_env_tree(report))

                ok = sum(1 for c in report.checks if c.status.value == "ok")
                total = len(report.checks)
                wn = sum(1 for c in report.checks if c.status.value == "warn")
                er = sum(1 for c in report.checks if c.status.value == "error")

                summary = f"{ok}/{total} 正常"
                if wn:
                    summary += f" | {wn} 警告"
                if er:
                    summary += f" | {er} 错误"
                self._root.after(0, lambda: self._env_summary.config(text=summary))

                if report.can_demo:
                    self._set_status("环境正常 (演示可用)", FG_OK)
                elif er > 0:
                    self._set_status(f"环境异常 ({er} 错误)", FG_ERROR)
                else:
                    self._set_status(f"环境已检测 ({wn} 警告)", FG_WARN)

                self._log(f"环境检测完成: {summary}", "ok")
            except Exception as e:
                self._log(f"环境检测失败: {e}", "error")
                self._set_status("检测失败", FG_ERROR)

        threading.Thread(target=_check, daemon=True).start()

    def _populate_env_tree(self, report) -> None:
        for item in self._env_tree.get_children():
            self._env_tree.delete(item)

        icons = {"ok": "✓", "warn": "!", "error": "✗", "info": "i"}
        categories = {"系统": [], "依赖包": [], "硬件": [], "标定": [], "配置": []}

        for check in report.checks:
            name = check.name.lower()
            if name in ("python version", "disk space"):
                categories["系统"].append(check)
            elif any(p in name for p in (
                "numpy", "scipy", "opencv", "pyyaml", "pyserial",
                "filterpy", "ultralytics", "open3d", "bvhsdk", "torch",
                "matplotlib", "cuda", "gpu"
            )):
                categories["依赖包"].append(check)
            elif name in ("cameras", "camera ids", "imu ports", "imu missing"):
                categories["硬件"].append(check)
            elif "calibrat" in name:
                categories["标定"].append(check)
            else:
                categories["配置"].append(check)

        for cat_name, checks in categories.items():
            if not checks:
                continue
            parent = self._env_tree.insert(
                "", tk.END, text=cat_name, values=("", ""), tags=("info",)
            )
            for c in checks:
                icon = icons.get(c.status.value, "?")
                self._env_tree.insert(
                    parent, tk.END, text=c.name,
                    values=(icon, c.message), tags=(c.status.value,)
                )
            self._env_tree.item(parent, open=True)

    # ── Action Handlers ────────────────────────────────────────────────
    def _on_calibrate(self) -> None:
        proceed = messagebox.askokcancel(
            "摄像头标定说明",
            "标定方式：ChArUco棋盘格\n\n"
            "步骤：\n"
            "1. 准备 ChArUco 标定板\n"
            "   (7x5 格, 35mm方格, 26mm标记)\n"
            "2. 将标定板依次在每个摄像头前移动\n"
            "3. 倾斜、旋转标定板以覆盖不同角度\n"
            "4. 确保所有摄像头都能看到标定板\n"
            "5. 保持标定板完全平放在摄像头视野中\n"
            "   进行外参标定\n\n"
            "提示：标定过程中会显示检测到的标记，\n"
            "按 'q' 可提前结束。\n\n"
            "点击确定开始标定。"
        )
        if not proceed:
            return

        duration = 30
        self._log(f"开始摄像头标定 ({duration}秒)...", "info")
        self._set_status("标定中...", FG_WARN)

        def _run() -> None:
            self._root.after(0, self._stop_camera_preview)
            time.sleep(0.5)

            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "scripts.calibrate",
                        "--config", self._config_path,
                        "--output", "config/calibration/",
                        "--duration", str(duration),
                    ],
                    capture_output=True, text=True, cwd=str(self._project_dir)
                )
                if result.returncode == 0:
                    self._log("标定完成!", "ok")
                    self._set_status("标定完成", FG_OK)
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            self._log(line.strip(), "info")
                    self._root.after(0, self._run_env_check)
                else:
                    self._log(f"标定失败 (退出码 {result.returncode})", "error")
                    stderr = result.stderr.strip() if result.stderr else ""
                    stdout = result.stdout.strip() if result.stdout else ""
                    for line in (stderr or stdout).split("\n")[-15:]:
                        if line.strip():
                            self._log(line.strip(), "error")
                    self._set_status("标定失败", FG_ERROR)
            except Exception as e:
                self._log(f"标定错误: {e}", "error")
                self._set_status("标定错误", FG_ERROR)
            finally:
                self._root.after(0, self._restart_camera_preview)

        self._run_in_thread(_run)

    def _on_calibrate_imu(self) -> None:
        proceed = messagebox.askokcancel(
            "IMU T-Pose 标定说明",
            "标定方式：T字形姿态\n\n"
            "步骤：\n"
            "1. 确保所有 IMU 传感器已连接\n"
            "2. 站立，双臂向两侧平伸（T字形）\n"
            "3. 保持静止约 3 秒\n"
            "4. 采集过程中不要移动\n\n"
            "此标定用于校正 IMU 的初始姿态偏移，\n"
            "提升动作融合精度。\n\n"
            "点击确定开始标定。"
        )
        if not proceed:
            return

        self._log("开始 IMU T-Pose 标定...", "info")
        self._set_status("IMU标定中...", FG_WARN)

        def _run() -> None:
            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "scripts.calibrate_imu",
                        "--config", self._config_path,
                        "--output", "config/calibration/",
                        "--hold-duration", "3.0",
                    ],
                    input="\n",
                    capture_output=True, text=True, cwd=str(self._project_dir)
                )
                if result.returncode == 0:
                    self._log("IMU 标定完成!", "ok")
                    self._set_status("IMU标定完成", FG_OK)
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            self._log(line.strip(), "info")
                    self._root.after(0, self._run_env_check)
                else:
                    self._log(f"IMU 标定失败 (退出码 {result.returncode})", "error")
                    stderr = result.stderr.strip() if result.stderr else ""
                    for line in stderr.split("\n")[-10:]:
                        if line.strip():
                            self._log(line.strip(), "error")
                    self._set_status("IMU标定失败", FG_ERROR)
            except Exception as e:
                self._log(f"IMU 标定错误: {e}", "error")
                self._set_status("IMU标定错误", FG_ERROR)

        self._run_in_thread(_run)

    def _on_record(self) -> None:
        from src.environment import EnvironmentChecker
        checker = EnvironmentChecker(self._config_path)
        report = checker.run_full_check()

        selected_cams = self._get_selected_cameras()
        if not selected_cams:
            messagebox.showinfo("未选择摄像头", "请至少启用一个摄像头。")
            return

        if not report.has_cameras:
            messagebox.showinfo(
                "无摄像头",
                "未检测到摄像头。\n请连接USB摄像头并点击刷新，\n或使用演示模式进行测试。"
            )
            return

        if not report.has_calibration:
            if not messagebox.askyesno(
                "无标定数据",
                "未找到标定数据。\n未标定录制将产生较低质量的结果。\n是否继续？"
            ):
                return

        mode_cn = self._get_mode_cn()
        mode = self._mode_var.get()
        selected_imus = self._get_selected_imus()
        cam_str = ",".join(str(c) for c in selected_cams)
        imu_str = ",".join(str(i) for i in selected_imus)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"recordings/session_{timestamp}.bvh"

        self._log(f"录制 [{mode_cn}] 摄像头={cam_str} IMU={imu_str}", "info")
        self._set_status("录制中...", FG_ERROR)
        self._recording = True
        self._update_info("mode", "录制中", FG_ERROR)

        def _run() -> None:
            self._root.after(0, self._stop_camera_preview)
            time.sleep(0.5)

            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "scripts.record",
                        "--config", self._config_path,
                        "--calibration", "config/calibration/",
                        "--output", output,
                        "--mode", mode,
                    ],
                    capture_output=True, text=True, cwd=str(self._project_dir)
                )
                if result.returncode == 0:
                    self._log(f"录制已保存: {output}", "ok")
                    self._set_status("录制完成", FG_OK)
                    stdout = result.stdout.strip() if result.stdout else ""
                    for line in stdout.split("\n"):
                        if "quality" in line.lower() or "质量" in line:
                            self._log(line.strip(), "warn")
                else:
                    self._log("录制失败", "error")
                    stderr = result.stderr.strip() if result.stderr else ""
                    for line in stderr.split("\n")[-10:]:
                        if line.strip():
                            self._log(line.strip(), "error")
                    self._set_status("录制失败", FG_ERROR)
            except Exception as e:
                self._log(f"录制错误: {e}", "error")
                self._set_status("录制错误", FG_ERROR)
            finally:
                self._recording = False
                self._root.after(0, lambda: self._update_info("mode", "空闲", FG_TEXT))
                self._root.after(0, self._restart_camera_preview)

        self._run_in_thread(_run)

    def _on_process(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择原始JSON数据文件",
            initialdir=str(self._project_dir / "recordings"),
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        output = Path(file_path).stem.replace("_raw", "") + "_smoothed.bvh"
        output_path = str(self._project_dir / "recordings" / output)

        self._log(f"正在处理: {file_path}", "info")
        self._set_status("处理中...", FG_WARN)
        self._update_info("mode", "处理中", FG_WARN)

        def _run() -> None:
            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "scripts.process",
                        "--config", self._config_path,
                        "--input", file_path,
                        "--output", output_path,
                    ],
                    capture_output=True, text=True, cwd=str(self._project_dir)
                )
                if result.returncode == 0:
                    self._log(f"处理完成: {output_path}", "ok")
                    self._set_status("处理完成", FG_OK)
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            self._log(line.strip(), "info")
                else:
                    self._log("处理失败", "error")
                    for line in result.stderr.strip().split("\n")[-5:]:
                        if line.strip():
                            self._log(line.strip(), "error")
                    self._set_status("处理失败", FG_ERROR)
            except Exception as e:
                self._log(f"处理错误: {e}", "error")
                self._set_status("处理错误", FG_ERROR)
            finally:
                self._root.after(0, lambda: self._update_info("mode", "空闲", FG_TEXT))

        self._run_in_thread(_run)

    def _on_run_tests(self) -> None:
        self._log("正在运行测试...", "info")
        self._set_status("测试中...", FG_WARN)

        def _run() -> None:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/", "-v"],
                    capture_output=True, text=True, cwd=str(self._project_dir),
                    timeout=120
                )
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        lvl = "ok" if "passed" in line.lower() else "info"
                        if "failed" in line.lower() or "error" in line.lower():
                            lvl = "error"
                        self._log(line.strip(), lvl)
                if result.returncode == 0:
                    self._log("所有测试通过!", "ok")
                    self._set_status("测试通过", FG_OK)
                else:
                    self._log("部分测试失败", "warn")
                    self._set_status("测试失败", FG_ERROR)
            except subprocess.TimeoutExpired:
                self._log("测试超时 (120秒)", "error")
                self._set_status("测试超时", FG_ERROR)
            except Exception as e:
                self._log(f"测试错误: {e}", "error")
                self._set_status("测试错误", FG_ERROR)

        self._run_in_thread(_run)

    # ── Demo ───────────────────────────────────────────────────────────
    def _on_play_demo(self) -> None:
        motion = self._demo_var.get()
        speed = self._speed_var.get()
        num_frames = 120

        cn_motion = {"walk": "行走", "wave": "挥手", "squat": "下蹲"}.get(motion, motion)
        self._log(f"生成演示: {cn_motion} (速度={speed:.1f}x)", "info")
        self._set_status("生成演示...", FG_WARN)

        def _gen() -> None:
            try:
                from src.demo_data import DemoDataGenerator
                gen = DemoDataGenerator(fps=30)
                self._demo_frames = gen.get_demo_sequence(motion, num_frames)
                self._demo_playing = True
                self._demo_index = 0
                self._root.after(0, lambda: self._stop_demo_btn.config(state=tk.NORMAL))
                self._update_info("mode", "演示", FG_ACCENT)
                self._update_info("motion", cn_motion, FG_ACCENT)
                self._log(f"演示就绪: {len(self._demo_frames)} 帧", "ok")
                self._set_status(f"播放: {cn_motion}", FG_ACCENT)
                self._play_next_frame()
            except Exception as e:
                self._log(f"演示生成错误: {e}", "error")
                self._set_status("演示错误", FG_ERROR)

        self._run_in_thread(_gen)

    def _play_next_frame(self) -> None:
        if not self._demo_playing or self._demo_index >= len(self._demo_frames):
            self._on_stop_demo()
            return

        frame = self._demo_frames[self._demo_index]
        self._skeleton_canvas.update(frame, self._demo_index)
        self._update_info("frames", str(self._demo_index + 1))

        for i, dot in enumerate(self._kp_indicators):
            dot.config(fg=FG_OK if i < len(frame) else FG_DIM)

        self._demo_index += 1
        delay = max(10, int(1000 / (30 * self._speed_var.get())))
        self._root.after(delay, self._play_next_frame)

    def _on_stop_demo(self) -> None:
        self._demo_playing = False
        self._stop_demo_btn.config(state=tk.DISABLED)
        self._update_info("mode", "空闲", FG_TEXT)
        self._set_status("演示已停止", FG_TEXT)
        for dot in self._kp_indicators:
            dot.config(fg=FG_DIM)

    def _on_export_demo_bvh(self) -> None:
        motion = self._demo_var.get()
        num_frames = 120
        cn_motion = {"walk": "行走", "wave": "挥手", "squat": "下蹲"}.get(motion, motion)
        self._log(f"导出演示为BVH: {cn_motion}", "info")
        self._set_status("导出BVH...", FG_WARN)

        def _run() -> None:
            try:
                from src.demo_data import DemoDataGenerator
                gen = DemoDataGenerator(fps=30)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                bvh_path = gen.generate_to_bvh(
                    motion, num_frames, f"recordings/demo_{motion}_{ts}.bvh"
                )
                raw_path = gen.save_raw_json(
                    motion, num_frames, f"recordings/demo_{motion}_{ts}_raw.json"
                )
                self._log(f"BVH 已导出: {bvh_path}", "ok")
                self._log(f"原始数据已保存: {raw_path}", "ok")
                self._set_status("导出完成", FG_OK)
            except Exception as e:
                self._log(f"导出错误: {e}", "error")
                self._set_status("导出失败", FG_ERROR)

        self._run_in_thread(_run)

    # ── Lifecycle ──────────────────────────────────────────────────────
    def _on_close(self) -> None:
        self._demo_playing = False
        self._cam_preview_running = False
        self._pose_running = False
        if self._cam_manager is not None:
            try:
                self._cam_manager.stop()
            except Exception:
                pass
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()


def main() -> None:
    app = MovCapApp()
    app.run()


if __name__ == "__main__":
    main()
