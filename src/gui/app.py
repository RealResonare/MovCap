import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk, filedialog
from typing import Optional

import numpy as np
import yaml

from .skeleton_canvas import SkeletonCanvas


BG_DARK = "#0f0f23"
BG_PANEL = "#1a1a2e"
BG_BUTTON = "#16213e"
BG_BUTTON_HOVER = "#1a1a4e"
FG_TEXT = "#e0e0e0"
FG_DIM = "#666688"
FG_ACCENT = "#00d4ff"
FG_OK = "#00ff88"
FG_WARN = "#ffcc00"
FG_ERROR = "#ff4444"


class MovCapApp:
    def __init__(self, config_path: str = "config/default.yaml") -> None:
        self._config_path = config_path
        self._project_dir = Path(__file__).parent.parent.parent
        os.chdir(self._project_dir)

        self._root = tk.Tk()
        self._root.title("MovCap - Visual-Inertial Motion Capture System")
        self._root.geometry("1100x750")
        self._root.minsize(900, 600)
        self._root.configure(bg=BG_DARK)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._demo_frames: list[np.ndarray] = []
        self._demo_playing = False
        self._demo_index = 0
        self._recording = False
        self._running_task: Optional[threading.Thread] = None

        self._setup_styles()
        self._build_ui()
        self._log("MovCap UI initialized")

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
            "Title.TLabel", background=BG_DARK, foreground=FG_ACCENT,
            font=("Segoe UI", 16, "bold")
        )
        style.configure(
            "Status.TLabel", background=BG_PANEL, foreground=FG_TEXT,
            font=("Consolas", 9)
        )
        style.configure(
            "Dark.TButton", background=BG_BUTTON, foreground=FG_TEXT,
            font=("Segoe UI", 10), padding=(12, 6)
        )
        style.map(
            "Dark.TButton",
            background=[("active", BG_BUTTON_HOVER)],
            foreground=[("active", FG_ACCENT)]
        )
        style.configure(
            "Accent.TButton", background="#0066cc", foreground="#ffffff",
            font=("Segoe UI", 10, "bold"), padding=(12, 6)
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#0088ff")]
        )
        style.configure(
            "Danger.TButton", background="#cc3333", foreground="#ffffff",
            font=("Segoe UI", 10, "bold"), padding=(12, 6)
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=BG_PANEL, background=FG_ACCENT
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self._root, bg=BG_DARK, height=50)
        header.pack(fill=tk.X, padx=10, pady=(10, 0))

        tk.Label(
            header, text="MovCap", bg=BG_DARK, fg=FG_ACCENT,
            font=("Segoe UI", 20, "bold")
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text="Visual-Inertial Motion Capture System",
            bg=BG_DARK, fg=FG_DIM, font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=(15, 0))

        self._status_label = tk.Label(
            header, text="Ready", bg=BG_DARK, fg=FG_OK,
            font=("Segoe UI", 10), anchor="e"
        )
        self._status_label.pack(side=tk.RIGHT)

        main_paned = tk.PanedWindow(
            self._root, orient=tk.HORIZONTAL, bg=BG_DARK,
            sashwidth=4, sashrelief=tk.FLAT
        )
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = tk.Frame(main_paned, bg=BG_DARK, width=320)
        main_paned.add(left_panel, minsize=280, width=340)

        right_panel = tk.Frame(main_paned, bg=BG_DARK)
        main_paned.add(right_panel, minsize=400)

        self._build_env_panel(left_panel)
        self._build_control_panel(left_panel)
        self._build_right_panel(right_panel)

    def _build_env_panel(self, parent: tk.Widget) -> None:
        env_frame = tk.LabelFrame(
            parent, text=" Environment Check ", bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 10, "bold"), relief=tk.GROOVE, bd=1
        )
        env_frame.pack(fill=tk.X, padx=5, pady=5)

        self._env_tree = ttk.Treeview(
            env_frame, columns=("status", "value"), show="tree",
            height=8, selectmode="browse"
        )
        self._env_tree.column("#0", width=140, minwidth=100)
        self._env_tree.column("status", width=40, minwidth=30, anchor="center")
        self._env_tree.column("value", width=160, minwidth=100)

        self._env_tree.heading("#0", text="Component", anchor="w")
        self._env_tree.heading("status", text="", anchor="center")
        self._env_tree.heading("value", text="Status", anchor="w")

        self._env_tree.tag_configure("ok", foreground=FG_OK)
        self._env_tree.tag_configure("warn", foreground=FG_WARN)
        self._env_tree.tag_configure("error", foreground=FG_ERROR)
        self._env_tree.tag_configure("info", foreground=FG_DIM)

        scrollbar = ttk.Scrollbar(
            env_frame, orient=tk.VERTICAL, command=self._env_tree.yview
        )
        self._env_tree.configure(yscrollcommand=scrollbar.set)

        self._env_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        btn_frame = tk.Frame(env_frame, bg=BG_PANEL)
        btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Button(
            btn_frame, text="Refresh", style="Dark.TButton",
            command=self._run_env_check
        ).pack(side=tk.LEFT)

        self._env_summary = tk.Label(
            btn_frame, text="", bg=BG_PANEL, fg=FG_DIM,
            font=("Consolas", 9)
        )
        self._env_summary.pack(side=tk.RIGHT)

        self._root.after(500, self._run_env_check)

    def _build_control_panel(self, parent: tk.Widget) -> None:
        ctrl_frame = tk.LabelFrame(
            parent, text=" Controls ", bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 10, "bold"), relief=tk.GROOVE, bd=1
        )
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        btn_configs = [
            ("Calibrate Cameras", self._on_calibrate, "Accent.TButton"),
            ("Record Session", self._on_record, "Accent.TButton"),
            ("Process Data", self._on_process, "Dark.TButton"),
            ("Run Tests", self._on_run_tests, "Dark.TButton"),
        ]

        for text, cmd, style in btn_configs:
            ttk.Button(
                ctrl_frame, text=text, style=style, command=cmd
            ).pack(fill=tk.X, padx=8, pady=3)

        sep = tk.Frame(ctrl_frame, bg="#333355", height=1)
        sep.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(
            ctrl_frame, text="Demo Mode (No Hardware)",
            bg=BG_PANEL, fg=FG_WARN, font=("Segoe UI", 9, "bold")
        ).pack(padx=8, pady=(0, 4))

        demo_frame = tk.Frame(ctrl_frame, bg=BG_PANEL)
        demo_frame.pack(fill=tk.X, padx=8, pady=3)

        self._demo_var = tk.StringVar(value="walk")
        for motion in ["walk", "wave", "squat"]:
            tk.Radiobutton(
                demo_frame, text=motion.capitalize(), variable=self._demo_var,
                value=motion, bg=BG_PANEL, fg=FG_TEXT, selectcolor=BG_BUTTON,
                activebackground=BG_PANEL, activeforeground=FG_ACCENT,
                font=("Segoe UI", 9)
            ).pack(side=tk.LEFT, padx=(0, 10))

        demo_btn_frame = tk.Frame(ctrl_frame, bg=BG_PANEL)
        demo_btn_frame.pack(fill=tk.X, padx=8, pady=3)

        ttk.Button(
            demo_btn_frame, text="Play Demo", style="Accent.TButton",
            command=self._on_play_demo
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self._stop_demo_btn = ttk.Button(
            demo_btn_frame, text="Stop", style="Danger.TButton",
            command=self._on_stop_demo, state=tk.DISABLED
        )
        self._stop_demo_btn.pack(side=tk.LEFT, padx=(3, 0))

        speed_frame = tk.Frame(ctrl_frame, bg=BG_PANEL)
        speed_frame.pack(fill=tk.X, padx=8, pady=3)

        tk.Label(
            speed_frame, text="Speed:", bg=BG_PANEL, fg=FG_DIM,
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT)

        self._speed_var = tk.DoubleVar(value=1.0)
        speed_scale = tk.Scale(
            speed_frame, from_=0.1, to=3.0, resolution=0.1,
            orient=tk.HORIZONTAL, variable=self._speed_var,
            bg=BG_PANEL, fg=FG_TEXT, highlightthickness=0,
            troughcolor=BG_BUTTON, activebackground=FG_ACCENT,
            font=("Segoe UI", 8), length=150, showvalue=True,
            digits=2
        )
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        ttk.Button(
            ctrl_frame, text="Export Demo to BVH", style="Dark.TButton",
            command=self._on_export_demo_bvh
        ).pack(fill=tk.X, padx=8, pady=3)

    def _build_right_panel(self, parent: tk.Widget) -> None:
        top_frame = tk.Frame(parent, bg=BG_DARK)
        top_frame.pack(fill=tk.BOTH, expand=True)

        viz_frame = tk.LabelFrame(
            top_frame, text=" Skeleton Preview ", bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 10, "bold"), relief=tk.GROOVE, bd=1
        )
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self._skeleton_canvas = SkeletonCanvas(viz_frame, width=480, height=480)

        right_info = tk.Frame(top_frame, bg=BG_DARK, width=250)
        right_info.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_info.pack_propagate(False)

        self._build_info_panel(right_info)
        self._build_log_panel(parent)

    def _build_info_panel(self, parent: tk.Widget) -> None:
        info_frame = tk.LabelFrame(
            parent, text=" Status ", bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 10, "bold"), relief=tk.GROOVE, bd=1
        )
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._info_labels: dict[str, tk.Label] = {}
        info_items = [
            ("mode", "Mode", "Idle"),
            ("frames", "Frames", "0"),
            ("fps", "FPS", "--"),
            ("motion", "Motion", "--"),
        ]

        for key, label, default in info_items:
            row = tk.Frame(info_frame, bg=BG_PANEL)
            row.pack(fill=tk.X, padx=8, pady=2)

            tk.Label(
                row, text=f"{label}:", bg=BG_PANEL, fg=FG_DIM,
                font=("Segoe UI", 9), anchor="w", width=8
            ).pack(side=tk.LEFT)

            lbl = tk.Label(
                row, text=default, bg=BG_PANEL, fg=FG_TEXT,
                font=("Consolas", 9, "bold"), anchor="w"
            )
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._info_labels[key] = lbl

        sep = tk.Frame(info_frame, bg="#333355", height=1)
        sep.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(
            info_frame, text="Keypoints (COCO 17):",
            bg=BG_PANEL, fg=FG_DIM, font=("Segoe UI", 9, "bold")
        ).pack(padx=8, anchor="w")

        kp_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]

        kp_frame = tk.Frame(info_frame, bg=BG_PANEL)
        kp_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 5))

        kp_canvas = tk.Canvas(
            kp_frame, bg=BG_PANEL, highlightthickness=0, width=200
        )
        kp_scroll = ttk.Scrollbar(
            kp_frame, orient=tk.VERTICAL, command=kp_canvas.yview
        )
        kp_inner = tk.Frame(kp_canvas, bg=BG_PANEL)

        kp_inner.bind(
            "<Configure>",
            lambda e: kp_canvas.configure(scrollregion=kp_canvas.bbox("all"))
        )
        kp_canvas.create_window((0, 0), window=kp_inner, anchor="nw")
        kp_canvas.configure(yscrollcommand=kp_scroll.set)

        self._kp_indicators = []
        for i, name in enumerate(kp_names):
            row = tk.Frame(kp_inner, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=1)

            dot = tk.Label(
                row, text="\u25cf", bg=BG_PANEL, fg=FG_DIM,
                font=("Segoe UI", 7), width=2
            )
            dot.pack(side=tk.LEFT)
            self._kp_indicators.append(dot)

            tk.Label(
                row, text=f"{i}: {name}", bg=BG_PANEL, fg=FG_DIM,
                font=("Consolas", 8), anchor="w"
            ).pack(side=tk.LEFT)

        kp_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        kp_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_log_panel(self, parent: tk.Widget) -> None:
        log_frame = tk.LabelFrame(
            parent, text=" Log ", bg=BG_PANEL, fg=FG_ACCENT,
            font=("Segoe UI", 10, "bold"), relief=tk.GROOVE, bd=1
        )
        log_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self._log_text = scrolledtext.ScrolledText(
            log_frame, height=6, bg="#0a0a1a", fg=FG_TEXT,
            font=("Consolas", 9), insertbackground=FG_ACCENT,
            selectbackground="#333366", relief=tk.FLAT, wrap=tk.WORD,
            state=tk.DISABLED
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._log_text.tag_configure("info", foreground=FG_TEXT)
        self._log_text.tag_configure("ok", foreground=FG_OK)
        self._log_text.tag_configure("warn", foreground=FG_WARN)
        self._log_text.tag_configure("error", foreground=FG_ERROR)
        self._log_text.tag_configure("time", foreground=FG_DIM)

    def _log(self, message: str, level: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")

        def _insert() -> None:
            self._log_text.configure(state=tk.NORMAL)
            self._log_text.insert(tk.END, f"[{timestamp}] ", "time")
            self._log_text.insert(tk.END, f"{message}\n", level)
            self._log_text.see(tk.END)
            self._log_text.configure(state=tk.DISABLED)

        self._root.after(0, _insert)

    def _set_status(self, text: str, color: str = FG_OK) -> None:
        self._root.after(0, lambda: self._status_label.config(text=text, fg=color))

    def _update_info(self, key: str, value: str, color: str = FG_TEXT) -> None:
        def _update() -> None:
            if key in self._info_labels:
                self._info_labels[key].config(text=value, fg=color)
        self._root.after(0, _update)

    def _run_env_check(self) -> None:
        self._log("Running environment check...")
        self._set_status("Checking environment...", FG_WARN)

        def _check() -> None:
            try:
                from src.environment import EnvironmentChecker
                checker = EnvironmentChecker(self._config_path)
                report = checker.run_full_check()

                self._root.after(0, lambda: self._populate_env_tree(report))

                ok_count = sum(1 for c in report.checks if c.status.value == "ok")
                total = len(report.checks)
                warn_count = sum(1 for c in report.checks if c.status.value == "warn")
                err_count = sum(1 for c in report.checks if c.status.value == "error")

                summary = f"{ok_count}/{total} OK"
                if warn_count:
                    summary += f" | {warn_count} warn"
                if err_count:
                    summary += f" | {err_count} error"

                self._root.after(0, lambda: self._env_summary.config(text=summary))

                if report.can_demo:
                    self._set_status("Environment OK (demo mode available)", FG_OK)
                elif err_count > 0:
                    self._set_status(f"Environment issues found ({err_count} errors)", FG_ERROR)
                else:
                    self._set_status(f"Environment checked ({warn_count} warnings)", FG_WARN)

                self._log(f"Environment check complete: {summary}", "ok")

            except Exception as e:
                self._log(f"Environment check failed: {e}", "error")
                self._set_status("Environment check failed", FG_ERROR)

        threading.Thread(target=_check, daemon=True).start()

    def _populate_env_tree(self, report) -> None:
        for item in self._env_tree.get_children():
            self._env_tree.delete(item)

        status_icons = {
            "ok": "\u2713",
            "warn": "!",
            "error": "\u2717",
            "info": "i",
        }

        categories = {
            "System": [],
            "Packages": [],
            "Hardware": [],
            "Calibration": [],
            "Config": [],
        }

        for check in report.checks:
            name = check.name.lower()
            if name in ("python version", "disk space"):
                categories["System"].append(check)
            elif any(pkg in name for pkg in (
                "numpy", "scipy", "opencv", "pyyaml", "pyserial",
                "filterpy", "ultralytics", "open3d", "bvhsdk", "torch", "matplotlib",
                "cuda", "gpu"
            )):
                categories["Packages"].append(check)
            elif name in ("cameras", "camera ids", "imu ports", "imu missing"):
                categories["Hardware"].append(check)
            elif "calibrat" in name:
                categories["Calibration"].append(check)
            else:
                categories["Config"].append(check)

        for cat_name, checks in categories.items():
            if not checks:
                continue

            parent = self._env_tree.insert(
                "", tk.END, text=cat_name, values=("", ""),
                tags=("info",)
            )

            for check in checks:
                status_icon = status_icons.get(check.status.value, "?")
                tag = check.status.value
                self._env_tree.insert(
                    parent, tk.END, text=check.name,
                    values=(status_icon, check.message),
                    tags=(tag,)
                )

            self._env_tree.item(parent, open=True)

    def _run_in_thread(self, func, *args) -> None:
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("Busy", "A task is already running.")
            return

        self._running_task = threading.Thread(target=func, args=args, daemon=True)
        self._running_task.start()

    def _on_calibrate(self) -> None:
        duration = 30
        self._log(f"Starting camera calibration ({duration}s)...", "info")
        self._set_status("Calibrating...", FG_WARN)

        def _calibrate() -> None:
            try:
                self._log("Running: scripts.calibrate", "info")
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
                    self._log("Calibration complete!", "ok")
                    self._set_status("Calibration complete", FG_OK)
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            self._log(line.strip(), "info")
                    self._root.after(0, self._run_env_check)
                else:
                    self._log(f"Calibration failed (exit code {result.returncode})", "error")
                    for line in result.stderr.strip().split("\n")[-10:]:
                        if line.strip():
                            self._log(line.strip(), "error")
                    self._set_status("Calibration failed", FG_ERROR)

            except Exception as e:
                self._log(f"Calibration error: {e}", "error")
                self._set_status("Calibration error", FG_ERROR)

        self._run_in_thread(_calibrate)

    def _on_record(self) -> None:
        from src.environment import EnvironmentChecker
        checker = EnvironmentChecker(self._config_path)
        report = checker.run_full_check()

        if not report.has_cameras:
            messagebox.showinfo(
                "No Cameras",
                "No cameras detected.\n\n"
                "Connect USB cameras and click Refresh,\n"
                "or use Demo Mode to test without hardware."
            )
            return

        if not report.has_calibration:
            if not messagebox.askyesno(
                "No Calibration",
                "No calibration data found.\n\n"
                "Recording without calibration will produce lower quality results.\n"
                "Continue anyway?"
            ):
                return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"recordings/session_{timestamp}.bvh"

        self._log(f"Starting recording: {output}", "info")
        self._set_status("Recording...", FG_ERROR)
        self._recording = True
        self._update_info("mode", "Recording", FG_ERROR)

        def _record() -> None:
            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "scripts.record",
                        "--config", self._config_path,
                        "--calibration", "config/calibration/",
                        "--output", output,
                    ],
                    capture_output=True, text=True, cwd=str(self._project_dir)
                )

                if result.returncode == 0:
                    self._log(f"Recording saved: {output}", "ok")
                    self._set_status("Recording complete", FG_OK)
                else:
                    self._log("Recording failed", "error")
                    for line in result.stderr.strip().split("\n")[-5:]:
                        if line.strip():
                            self._log(line.strip(), "error")
                    self._set_status("Recording failed", FG_ERROR)

            except Exception as e:
                self._log(f"Recording error: {e}", "error")
                self._set_status("Recording error", FG_ERROR)
            finally:
                self._recording = False
                self._root.after(0, lambda: self._update_info("mode", "Idle", FG_TEXT))

        self._run_in_thread(_record)

    def _on_process(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select raw JSON data file",
            initialdir=str(self._project_dir / "recordings"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        output = Path(file_path).stem.replace("_raw", "") + "_smoothed.bvh"
        output_path = str(self._project_dir / "recordings" / output)

        self._log(f"Processing: {file_path}", "info")
        self._set_status("Processing...", FG_WARN)
        self._update_info("mode", "Processing", FG_WARN)

        def _process() -> None:
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
                    self._log(f"Processing complete: {output_path}", "ok")
                    self._set_status("Processing complete", FG_OK)
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            self._log(line.strip(), "info")
                else:
                    self._log("Processing failed", "error")
                    for line in result.stderr.strip().split("\n")[-5:]:
                        if line.strip():
                            self._log(line.strip(), "error")
                    self._set_status("Processing failed", FG_ERROR)

            except Exception as e:
                self._log(f"Processing error: {e}", "error")
                self._set_status("Processing error", FG_ERROR)
            finally:
                self._root.after(0, lambda: self._update_info("mode", "Idle", FG_TEXT))

        self._run_in_thread(_process)

    def _on_run_tests(self) -> None:
        self._log("Running tests...", "info")
        self._set_status("Testing...", FG_WARN)

        def _test() -> None:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/", "-v"],
                    capture_output=True, text=True, cwd=str(self._project_dir),
                    timeout=120
                )

                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        level = "ok" if "passed" in line.lower() else "info"
                        if "failed" in line.lower() or "error" in line.lower():
                            level = "error"
                        self._log(line.strip(), level)

                if result.returncode == 0:
                    self._log("All tests passed!", "ok")
                    self._set_status("Tests passed", FG_OK)
                else:
                    self._log("Some tests failed", "warn")
                    self._set_status("Tests failed", FG_ERROR)

            except subprocess.TimeoutExpired:
                self._log("Tests timed out (120s)", "error")
                self._set_status("Tests timed out", FG_ERROR)
            except Exception as e:
                self._log(f"Test error: {e}", "error")
                self._set_status("Test error", FG_ERROR)

        self._run_in_thread(_test)

    def _on_play_demo(self) -> None:
        motion = self._demo_var.get()
        speed = self._speed_var.get()
        num_frames = 120

        self._log(f"Generating demo: {motion} (speed={speed:.1f}x)", "info")
        self._set_status("Generating demo...", FG_WARN)

        def _generate() -> None:
            try:
                from src.demo_data import DemoDataGenerator
                gen = DemoDataGenerator(fps=30)
                self._demo_frames = gen.get_demo_sequence(motion, num_frames)
                self._demo_playing = True
                self._demo_index = 0

                self._root.after(0, lambda: self._stop_demo_btn.config(state=tk.NORMAL))
                self._update_info("mode", "Demo", FG_ACCENT)
                self._update_info("motion", motion.capitalize(), FG_ACCENT)
                self._log(f"Demo ready: {len(self._demo_frames)} frames", "ok")
                self._set_status(f"Playing demo: {motion}", FG_ACCENT)

                self._play_next_frame()

            except Exception as e:
                self._log(f"Demo generation error: {e}", "error")
                self._set_status("Demo error", FG_ERROR)

        self._run_in_thread(_generate)

    def _play_next_frame(self) -> None:
        if not self._demo_playing or self._demo_index >= len(self._demo_frames):
            self._on_stop_demo()
            return

        frame = self._demo_frames[self._demo_index]
        self._skeleton_canvas.update(frame, self._demo_index)
        self._update_info("frames", str(self._demo_index + 1))

        for i, dot in enumerate(self._kp_indicators):
            if i < len(frame):
                dot.config(fg=FG_OK)
            else:
                dot.config(fg=FG_DIM)

        self._demo_index += 1
        speed = self._speed_var.get()
        delay = max(10, int(1000 / (30 * speed)))
        self._root.after(delay, self._play_next_frame)

    def _on_stop_demo(self) -> None:
        self._demo_playing = False
        self._stop_demo_btn.config(state=tk.DISABLED)
        self._update_info("mode", "Idle", FG_TEXT)
        self._set_status("Demo stopped", FG_TEXT)

        for dot in self._kp_indicators:
            dot.config(fg=FG_DIM)

    def _on_export_demo_bvh(self) -> None:
        motion = self._demo_var.get()
        num_frames = 120

        self._log(f"Exporting demo to BVH: {motion}", "info")
        self._set_status("Exporting BVH...", FG_WARN)

        def _export() -> None:
            try:
                from src.demo_data import DemoDataGenerator
                gen = DemoDataGenerator(fps=30)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                bvh_path = gen.generate_to_bvh(
                    motion, num_frames,
                    f"recordings/demo_{motion}_{timestamp}.bvh"
                )
                raw_path = gen.save_raw_json(
                    motion, num_frames,
                    f"recordings/demo_{motion}_{timestamp}_raw.json"
                )

                self._log(f"BVH exported: {bvh_path}", "ok")
                self._log(f"Raw data saved: {raw_path}", "ok")
                self._set_status("Export complete", FG_OK)

            except Exception as e:
                self._log(f"Export error: {e}", "error")
                self._set_status("Export failed", FG_ERROR)

        self._run_in_thread(_export)

    def _on_close(self) -> None:
        self._demo_playing = False
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()


def main() -> None:
    app = MovCapApp()
    app.run()


if __name__ == "__main__":
    main()
