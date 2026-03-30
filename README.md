# MovCap — 视觉-惯性动作捕捉系统

基于 3 个 USB 摄像头和 8 个 BNO055 IMU 传感器的全身动作捕捉系统，输出标准 BVH 骨骼动画文件。

## 系统架构

```
摄像头 (×3) ──→ 2D 姿态估计 (YOLO11-Pose) ──→ 3D 三角测量 ──┐
                                                            ├──→ UKF 融合 ──→ 骨骼 ──→ BVH
IMU (×8) ────→ 四元数 + 加速度 ─────────────────────────────────┘
```

**数据流**：摄像头捕获图像 → YOLO11-Pose 检测 17 个 COCO 关键点 → 多视角三角测量重建 3D 姿态 → IMU 数据预处理 → 无迹卡尔曼滤波 (UKF) 融合视觉与惯性数据 → 骨骼模型解算关节角度 → 导出 BVH 文件

## 硬件要求

| 组件 | 规格 |
|------|------|
| USB 摄像头 | 3× Logitech C920/C922 (推荐) |
| IMU 传感器 | 8× BNO055 (Adafruit/SparkFun 模块) |
| USB 集线器 | 带足够带宽 |
| 标定板 | 打印 ChArUco 棋盘格 |

**IMU 佩戴位置**：
- 0: 头部 | 1: 胸部 | 2: 腰部
- 3: 左上臂 | 4: 右上臂 | 5: 左前臂 | 6: 右前臂 | 7: 左大腿

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 GUI（推荐）

```bash
python -m scripts.gui
```

或在 Windows 上双击 `start.bat` 打开交互式菜单。

### 3. 命令行工作流

```bash
# 生成 ChArUco 标定板 (打印后用于标定)
python -m scripts.calibrate --generate-board --output config/calibration/

# 标定摄像头
python -m scripts.calibrate --output config/calibration/

# 录制动作
python -m scripts.record --duration 30 --output recordings/session1.bvh

# 离线处理 (重新处理已录制数据)
python -m scripts.process --input recordings/session1_raw.json --output session1.bvh
```

## 项目结构

```
MovCap/
├── src/
│   ├── acquisition/      # 摄像头 + IMU 数据采集
│   ├── calibration/      # 摄像头内参/外参标定
│   ├── pose/             # 2D 姿态估计 + 3D 三角测量
│   ├── fusion/           # 视觉-IMU 传感器融合 (UKF)
│   ├── skeleton/         # 骨骼模型 + BVH 导出
│   ├── visualization/    # 实时 3D 可视化
│   ├── gui/              # 图形界面
│   ├── pipeline.py       # 主处理流水线
│   ├── demo_data.py      # 演示数据生成器
│   └── environment.py    # 环境管理
├── scripts/
│   ├── calibrate.py      # 标定脚本
│   ├── record.py         # 录制脚本
│   ├── process.py        # 离线处理脚本
│   └── gui.py            # GUI 启动脚本
├── config/
│   ├── default.yaml      # 主配置文件
│   ├── skeleton_model.yaml  # 骨骼层级定义
│   └── calibration/      # 标定数据存储
├── cpp/
│   ├── triangulation/    # C++ 性能模块 (pybind11)
│   └── bindings/         # Python 绑定
├── tests/                # 测试套件
├── tools/                # 辅助工具
├── pyproject.toml        # 项目元数据
└── start.bat             # Windows 交互式菜单
```

## 配置说明

所有参数集中在 `config/default.yaml`：

| 配置项 | 说明 |
|--------|------|
| `cameras.devices` | 摄像头设备 ID 列表 |
| `cameras.resolution` | 分辨率 (默认 1280×720) |
| `cameras.fps` | 帧率 (默认 30) |
| `imu.ports` | IMU 串口列表 (COM3-COM10) |
| `imu.sample_rate_hz` | IMU 采样率 (默认 100Hz) |
| `pose2d.model` | YOLO11 姿态模型 (首次运行自动下载) |
| `pose2d.confidence_threshold` | 关键点置信度阈值 |
| `fusion.type` | 融合算法: `ukf` 或 `ekf` |
| `temporal_filter.type` | 时域滤波器: `savgol` |

## 功能特性

- **多视角重建**：至少 2 个摄像头可见即可三角测量 3D 关键点
- **传感器融合**：UKF/EKF 融合视觉与 IMU 数据，补偿遮挡
- **实时可视化**：Open3D 3D 查看器实时显示骨骼
- **BVH 导出**：标准 BVH 格式，兼容 Blender/Maya 等软件
- **GUI 界面**：Tkinter 图形界面，支持录制/回放/导出
- **演示模式**：内置行走/挥手/深蹲等合成动画，无需硬件即可测试
- **时域滤波**：Savitzky-Golay 滤波器平滑输出
- **标定工具**：ChArUco 板自动生成与摄像头标定

## 测试

```bash
pytest tests/ -v
```

测试覆盖：标定、三角测量、传感器融合、BVH 导出。

## 依赖

| 包 | 用途 |
|----|------|
| ultralytics | YOLO11-Pose 姿态估计 |
| opencv-python | 图像采集与处理 |
| numpy / scipy | 数值计算 |
| filterpy | 卡尔曼滤波 |
| pyserial | IMU 串口通信 |
| bvhsdk | BVH 文件读写 |
| open3d | 3D 可视化 |
| pybind11 | C++ 性能模块绑定 |

## 许可证

MIT License
