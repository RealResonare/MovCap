# MovCap 核心代码解析

> Visual-Inertial Motion Capture System: 3 USB cameras + 8 BNO055 IMUs → BVH output

---

## 目录

1. [多摄像头标定模块](#1-多摄像头标定模块)
2. [YOLO11-Pose 姿态估计](#2-yolo11-pose-姿态估计)
3. [多视角三角测量优化](#3-多视角三角测量优化)
4. [UKF 传感器融合算法](#4-ukf-传感器融合算法)
5. [骨骼模型与 BVH 导出](#5-骨骼模型与-bvh-导出)
6. [实时可视化与 GUI](#6-实时可视化与-gui)
7. [演示模式（无需硬件）](#7-演示模式无需硬件即可运行)
8. [主流水线 Pipeline](#8-主流水线-pipeline)

---

## 1. 多摄像头标定模块

### 1.1 Charuco 标定板检测器

**文件**: `src/calibration/charuco_detector.py`

```python
class CharucoDetector:
    # 使用 ArUco + ChArUco 棋盘格进行高精度角点检测
    # ChArUco 相比传统棋盘格的优势：部分遮挡时仍可检测

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        cal_cfg = cfg["calibration"]
        dict_name = cal_cfg["charuco_dict"]
        dict_id = _ARUCO_DICTS[dict_name]

        # 获取预定义的 ArUco 字典（DICT_4X4_50, DICT_5X5_100 等）
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self._detector_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._detector_params)

        # 创建 ChArUco 标定板：行列格子数、格子物理尺寸、标记物理尺寸
        self._board = cv2.aruco.CharucoBoard(
            size=(cal_cfg["charuco_squares_x"], cal_cfg["charuco_squares_y"]),
            squareLength=cal_cfg["charuco_square_length_m"],
            markerLength=cal_cfg["charuco_marker_length_m"],
            dictionary=self._aruco_dict,
        )

    def detect(self, image: np.ndarray) -> Optional[CharucoDetection]:
        # 第一步：检测 ArUco 标记角点
        corners, ids, rejected = self._detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:  # 至少需要 4 个标记
            return None

        # 第二步：插值获取 ChArUco 精确角点（亚像素精度）
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self._board
        )

        # 获取对应的世界坐标点（3D），用于后续标定
        obj_pts = self._board.getChessboardCorners()[charuco_ids.flatten()]

        return CharucoDetection(
            corners=corners, ids=ids,
            image_points=charuco_corners,   # 2D 图像坐标
            object_points=obj_pts,          # 3D 世界坐标
            rejected=rejected,
        )
```

### 1.2 内参标定

**文件**: `src/calibration/intrinsic_calib.py`

```python
class IntrinsicCalibrator:
    # 单目相机内参标定：从多帧 ChArUco 检测结果中求解相机矩阵和畸变系数

    def calibrate(
        self, detections: list[CharucoDetection], camera_id: int, image_size: tuple[int, int]
    ) -> IntrinsicResult:
        all_corners: list[np.ndarray] = []
        all_ids: list[np.ndarray] = []

        for det in detections:
            all_corners.append(det.image_points)
            all_ids.append(det.ids)

        # 调用 OpenCV 的 ChArUco 标定函数
        # 返回: 重投影误差(ret), 相机矩阵(mtx), 畸变系数(dist), 旋转向量(rvecs), 平移向量(tvecs)
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=self._detector.board,
            imageSize=image_size,
            cameraMatrix=None,  # 自动初始化
            distCoeffs=None,
        )

        # 检查重投影误差是否超过阈值
        if ret > self._max_error:
            logger.warning("Camera %d reproj error %.3fpx > threshold %.3fpx",
                          camera_id, ret, self._max_error)

        result = IntrinsicResult(
            camera_id=camera_id,
            camera_matrix=mtx,       # 3x3 相机内参矩阵 [fx,0,cx; 0,fy,cy; 0,0,1]
            dist_coeffs=dist,        # 畸变系数 (k1, k2, p1, p2, k3)
            reprojection_error=ret,  # 重投影误差（像素），越小越好
            image_size=image_size,
        )
        self._calibrations[camera_id] = result
        return result
```

### 1.3 外参标定（双目标定）

**文件**: `src/calibration/extrinsic_calib.py`

```python
class ExtrinsicCalibrator:
    # 双目相机外参标定：求解两个相机之间的旋转 R 和平移 T

    def calibrate_pair(
        self,
        detections_1: list[CharucoDetection],  # 相机1的检测结果
        detections_2: list[CharucoDetection],  # 相机2的检测结果
        intrinsic_1: IntrinsicResult,          # 相机1的内参
        intrinsic_2: IntrinsicResult,          # 相机2的内参
        camera_id_1: int,
        camera_id_2: int,
    ) -> StereoResult:
        # 匹配两相机在同一帧中看到的共同 ChArUco 角点
        for det1 in detections_1:
            for det2 in detections_2:
                common_ids = np.intersect1d(det1.ids.flatten(), det2.ids.flatten())
                if len(common_ids) < 4:
                    continue
                # 提取共同角点在两个图像中的像素坐标
                for cid in common_ids:
                    idx1 = np.where(det1.ids.flatten() == cid)[0][0]
                    idx2 = np.where(det2.ids.flatten() == cid)[0][0]
                    pts1.append(det1.image_points[idx1])
                    pts2.append(det2.image_points[idx2])

        # 双目标定：固定内参，只优化外参（R, T）
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            [self._detector.board.getChessboardCorners()],
            all_pts_1, all_pts_2,
            intrinsic_1.camera_matrix, intrinsic_1.dist_coeffs,  # 固定内参
            intrinsic_2.camera_matrix, intrinsic_2.dist_coeffs,
            intrinsic_1.image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC,  # 关键标志：不优化内参
        )

        # R: 3x3 旋转矩阵，T: 3x1 平移向量
        # E: 本质矩阵，F: 基础矩阵
        return StereoResult(camera_id_1, camera_id_2, R, T, E, F, ret)

    def get_projection_matrices(
        self, reference_camera_id: int, intrinsics: dict[int, IntrinsicResult]
    ) -> dict[int, np.ndarray]:
        # 以参考相机为原点，构建所有相机的 3x4 投影矩阵 P = K[R|T]
        # 参考相机 P = K * [I|0]，其他相机 P = K * [R|T]
        projs[reference_camera_id] = ref_int.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])

        for (cam1, cam2), pair in self._pair_transforms.items():
            if cam1 == reference_camera_id:
                projs[cam2] = other_int.camera_matrix @ np.hstack([pair.R, pair.T])
            elif cam2 == reference_camera_id:
                # 反向变换 R_inv = R^T, T_inv = -R^T * T
                R_inv = pair.R.T
                T_inv = -R_inv @ pair.T
                projs[cam1] = other_int.camera_matrix @ np.hstack([R_inv, T_inv])

        return projs
```

---

## 2. YOLO11-Pose 姿态估计

**文件**: `src/pose/pose2d_estimator.py`

```python
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# 标准 COCO 17 关键点定义

NUM_KEYPOINTS = len(COCO_KEYPOINTS)  # = 17


class Pose2DEstimator:
    # 使用 YOLO11-Pose 模型进行 2D 人体姿态估计

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        pose_cfg = cfg["pose2d"]
        self._confidence_threshold = pose_cfg["confidence_threshold"]  # 置信度阈值
        self._device = pose_cfg["device"]        # 推理设备 (cpu/cuda)
        self._model = YOLO(pose_cfg["model"])    # 加载 YOLO11-Pose 模型

    def estimate(self, image: np.ndarray) -> list[Pose2D]:
        # 单帧推理
        results = self._model(
            image,
            conf=self._confidence_threshold,  # 过滤低置信度检测
            device=self._device,
            verbose=False,
        )

        poses: list[Pose2D] = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            # keypoints.data.shape: (num_persons, 17, 3) — (x, y, confidence)
            kpts = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for i, (kpts_person, box) in enumerate(zip(kpts, boxes)):
                xy = kpts_person[:, :2]    # 17 个关键点的像素坐标
                conf = kpts_person[:, 2]   # 每个关键点的置信度
                bbox = box[:4]             # 人体检测框

                poses.append(Pose2D(keypoints=xy, confidence=conf, bbox=bbox, person_id=i))

        # 按平均置信度降序排列（最可靠的人优先）
        poses.sort(key=lambda p: -p.confidence.mean())
        return poses
```

---

## 3. 多视角三角测量优化

**文件**: `src/pose/pose3d_reconstructor.py`

```python
class Pose3DReconstructor:
    # 多视角 2D 关键点 → 3D 三角测量重建

    def triangulate(self, poses_2d: dict[int, list[Pose2D]]) -> list[Pose3D]:
        # 筛选有投影矩阵且有检测的相机
        camera_ids = sorted(set(poses_2d.keys()) & set(self._projection_matrices.keys()))
        if len(camera_ids) < self._min_views:  # 至少需要 min_views 个视角
            return []

        # 跨视角人物匹配（贪心匹配）
        person_groups = self._match_persons(poses_2d, camera_ids)

        for group in person_groups:
            # 对每个关键点逐个三角测量
            for j in range(NUM_KEYPOINTS):
                pts_2d = []
                proj_mats = []
                view_confs = []

                for cam_id in camera_ids:
                    pose = group.get(cam_id)
                    if pose is None:
                        continue
                    pt = pose.keypoints[j]
                    c = pose.confidence[j]
                    if c < 0.3:  # 过滤低置信度视角
                        continue
                    pts_2d.append(pt)
                    proj_mats.append(self._projection_matrices[cam_id])
                    view_confs.append(c)

                if len(pts_2d) < self._min_views:
                    continue

                # 执行三角测量
                pt_3d, err = self._triangulate_joint(pts_2d, proj_mats)

                # 检查重投影误差是否在阈值内
                if pt_3d is not None and err < self._reproj_threshold:
                    kpts_3d[j] = pt_3d
                    conf[j] = np.mean(view_confs)

    def _triangulate_joint(
        self, pts_2d: list[np.ndarray], proj_mats: list[np.ndarray]
    ) -> tuple[Optional[np.ndarray], float]:
        # 两视角：使用 OpenCV 内置的 cv2.triangulatePoints
        if len(pts_2d) == 2:
            point_4d = cv2.triangulatePoints(proj_mats[0], proj_mats[1], p1, p2)
            # 齐次坐标转非齐次坐标
            point_3d = (point_4d[:3, 0] / point_4d[3, 0]).astype(np.float64)
            err = self._compute_reprojection_error(point_3d, pts_2d, proj_mats)
            return point_3d, err
        else:
            # 多视角（>2）：使用 DLT（直接线性变换）
            return self._triangulate_dlt(pts_2d, proj_mats)

    def _triangulate_dlt(
        self, pts_2d: list[np.ndarray], proj_mats: list[np.ndarray]
    ) -> tuple[Optional[np.ndarray], float]:
        # DLT 算法：构造超定方程组 Ax = 0
        n = len(pts_2d)
        A = np.zeros((2 * n, 4), dtype=np.float64)

        for i, (pt, P) in enumerate(zip(pts_2d, proj_mats)):
            x, y = pt[0], pt[1]
            # 每个视角贡献两行：x*P3 - P1, y*P3 - P2
            A[2 * i] = x * P[2] - P[0]
            A[2 * i + 1] = y * P[2] - P[1]

        # SVD 分解求解最小特征向量
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]  # 最小奇异值对应的右奇异向量

        point_3d = X[:3] / X[3]  # 齐次→非齐次

        err = self._compute_reprojection_error(point_3d, pts_2d, proj_mats)
        return point_3d, err

    def _compute_reprojection_error(
        self, point_3d: np.ndarray, pts_2d: list[np.ndarray], proj_mats: list[np.ndarray]
    ) -> float:
        # 计算 3D 点在各视角下的重投影误差
        for pt_2d, P in zip(pts_2d, proj_mats):
            proj = P @ np.append(point_3d, 1.0)  # 齐次投影
            proj_2d = proj[:2] / proj[2]         # 归一化
            err = np.linalg.norm(proj_2d - pt_2d)  # 像素距离
        return total_err / count
```

---

## 4. UKF 传感器融合算法

### 4.1 IMU 预处理

**文件**: `src/fusion/imu_preprocessor.py`

```python
class IMUPreprocessor:
    # BNO055 IMU 数据预处理：四元数归一化、偏置校正、坐标变换

    def calibrate_bias(
        self, samples: dict[int, list[IMUData]], static_duration_s: float = 2.0
    ) -> None:
        # 静态偏置校正：在静止状态下采集数据求平均，减去重力得到偏置
        for sid, data_list in samples.items():
            accel_sum = np.zeros(3)
            for d in data_list:
                accel_sum += d.linear_accel
            # 偏置 = 平均加速度 - 重力向量
            self._bias_offsets[sid] = accel_sum / count - self._gravity  # gravity = [0, -9.81, 0]

    def process(self, data: IMUData) -> Optional[ProcessedIMU]:
        # 四元数归一化
        q = data.quaternion.copy()
        q /= np.linalg.norm(q)

        # 减去偏置
        if self._calibrated and data.sensor_id in self._bias_offsets:
            accel -= self._bias_offsets[data.sensor_id]

        # 四元数→旋转矩阵，将加速度从机体坐标系转到全局坐标系
        R = self._quaternion_to_rotation_matrix(q)
        global_accel = R @ accel

        return ProcessedIMU(..., global_accel=global_accel)

    @staticmethod
    def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        # 四元数 [w, x, y, z] → 3x3 旋转矩阵
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
```

### 4.2 UKF 融合核心

**文件**: `src/fusion/ukf_fusion.py`

```python
class VisualIMUFusion:
    # 无迹卡尔曼滤波器 (UKF)：融合视觉 3D 姿态和 IMU 数据
    # 每个关节独立维护一个 UKF（6 维状态: [x, y, z, vx, vy, vz]）

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        fusion_cfg = cfg["fusion"]
        self._process_noise_std = fusion_cfg["process_noise_std"]    # 过程噪声标准差
        self._measurement_noise_std = fusion_cfg["measurement_noise_std"]  # 观测噪声标准差
        self._imu_weight = fusion_cfg["imu_weight"]  # IMU 数据权重

        self._dim_x = 6   # 状态维度：位置(x,y,z) + 速度(vx,vy,vz)
        self._dim_z = 3   # 观测维度：位置(x,y,z)

        self._joint_filters: dict[int, UnscentedKalmanFilter] = {}
        self._dt = 1.0 / 30.0  # 默认帧率 30fps

    def _create_filters(self) -> None:
        # 为 17 个关节各创建一个 UKF
        for j in range(NUM_KEYPOINTS):
            points = MerweScaledSigmaPoints(
                n=self._dim_x,
                alpha=0.1,   # sigma 点扩散程度
                beta=2.0,    # 高斯分布最优值
                kappa=0.0,   # 次级缩放参数
            )

            ukf = UnscentedKalmanFilter(
                dim_x=self._dim_x,
                dim_z=self._dim_z,
                dt=self._dt,
                hx=self._measurement_function,   # 观测模型
                fx=self._state_transition,        # 状态转移模型
                points=points,
            )

            # Q: 过程噪声协方差矩阵
            ukf.Q = np.eye(self._dim_x) * self._process_noise_std ** 2
            ukf.Q[3:, 3:] *= 10.0  # 速度分量的噪声更大

            # R: 观测噪声协方差矩阵
            ukf.R = np.eye(self._dim_z) * self._measurement_noise_std ** 2

            self._joint_filters[j] = ukf

    @staticmethod
    def _state_transition(x: np.ndarray, dt: float) -> np.ndarray:
        # 状态转移方程（恒速模型）
        # x(k+1) = F * x(k)
        # [x, y, z, vx, vy, vz] → [x+vx*dt, y+vy*dt, z+vz*dt, vx, vy, vz]
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        return F @ x

    @staticmethod
    def _measurement_function(x: np.ndarray) -> np.ndarray:
        # 观测模型：直接观测位置，不观测速度
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])
        return H @ x

    def predict(self) -> None:
        # 预测步骤：所有关节的 UKF 同时执行预测
        for ukf in self._joint_filters.values():
            ukf.predict(fx_args=(self._dt,))

    def update_visual(self, pose_3d: Pose3D) -> None:
        # 视觉更新：用三角测量得到的 3D 关键点位置更新 UKF
        for j in range(len(self._joint_filters)):
            if pose_3d.confidence[j] > 0.3:
                ukf = self._joint_filters[j]
                # 自适应噪声：置信度越低，观测噪声越大
                noise_scale = 1.0 / max(pose_3d.confidence[j], 0.1)
                ukf.R = np.eye(self._dim_z) * self._measurement_noise_std ** 2 * noise_scale
                ukf.update(pose_3d.keypoints_3d[j])

    def update_imu(
        self, imu_data: dict[int, ProcessedIMU], joint_to_imu: dict[int, int]
    ) -> None:
        # IMU 更新：用加速度数据修正位置和速度
        for joint_idx, imu_idx in joint_to_imu.items():
            imu = imu_data.get(imu_idx)
            ukf = self._joint_filters[joint_idx]

            accel = imu.global_accel  # 全局坐标系下的加速度
            delta_v = accel * self._dt                    # 速度增量
            delta_pos = 0.5 * accel * self._dt ** 2       # 位置增量

            # 按 IMU 权重融合
            ukf.x[:3] += delta_pos * self._imu_weight
            ukf.x[3:] += delta_v * self._imu_weight

    def fuse_step(self, pose_3d, imu_data, joint_to_imu, timestamp_ns) -> FusedPose:
        # 完整融合步骤：预测 → 视觉更新 → IMU 更新
        self.predict()

        if pose_3d is not None:
            self.update_visual(pose_3d)

        self.update_imu(imu_data, joint_to_imu)

        return self.get_fused_pose(timestamp_ns)

    def get_fused_pose(self, timestamp_ns: int) -> FusedPose:
        # 从 UKF 状态中提取融合后的姿态
        for j, ukf in self._joint_filters.items():
            keypoints[j] = ukf.x[:3]      # 位置
            velocities[j] = ukf.x[3:]     # 速度

            # 置信度 = 1 - trace(P[:3,:3]) / 0.1，协方差越大置信度越低
            trace_P = np.trace(ukf.P[:3, :3])
            confidence[j] = max(0.0, 1.0 - trace_P / 0.1)
```

### 4.3 时域滤波（Savitzky-Golay）

**文件**: `src/fusion/temporal_filter.py`

```python
class TemporalFilter:
    # Savitzky-Golay 时域平滑滤波：消除抖动，保持运动特征

    def get_filtered(self) -> FusedPose | None:
        if len(self._buffer) < self._window_length:
            return self._buffer[-1]

        window = self._buffer[-self._window_length:]

        # 对每个关节的每个维度 (x, y, z) 分别滤波
        for j in range(n_joints):
            for dim in range(3):
                signal = positions[:, j, dim]
                # Savitzky-Golay 滤波：保留信号的高阶特征，同时平滑噪声
                filtered_positions[:, j, dim] = savgol_filter(
                    signal, self._window_length, self._polyorder
                )
```

---

## 5. 骨骼模型与 BVH 导出

### 5.1 骨骼模型

**文件**: `src/skeleton/skeleton_model.py`

```python
class SkeletonModel:
    # 层级骨骼模型：从 YAML 配置加载关节定义，支持正向运动学

    def __init__(self, config_path: str = "config/skeleton_model.yaml") -> None:
        # 从配置文件加载关节定义
        skel_cfg = cfg["skeleton"]
        for name, joint_cfg in skel_cfg.items():
            joint = Joint(
                name=name,
                parent=joint_cfg["parent"],      # 父关节名
                offset=np.array(joint_cfg["offset"]),  # 相对父关节的偏移
                channels=joint_cfg["channels"],  # BVH 通道定义
                coco_indices=joint_cfg.get("coco_mapping"),  # COCO 关键点映射
            )
            self._joints[name] = joint

    def forward_kinematics(
        self,
        joint_rotations: dict[str, np.ndarray],  # 每个关节的旋转矩阵
        root_position: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        # 正向运动学：从关节旋转计算所有关节的世界坐标位置
        self._fk_recursive(self._root, root_pos, np.eye(3), joint_rotations, positions)
        return positions

    def _fk_recursive(self, joint_name, parent_pos, parent_rot, rotations, positions):
        # 递归遍历骨骼层级树
        joint = self._joints[joint_name]
        local_rot = rotations.get(joint_name, np.eye(3))

        world_rot = parent_rot @ local_rot  # 世界旋转 = 父旋转 × 局部旋转
        world_pos = parent_pos + parent_rot @ joint.offset  # 世界位置 = 父位置 + 旋转后的偏移

        positions[joint_name] = world_pos

        for child_name in joint.children:
            self._fk_recursive(child_name, world_pos, world_rot, rotations, positions)
```

### 5.2 关节角度求解器

**文件**: `src/skeleton/joint_angle_solver.py`

```python
class JointAngleSolver:
    # 从 3D 关键点位置反解关节旋转角度（用于 BVH 导出）

    def solve(self, keypoints_3d: np.ndarray, joint_names: list[str]) -> dict[str, np.ndarray]:
        # 将 COCO 关键点名称映射到 3D 位置
        for i, name in enumerate(joint_names):
            positions[name] = keypoints_3d[i]

        for joint_name in self._skeleton.joint_names:
            joint = self._skeleton.joints[joint_name]
            if joint.parent is None:
                rotations[joint_name] = np.eye(3)  # 根关节无旋转
                continue

            # 计算骨骼方向向量
            bone_vector = child_pos - parent_pos
            bone_dir = bone_vector / np.linalg.norm(bone_vector)

            # 获取静止姿态下的方向（腿向下，手臂向左/右）
            rest_dir = self._get_rest_direction(joint_name)

            # 计算从静止方向到当前方向的旋转矩阵
            R = self._rotation_between_vectors(rest_dir, bone_dir)
            rotations[joint_name] = R

        # 将旋转矩阵转换为欧拉角（XYZ 顺序），用于 BVH 格式
        return self._euler_from_rotations(rotations, root_pos)

    @staticmethod
    def _get_rest_direction(joint_name: str) -> np.ndarray:
        # 不同部位的静止方向定义
        if "Leg" in joint_name or "Foot" in joint_name:
            return np.array([0.0, -1.0, 0.0])   # 腿向下
        if joint_name.startswith("Left"):
            return np.array([1.0, 0.0, 0.0])    # 左臂向右
        if joint_name.startswith("Right"):
            return np.array([-1.0, 0.0, 0.0])   # 右臂向左
        return np.array([0.0, 1.0, 0.0])        # 脊柱向上

    def _rotation_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        # 罗德里格斯旋转公式：计算从 v1 到 v2 的旋转矩阵
        axis = np.cross(v1, v2)            # 旋转轴 = 叉积
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))  # 旋转角 = 点积的反余弦
        return self._rotation_matrix(axis, angle)

    @staticmethod
    def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        # 罗德里格斯公式：轴角 → 旋转矩阵
        c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
        x, y, z = axis
        return np.array([
            [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c  ],
        ])
```

### 5.3 BVH 导出器

**文件**: `src/skeleton/bvh_exporter.py`

```python
class BVHExporter:
    # 将关节旋转数据导出为标准 BVH (Biovision Hierarchy) 文件

    def __init__(self, skeleton: SkeletonModel, frame_time: float = 1.0 / 30.0) -> None:
        self._skeleton = skeleton
        self._frame_time = frame_time  # 每帧时间（秒）
        self._motion_data: list[dict[str, np.ndarray]] = []  # 运动帧数据

    def add_frame(self, rotations: dict[str, np.ndarray]) -> None:
        self._motion_data.append(rotations)

    def export_raw(self, output_path: str | Path) -> None:
        # 写入标准 BVH 格式文件
        with open(output_path, "w") as f:
            # 第一部分：HIERARCHY（层级定义）
            f.write("HIERARCHY\n")
            self._write_hierarchy(f, self._skeleton.root, 0)

            # 第二部分：MOTION（运动数据）
            f.write("MOTION\n")
            f.write(f"Frames: {len(self._motion_data)}\n")
            f.write(f"Frame Time: {self._frame_time:.6f}\n")

            for frame_data in self._motion_data:
                # 按关节顺序写出每帧的角度值
                for name in joint_order:
                    if name in frame_data:
                        vals = frame_data[name]
                        line_values.extend(f"{v:.6f}" for v in vals)
                    else:
                        # 无数据的关节填 0
                        line_values.extend(["0.000000"] * n)
                f.write(" ".join(line_values) + "\n")

    def _write_hierarchy(self, f, joint_name: str, depth: int) -> None:
        # 递归写入层级结构
        indent = "\t" * depth
        if depth == 0:
            f.write("ROOT " + joint_name + "\n")    # 根关节
        else:
            f.write(indent + "JOINT " + joint_name + "\n")  # 普通关节

        f.write(indent + "{\n")
        f.write(indent + f"\tOFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
        f.write(indent + f"\tCHANNELS {len(joint.channels)} {' '.join(joint.channels)}\n")

        for child_name in joint.children:
            self._write_hierarchy(f, child_name, depth + 1)

        if not joint.children:
            # 叶子关节添加 End Site
            f.write(indent + "\tEnd Site\n")
            f.write(indent + "\t{\n")
            f.write(indent + "\t\tOFFSET 0.000000 0.000000 0.000000\n")
            f.write(indent + "\t}\n")

        f.write(indent + "}\n")
```

---

## 6. 实时可视化与 GUI

### 6.1 Open3D 实时 3D 查看器

**文件**: `src/visualization/live_viewer.py`

```python
class LiveViewer:
    # 使用 Open3D 实时渲染 3D 骨骼骨架

    SKELETON_EDGES = [
        ("Hips", "Spine"), ("Spine", "Chest"), ("Chest", "Neck"), ("Neck", "Head"),
        ("Chest", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"),
        ("LeftForeArm", "LeftHand"),
        # ... 完整的 20 条骨骼连接线
    ]

    def initialize(self) -> None:
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(self._window_name, width=1280, height=720)

        # 添加坐标系参考
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self._vis.add_geometry(coord)

        # 为每个关节创建小球体
        for name, _ in SKELETON_EDGES:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            color = JOINT_COLORS.get(name, [1, 1, 1])  # 每个关节不同颜色
            sphere.paint_uniform_color(color)
            self._spheres[name] = sphere

        # 为每条骨骼创建 LineSet
        for i, (p, c) in enumerate(SKELETON_EDGES):
            line = o3d.geometry.LineSet()
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            self._geometry[f"{p}_{c}"] = line

    def update(self, joint_positions: dict[str, np.ndarray]) -> None:
        # 每帧更新：移动球体位置和骨骼线段端点
        for name, sphere in self._spheres.items():
            if name in joint_positions:
                pos = joint_positions[name]
                sphere.translate(pos - sphere.get_center(), relative=False)
                self._vis.update_geometry(sphere)

        for parent, child in SKELETON_EDGES:
            key = f"{parent}_{child}"
            line = self._geometry[key]
            if parent in joint_positions and child in joint_positions:
                points = o3d.utility.Vector3dVector([p_pos, c_pos])
                line.points = points
                self._vis.update_geometry(line)

        self._vis.poll_events()
        self._vis.update_renderer()
```

### 6.2 Tkinter GUI 主界面

**文件**: `src/gui/app.py`

```python
class MovCapApp:
    # 主 GUI 应用：集成标定、录制、处理、演示等功能

    def _build_ui(self) -> None:
        # 布局：左侧控制面板 + 右侧可视化面板
        main_paned = tk.PanedWindow(self._root, orient=tk.HORIZONTAL)
        left_panel  # 环境检查 + 控制按钮
        right_panel # 骨骼预览 + 状态信息 + 日志

    def _build_control_panel(self, parent) -> None:
        # 四个主要操作按钮
        ttk.Button("Calibrate Cameras", self._on_calibrate)   # 相机标定
        ttk.Button("Record Session", self._on_record)         # 录制会话
        ttk.Button("Process Data", self._on_process)          # 处理数据
        ttk.Button("Run Tests", self._on_run_tests)           # 运行测试

        # 演示模式控件
        tk.Radiobutton("walk", "wave", "squat")  # 选择动作类型
        ttk.Button("Play Demo", self._on_play_demo)           # 播放演示
        ttk.Button("Export Demo to BVH", self._on_export_demo_bvh)  # 导出 BVH
        tk.Scale(from_=0.1, to=3.0, variable=self._speed_var) # 播放速度

    def _on_calibrate(self) -> None:
        # 调用外部脚本执行标定
        result = subprocess.run(
            [sys.executable, "-m", "scripts.calibrate",
             "--config", self._config_path,
             "--output", "config/calibration/",
             "--duration", str(duration)],
            capture_output=True, text=True, cwd=str(self._project_dir)
        )

    def _on_play_demo(self) -> None:
        motion = self._demo_var.get()
        speed = self._speed_var.get()
        gen = DemoDataGenerator(fps=30)
        self._demo_frames = gen.get_demo_sequence(motion, 120)
        self._play_next_frame()

    def _play_next_frame(self) -> None:
        # 定时器驱动的动画播放
        frame = self._demo_frames[self._demo_index]
        self._skeleton_canvas.update(frame, self._demo_index)  # 更新 2D 骨骼画布
        self._demo_index += 1
        delay = max(10, int(1000 / (30 * speed)))  # 根据速度调整帧间隔
        self._root.after(delay, self._play_next_frame)  # 调度下一帧
```

### 6.3 2D 骨骼画布

**文件**: `src/gui/skeleton_canvas.py`

```python
class SkeletonCanvas:
    # COCO 17 关键点的 2D 可视化画布

    SKELETON_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),   # 头部
        (0, 5), (0, 6),                      # 颈→肩
        (5, 7), (7, 9),                      # 左臂
        (6, 8), (8, 10),                     # 右臂
        (5, 11), (6, 12),                    # 肩→髋
        (11, 12),                            # 髋连线
        (11, 13), (13, 15),                  # 左腿
        (12, 14), (14, 16),                  # 右腿
    ]

    def update(self, keypoints_3d: np.ndarray, frame_num: int = 0) -> None:
        # 将 3D 关键点投影到 2D 画布
        xy = kpts[:, :2]  # 取 x, y 分量

        # 自适应缩放：计算范围和偏移，使骨架居中
        x_range = np.ptp(xy[:, 0])
        y_range = np.ptp(xy[:, 1])
        scale = min(
            (self._width - 2 * margin) / x_range,
            (self._height - 2 * margin) / y_range
        ) * 0.8

        # 更新骨骼线段和关节圆圈的坐标
        for pair, item in self._bone_items.items():
            i, j = pair
            self._canvas.coords(item, sx[i], sy[i], sx[j], sy[j])
```

---

## 7. 演示模式（无需硬件即可运行）

**文件**: `src/demo_data.py`

```python
class DemoDataGenerator:
    # 生成合成运动数据，无需摄像头和 IMU 即可测试完整流水线

    # T 姿态基准：17 个 COCO 关键点的 3D 坐标
    _TPOSE_OFFSETS = np.array([
        [ 0.00,  1.60,  0.10],   # 0  nose
        [-0.18,  1.45,  0.00],   # 5  left_shoulder
        [-0.46,  1.45,  0.00],   # 7  left_elbow
        [-0.72,  1.45,  0.00],   # 9  left_wrist
        [-0.10,  0.95,  0.00],   # 11 left_hip
        [-0.10,  0.50,  0.00],   # 13 left_knee
        [-0.10,  0.08,  0.00],   # 15 left_ankle
        # ... 完整 17 点
    ], dtype=np.float64)

    def _walk_pose(self, t: float) -> np.ndarray:
        # 行走动画：基于时间的参数化运动模型
        kpts = _TPOSE_OFFSETS.copy()
        phase = t * 2.0 * np.pi

        # 髋部摆动 + 垂直起伏
        hip_sway = 0.02 * np.sin(phase)
        vertical_bob = 0.02 * abs(np.sin(phase))

        # 步行腿步态（正弦相位差）
        step = 0.25 * np.sin(phase)
        kpts[11, 2] += step * 0.5    # left_hip 前后
        kpts[12, 2] -= step * 0.5    # right_hip 反相

        # 膝盖弯曲
        l_knee_bend = max(0, -np.sin(phase)) * 0.15
        r_knee_bend = max(0, np.sin(phase)) * 0.15

        # 手臂摆动（与腿反相）
        arm_swing = 0.15 * np.sin(phase + np.pi)
        kpts[7, 2] += arm_swing
        kpts[8, 2] -= arm_swing

        return kpts

    def generate_to_bvh(self, motion="walk", num_frames=120, output_path="recordings/demo.bvh") -> Path:
        # 演示数据 → BVH 导出完整流程
        frames = self.get_demo_sequence(motion, num_frames)
        exporter = BVHExporter(self._skeleton, frame_time=self._dt)
        for kpts in frames:
            angles = self._angle_solver.solve(kpts, COCO_KEYPOINTS)  # 位置→角度
            exporter.add_frame(angles)
        exporter.export_raw(output_path)
```

---

## 8. 主流水线 Pipeline

**文件**: `src/pipeline.py`

```python
class MoCapPipeline:
    # 完整的动捕流水线：采集 → 同步 → 2D 检测 → 3D 重建 → 融合 → 滤波 → BVH

    def process_frame(self) -> Optional[FusedPose]:
        # 第一步：从所有相机读取帧
        frames = self._camera_manager.read(timeout_ms=2000)
        imu_data = self._imu_manager.read_all(timeout_ms=100)

        # 第二步：时间同步（最近邻匹配）
        for cam_id, frame in frames.items():
            self._synchronizer.add_camera_frame(frame)
        sample = self._synchronizer.get_synced_sample()

        # 第三步：每台相机独立进行 2D 姿态估计
        for cam_id, frame in sample.frames.items():
            poses = self._pose2d.estimate(frame.image)
            poses_2d[cam_id] = poses

        # 第四步：多视角三角测量 → 3D 姿态
        if len(poses_2d) >= 2:
            pose_3d_list = self._pose3d.triangulate(poses_2d)
            pose_3d = pose_3d_list[0]

        # 第五步：IMU 数据预处理
        for sid, data in sample.imu_data.items():
            processed_imu[sid] = self._imu_preprocessor.process(data)

        # 第六步：UKF 视觉-IMU 融合
        fused = self._fusion.fuse_step(
            pose_3d=pose_3d,
            imu_data=processed_imu,
            joint_to_imu=DEFAULT_JOINT_TO_IMU,
            timestamp_ns=sample.timestamp_ns,
        )

        # 第七步：时域滤波平滑
        self._temporal_filter.add(fused)
        filtered = self._temporal_filter.get_filtered()

        return fused

    def process_to_bvh(self, num_frames=0, output_path=None) -> None:
        # 完整录制流程：循环采集 → 融合 → 角度求解 → BVH 导出
        self.start()
        while num_frames <= 0 or count < num_frames:
            fused = self.process_frame()
            if fused is None:
                continue
            angles = self._angle_solver.solve(fused.keypoints_3d, COCO_KEYPOINTS)
            self._bvh_exporter.add_frame(angles)
            count += 1

        if output_path is not None:
            self._bvh_exporter.export_raw(output_path)
```

---

## 数据流总结

```
3×USB Camera → Frame Capture → ChArUco Calibration (intrinsic + extrinsic)
                                      ↓
                              YOLO11-Pose 2D Detection (per camera)
                                      ↓
                        Multi-View Triangulation (DLT / cv2.triangulatePoints)
                                      ↓
8×BNO055 IMU → Preprocessing (bias correction, quaternion normalization)
                      ↓
            UKF Fusion (per-joint 6D state: [x,y,z,vx,vy,vz])
                      ↓
            Savitzky-Golay Temporal Filter
                      ↓
            Joint Angle Solver (position → rotation matrix → Euler angles)
                      ↓
            BVH Export (HIERARCHY + MOTION)
```
