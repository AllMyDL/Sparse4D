import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import mmcv
import numpy as np


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> List[float]:
    """将 3x3 旋转矩阵转换为 [w, x, y, z] 四元数。

    Sparse4D 当前数据集实现中默认使用四元数来表达
    `lidar2ego_rotation` / `ego2global_rotation`。
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(rotation)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rotation[2, 1] - rotation[1, 2]) * s
        qy = (rotation[0, 2] - rotation[2, 0]) * s
        qz = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = 2.0 * np.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            )
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = 2.0 * np.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            )
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            )
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    return quat.tolist()


def ensure_path(root_path: str, path: str) -> str:
    """把相对路径转成相对 root_path 的规范路径。"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root_path, path))


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """由旋转矩阵和平移向量拼 4x4 齐次变换矩阵。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """求 4x4 刚体变换逆矩阵。"""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv_transform = np.eye(4, dtype=np.float64)
    inv_transform[:3, :3] = rotation.T
    inv_transform[:3, 3] = -rotation.T @ translation
    return inv_transform


def convert_boxes_to_lidar(
    boxes: List[Dict[str, Any]], global_to_lidar: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把自定义标注框转换成 Sparse4D 训练所需字段。

    你需要根据自己的标注格式调整这里的字段读取逻辑。

    期望每个 box 至少包含：
    - name: 类别名
    - size: [l, w, h]
    - yaw: 目标朝向，弧度制
    - center_global 或 center_lidar
    可选：
    - velocity_global 或 velocity_lidar
    - num_lidar_pts
    - num_radar_pts
    - instance_id
    """
    gt_boxes = []
    gt_names = []
    gt_velocity = []
    num_lidar_pts = []
    num_radar_pts = []
    instance_inds = []

    lidar_rotation = global_to_lidar[:3, :3]
    for box in boxes:
        if "center_lidar" in box:
            center_lidar = np.asarray(box["center_lidar"], dtype=np.float32)
        else:
            center_global = np.asarray(box["center_global"], dtype=np.float64)
            center_lidar = (
                global_to_lidar[:3, :3] @ center_global
                + global_to_lidar[:3, 3]
            ).astype(np.float32)

        size = np.asarray(box["size"], dtype=np.float32)
        yaw = float(box["yaw"])

        if "velocity_lidar" in box:
            velocity = np.asarray(box["velocity_lidar"][:2], dtype=np.float32)
        elif "velocity_global" in box:
            velocity_global = np.asarray(
                [box["velocity_global"][0], box["velocity_global"][1], 0.0],
                dtype=np.float64,
            )
            velocity_lidar = lidar_rotation @ velocity_global
            velocity = velocity_lidar[:2].astype(np.float32)
        else:
            velocity = np.array([0.0, 0.0], dtype=np.float32)

        gt_boxes.append(
            np.array(
                [
                    center_lidar[0],
                    center_lidar[1],
                    center_lidar[2],
                    size[0],
                    size[1],
                    size[2],
                    yaw,
                ],
                dtype=np.float32,
            )
        )
        gt_names.append(box["name"])
        gt_velocity.append(velocity)
        num_lidar_pts.append(int(box.get("num_lidar_pts", 1)))
        num_radar_pts.append(int(box.get("num_radar_pts", 0)))
        instance_inds.append(int(box.get("instance_id", -1)))

    if len(gt_boxes) == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.array([], dtype=object),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(gt_boxes).astype(np.float32),
        np.array(gt_names),
        np.stack(gt_velocity).astype(np.float32),
        np.array(num_lidar_pts, dtype=np.int64),
        np.array(num_radar_pts, dtype=np.int64),
        np.array(instance_inds, dtype=np.int64),
    )


def build_camera_info(
    camera: Dict[str, Any], root_path: str
) -> Dict[str, Any]:
    """构建单个相机的 `cams[cam_name]` 字段。

    约定输入 camera 至少包含：
    - name: 相机名，如 CAM_FRONT
    - image_path: 图像路径
    - cam_intrinsic: 3x3 相机内参
    - cam_to_lidar_rotation 或 lidar_to_cam_rotation
    - cam_to_lidar_translation 或 lidar_to_cam_translation
    """
    data_path = ensure_path(root_path, camera["image_path"])
    intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float32)

    if "cam_to_lidar_rotation" in camera and "cam_to_lidar_translation" in camera:
        sensor2lidar_rotation = np.asarray(
            camera["cam_to_lidar_rotation"], dtype=np.float32
        )
        sensor2lidar_translation = np.asarray(
            camera["cam_to_lidar_translation"], dtype=np.float32
        )
    elif "lidar_to_cam_rotation" in camera and "lidar_to_cam_translation" in camera:
        lidar_to_cam = make_transform(
            camera["lidar_to_cam_rotation"], camera["lidar_to_cam_translation"]
        )
        cam_to_lidar = invert_transform(lidar_to_cam)
        sensor2lidar_rotation = cam_to_lidar[:3, :3].astype(np.float32)
        sensor2lidar_translation = cam_to_lidar[:3, 3].astype(np.float32)
    else:
        raise ValueError(
            "camera 必须提供 cam_to_lidar_* 或 lidar_to_cam_* 标定参数"
        )

    return dict(
        data_path=data_path,
        type=camera["name"],
        sample_data_token=camera.get("token", camera["name"]),
        sensor2ego_translation=[0.0, 0.0, 0.0],
        sensor2ego_rotation=[1.0, 0.0, 0.0, 0.0],
        ego2global_translation=[0.0, 0.0, 0.0],
        ego2global_rotation=[1.0, 0.0, 0.0, 0.0],
        timestamp=int(camera.get("timestamp", 0)),
        sensor2lidar_rotation=sensor2lidar_rotation,
        sensor2lidar_translation=sensor2lidar_translation,
        cam_intrinsic=intrinsic,
    )


def build_one_info(
    sample: Dict[str, Any], root_path: str, with_gt: bool = True
) -> Dict[str, Any]:
    """把一个原始样本转换成 Sparse4D 所需的单帧 info。

    这个函数是整个模板里最重要的入口。
    你后续最可能修改的就是这里和 `load_raw_samples()`。
    """
    sample_token = sample["token"]
    lidar_path = ensure_path(root_path, sample["lidar_path"])
    timestamp = int(sample["timestamp"])

    lidar_to_ego = make_transform(
        sample["lidar2ego_rotation_matrix"],
        sample["lidar2ego_translation"],
    )
    ego_to_global = make_transform(
        sample["ego2global_rotation_matrix"],
        sample["ego2global_translation"],
    )
    lidar_to_global = ego_to_global @ lidar_to_ego
    global_to_lidar = invert_transform(lidar_to_global)

    info = dict(
        token=sample_token,
        lidar_path=lidar_path,
        sweeps=[],
        cams={},
        lidar2ego_translation=np.asarray(
            sample["lidar2ego_translation"], dtype=np.float32
        ).tolist(),
        lidar2ego_rotation=rotation_matrix_to_quaternion(
            sample["lidar2ego_rotation_matrix"]
        ),
        ego2global_translation=np.asarray(
            sample["ego2global_translation"], dtype=np.float32
        ).tolist(),
        ego2global_rotation=rotation_matrix_to_quaternion(
            sample["ego2global_rotation_matrix"]
        ),
        timestamp=timestamp,
    )

    for camera in sample["cameras"]:
        cam_info = build_camera_info(camera, root_path)
        info["cams"][camera["name"]] = cam_info

    # 如果你有历史点云帧，可以在这里把它们补成 nuScenes 风格的 sweeps 列表。
    # 当前模板先留空，便于先把单帧版本跑通。
    if sample.get("sweeps") is not None:
        info["sweeps"] = sample["sweeps"]

    if with_gt:
        (
            gt_boxes,
            gt_names,
            gt_velocity,
            num_lidar_pts,
            num_radar_pts,
            instance_inds,
        ) = convert_boxes_to_lidar(sample.get("anns", []), global_to_lidar)
        info["gt_boxes"] = gt_boxes
        info["gt_names"] = gt_names
        info["gt_velocity"] = gt_velocity
        info["num_lidar_pts"] = num_lidar_pts
        info["num_radar_pts"] = num_radar_pts
        info["valid_flag"] = (num_lidar_pts + num_radar_pts) > 0
        info["instance_inds"] = instance_inds

    return info


def load_raw_samples(annotation_path: str) -> List[Dict[str, Any]]:
    """读取你自己的原始标注。

    这是模板里第二个需要你重点替换的函数。

    当前默认假设你已经准备了一个 json / pkl 文件，
    它按“每个元素对应一个时间帧”的方式组织样本，形如：

    [
        {
            "token": "000001",
            "timestamp": 1710000000000000,
            "lidar_path": "lidar/000001.bin",
            "lidar2ego_rotation_matrix": [[...], [...], [...]],
            "lidar2ego_translation": [x, y, z],
            "ego2global_rotation_matrix": [[...], [...], [...]],
            "ego2global_translation": [x, y, z],
            "cameras": [
                {
                    "name": "CAM_FRONT",
                    "image_path": "images/front/000001.jpg",
                    "cam_intrinsic": [[...], [...], [...]],
                    "cam_to_lidar_rotation": [[...], [...], [...]],
                    "cam_to_lidar_translation": [x, y, z]
                }
            ],
            "anns": [
                {
                    "name": "car",
                    "center_global": [x, y, z],
                    "size": [l, w, h],
                    "yaw": 0.1,
                    "velocity_global": [vx, vy],
                    "num_lidar_pts": 18,
                    "instance_id": 7
                }
            ]
        }
    ]

    如果你的数据现在不是这个结构，建议先写一个简单的预处理脚本，
    把原始标注先整理成这个中间格式，再复用当前模板。
    """
    if annotation_path.endswith(".pkl"):
        samples = mmcv.load(annotation_path)
    else:
        samples = mmcv.load(annotation_path, file_format="json")

    if not isinstance(samples, list):
        raise TypeError("原始标注文件应当解析为 list，每个元素代表一个样本")
    return samples


def dump_infos(
    infos: List[Dict[str, Any]], out_path: str, version: str = "custom"
) -> None:
    """写出 Sparse4D 可读的 infos.pkl。"""
    infos = sorted(infos, key=lambda x: x["timestamp"])
    mmcv.dump(dict(infos=infos, metadata=dict(version=version)), out_path)
    print(f"Saved {len(infos)} samples to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert custom multi-camera + lidar data to Sparse4D infos"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        required=True,
        help="自定义数据根目录",
    )
    parser.add_argument(
        "--ann-path",
        type=str,
        required=True,
        help="你自己的原始标注文件路径，默认支持 json/pkl",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="输出的 Sparse4D info pkl 路径",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="custom",
        help="写入 metadata 的版本名",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="测试集模式，不写 gt_* 字段",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_samples = load_raw_samples(args.ann_path)

    infos = []
    for sample in raw_samples:
        info = build_one_info(
            sample=sample,
            root_path=args.root_path,
            with_gt=not args.test_mode,
        )
        infos.append(info)

    dump_infos(infos, args.out_path, version=args.version)


if __name__ == "__main__":
    main()
