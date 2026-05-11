# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import numpy as np
from nuscenes.nuscenes import NuScenes
import mmcv
from pyquaternion import Quaternion

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def create_nuscenes_infos(
    root_path, info_prefix, version="v1.0-trainval", max_sweeps=10
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    # 延迟导入，避免其他脚本 import 本文件时强制依赖 nuScenes SDK。
    from nuscenes.nuscenes import NuScenes

    # nusc: nuScenes 官方数据集对象，用来按 token 访问场景、样本和标注。
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits

    # available_vers: 本脚本支持的数据集版本集合。
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        # train_scenes / val_scenes: 官方划分给出的场景名称列表。
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.
    # 过滤本地磁盘上真实存在的场景，避免索引里有但文件未下载完整。
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes)
    )
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    # 后续判断样本属于 train/val 时使用 scene token，因此这里把名字映射为 token 集合。
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in val_scenes
        ]
    )

    # test: 测试集没有真值标注，因此后续需要跳过 annotation 收集。
    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print(
            "train scene: {}, val scene: {}".format(
                len(train_scenes), len(val_scenes)
            )
        )
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps
    )

    # metadata: 写入 pkl 的版本信息，供后续加载时校验。
    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = "{}_infos_test.pkl".format(info_prefix)
        mmcv.dump(data, info_path)
    else:
        print(
            "train sample: {}, val sample: {}".format(
                len(train_nusc_infos), len(val_nusc_infos)
            )
        )
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = "{}_infos_train.pkl".format(info_prefix)
        mmcv.dump(data, info_path)
        data["infos"] = val_nusc_infos
        info_val_path = "{}_infos_val.pkl".format(info_prefix)
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        # scene_token: 当前场景在 nuScenes 中的唯一标识。
        scene_token = scene["token"]
        # scene_rec: 场景完整记录；sample_rec: 该场景第一帧 sample。
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        # sd_rec: LIDAR_TOP 对应的 sample_data 记录。
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            # lidar_path: 当前激光点云文件路径；boxes: 该帧 3D 框；第三项是相机内参/额外信息。
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(
    nusc, train_scenes, val_scenes, test=False, max_sweeps=10
):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    # train_nusc_infos / val_nusc_infos: 最终分别写入 train 和 val pkl 的样本信息列表。
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        # lidar_token: 当前关键帧顶置激光雷达 sample_data 的 token。
        lidar_token = sample["data"]["LIDAR_TOP"]
        # sd_rec: 顶置激光雷达的 sample_data 记录。
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        # cs_record: 传感器到 ego 车体坐标系的标定参数。
        cs_record = nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        # pose_record: ego 到 global 世界坐标系的位姿。
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        # lidar_path: 点云文件路径；boxes: nuScenes SDK 返回的该帧标注框对象列表。
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        # info: 当前样本最终写入 pkl 的核心信息字典。
        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
        }

        # l2e_*: lidar -> ego；e2g_*: ego -> global。
        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        # 将四元数旋转转换为 3x3 旋转矩阵，便于后续坐标变换。
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        # camera_types: nuScenes 六个环视相机名称。
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            # cam_token: 当前相机 sample_data token。
            cam_token = sample["data"][cam]
            # cam_path: 图像路径；cam_intrinsic: 3x3 相机内参矩阵。
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            # cam_info: 当前相机到顶置激光雷达坐标系的外参以及基础路径信息。
            cam_info = obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            # 将相机内参一并写入该相机信息。
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        # sweeps: 当前关键帧之前若干历史点云帧的信息。
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                # 对每一个历史点云，同样计算到当前顶置雷达坐标系的变换。
                sweep = obtain_sensor2top(
                    nusc,
                    sd_rec["prev"],
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    "lidar",
                )
                sweeps.append(sweep)
                # 沿时间链继续向前追溯。
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps
        # obtain annotation
        if not test:
            # annotations: 原始 annotation record 列表。
            annotations = [
                nusc.get("sample_annotation", token)
                for token in sample["anns"]
            ]
            # locs: 每个 3D 框中心点坐标 [N, 3]。
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            # dims: 每个 3D 框尺寸，nuScenes 原生顺序是 [w, l, h]。
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            # rots: 每个 3D 框的 yaw 角，单位为弧度。
            rots = np.array(
                [b.orientation.yaw_pitch_roll[0] for b in boxes]
            ).reshape(-1, 1)
            # velocity: 从 SDK 查询目标在 global 坐标系下的速度，只保留 xy 两个分量。
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )
            # valid_flag: 至少被 1 个 lidar/radar 点击中的目标标记为有效。
            valid_flag = np.array(
                [
                    (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                    for anno in annotations
                ],
                dtype=bool,
            ).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                # 先补成 3 维速度向量，z 速度默认设为 0。
                velo = np.array([*velocity[i], 0.0])
                # 速度和位置一样，需要从 global 旋转回当前 lidar 坐标系。
                velo = (
                    velo
                    @ np.linalg.inv(e2g_r_mat).T
                    @ np.linalg.inv(l2e_r_mat).T
                )
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    # 将 nuScenes 原始细粒度类别映射到检测任务使用的标准类别名。
                    names[i] = NameMapping[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            # 将尺寸顺序从 [w, l, h] 调整为当前项目使用的 [l, w, h]。
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            # instance_inds: 实例级别索引，同一物体跨时间帧可共享同一个 instance id。
            info["instance_inds"] = np.array(
                [
                    nusc.getind("instance", x["instance_token"])
                    for x in annotations
                ]
            )
            # 依次写入检测训练所需的所有 GT 字段。
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations]
            )
            info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations]
            )
            info["valid_flag"] = valid_flag

        # 根据场景 token 决定该样本归属 train 还是 val。
        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    # 先取出当前传感器帧的 sample_data、标定和 ego pose。
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get(
        "calibrated_sensor", sd_rec["calibrated_sensor_token"]
    )
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    # data_path: 当前传感器文件在磁盘上的路径。
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    # sweep: 记录该历史帧/相机帧的基础元信息和坐标变换参数。
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    # l2e_r_s / l2e_t_s: 当前传感器 -> 当前 ego；e2g_r_s / e2g_t_s: 当前 ego -> global。
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    # 将当前传感器自身的旋转四元数转换为矩阵。
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    # R: 当前传感器坐标系到参考 Top LiDAR 坐标系的旋转矩阵。
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    # T: 当前传感器坐标系到参考 Top LiDAR 坐标系的平移向量。
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    # 存储为 points @ R.T + T 的形式，便于后续直接对点云/像素射线使用。
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


if __name__ == "__main__":
    import argparse

    # 支持一次处理多个版本，如默认同时生成 trainval 和 test 两套 info。
    parser = argparse.ArgumentParser(description="nuscenes converter")
    parser.add_argument("--root_path", type=str, default="./data/nuscenes")
    parser.add_argument("--info_prefix", type=str, default="nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-trainval,v1.0-test")
    parser.add_argument("--max_sweeps", type=int, default=10)
    args = parser.parse_args()

    # 用逗号分隔多个版本，逐个生成对应的 pkl 信息文件。
    versions = args.version.split(",")
    for version in versions:
        create_nuscenes_infos(
            args.root_path,
            args.info_prefix,
            version=version,
            max_sweeps=args.max_sweeps,
        )
