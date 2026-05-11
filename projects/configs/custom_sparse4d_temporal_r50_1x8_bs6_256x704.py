_base_ = ["./sparse4dv3_temporal_r50_1x8_bs6_256x704.py"]

# 这些值沿用原始配置，单独写出来是为了避免在继承配置里依赖
# `_base_` 变量本身参与 Python 表达式。
total_batch_size = 48
num_gpus = 8
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
checkpoint_epoch_interval = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
tracking_test = True
tracking_threshold = 0.2

# ================== custom dataset basics ========================
# 下面这些字段是你迁移到自定义数据时最先要改的部分。
dataset_type = "NuScenes3DDetTrackDataset"
data_root = "data/custom/"
anno_root = "data/custom_infos/"

# 如果你的类别和 nuScenes 不同，就在这里改。
# 同时也要确保 custom_converter.py 产出的 gt_names 与这里一致。
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
num_classes = len(class_names)

# 如果不是 6 相机，这里要同步改。
num_cams = 6

# 如果你重新聚类了 anchor，改成自己的 anchor 文件路径。
custom_anchor = "data/custom_infos/custom_kmeans900.npy"

# 把这里改成你原始图像分辨率。
custom_image_h = 900
custom_image_w = 1600

# input_shape 是网络最终输入分辨率，通常可以先沿用原配置。
input_shape = (704, 256)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": custom_image_h,
    "W": custom_image_w,
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
}

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend="disk"),
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_depth",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=["img", "timestamp", "projection_mat", "image_wh"],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    version="custom",
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "custom_infos_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "custom_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_test,
        tracking_threshold=tracking_threshold,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "custom_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=tracking_test,
        tracking_threshold=tracking_threshold,
    ),
)

model = dict(
    head=dict(
        instance_bank=dict(anchor=custom_anchor),
        deformable_model=dict(num_cams=num_cams),
        refine_layer=dict(num_cls=num_classes),
    )
)

# 如果你没有时序标注或想先跑通单帧版本，也可以先把 tracking_test 关掉。
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval,
    pipeline=[
        dict(type="LoadMultiViewImageFromFiles", to_float32=True),
        dict(type="Collect", keys=["img"], meta_keys=["timestamp", "lidar2img"]),
    ],
)
