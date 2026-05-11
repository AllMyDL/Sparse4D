"""
ops 模块：自定义的可变形聚合操作
=====================================
该模块实现了 Sparse4D 中的核心算子 - 可变形特征聚合（Deformable Aggregation）
这是一个高性能的 CUDA 操作，用于多摄像头、多尺度特征的自适应采样和聚合
"""
import torch

from .deformable_aggregation import DeformableAggregationFunction


def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    """
    可变形特征聚合函数的包装器

    将多个相机的多尺度特征进行自适应采样和聚合，实现动态的特征关联

    参数:
        feature_maps (Tensor): 格式化后的多相机多尺度特征 [bs, num_feat, num_embeds]
        spatial_shape (Tensor): 各相机各尺度的空间尺寸 [num_cams, num_scales, 2]
        scale_start_index (Tensor): 各尺度特征在压平张量中的起始索引 [num_cams, num_scales]
        sampling_location (Tensor): 采样点的坐标位置 [bs, num_anchors, num_pts, num_cams, 2]
        weights (Tensor): 各采样点的权重系数 [bs, num_anchors, num_pts, num_cams, num_scales, num_groups]

    返回:
        output (Tensor): 聚合后的特征张量 [bs, num_anchors, num_embeds]
    """
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def feature_maps_format(feature_maps, inverse=False):
    """
    特征图格式转换函数

    功能：
    1. 正向：将多个特征图列表转换为压平的、便于 CUDA 处理的格式
    2. 反向：将压平的特征图还原为原始的多相机多尺度格式

    参数:
        feature_maps:
            - 正向模式: 特征图列表，每个元素形状 [bs, num_cams, num_levels, height, width, channels]
            - 反向模式: 元组 (col_feats, spatial_shape, scale_start_index)
        inverse (bool): 是否执行反向转换，默认为 False（正向转换）

    返回:
        - 正向: [col_feats, spatial_shape, scale_start_index]
            - col_feats: 压平后的特征 [bs, total_feat_points, channels]
            - spatial_shape: 各相机各尺度的空间尺寸 [num_cams, num_scales, 2]
            - scale_start_index: 各尺度在压平张量中的起始位置 [num_cams, num_scales]
        - 反向: 多相机多尺度特征列表
    """
    # 反向转换：从压平格式恢复到原始格式
    if inverse:
        # 解包输入数据
        col_feats, spatial_shape, scale_start_index = feature_maps
        # 获取相机数和尺度数
        num_cams, num_levels = spatial_shape.shape[:2]

        # 计算各相机各尺度的特征点数量（高度 × 宽度）
        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        split_size = split_size.cpu().numpy().tolist()

        # 初始化相机分组索引和大小
        idx = 0
        cam_split = [1]  # 相机组内相机数量
        cam_split_size = [sum(split_size[0])]  # 各相机组的总特征点数

        # 按空间尺寸分组相机（同尺寸相机分为一组）
        for i in range(num_cams - 1):
            # 如果相邻相机的尺寸不同，则开始新的分组
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])

        # 重新整形 col_feats 为多相机格式
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        # 将 spatial_shape 转为 Python 列表便于处理
        spatial_shape = spatial_shape.cpu().numpy().tolist()
        mc_ms_feat = []
        shape_index = 0

        # 逐个恢复各相机各尺度的特征格式
        for i, feat in enumerate(mc_feat):
            # 按各尺度的特征点数分割
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                # 恢复为 [bs, num_cams, height, width, channels] 的格式
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                # 调整维度顺序为 [bs, num_cams, channels, height, width]
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

    # 正向转换：将多个特征图转换为压平格式
    # 处理嵌套列表/元组的情况（多组特征图）
    if isinstance(feature_maps[0], (list, tuple)):
        # 递归地格式化每组特征图
        formated = [feature_maps_format(x) for x in feature_maps]
        # 拼接各组的压平特征
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        # 拼接各组的空间形状信息
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        # 拼接各组的尺度起始索引
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    # 获取批次大小和相机数
    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []

    # 压平各尺度的特征图
    col_feats = []
    for i, feat in enumerate(feature_maps):
        # 记录各尺度的空间尺寸 [height, width]
        spatial_shape.append(feat.shape[-2:])
        # 调整张量形状为 [bs, num_cams, num_levels, total_pixels]
        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )

    # 拼接所有尺度的特征，并调整维度顺序为 [bs, num_cams*total_pixels, channels]
    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)

    # 复制空间形状信息为各相机数量的副本
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )

    # 计算各尺度特征在压平张量中的起始索引
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)

    # 返回格式化后的特征数据
    feature_maps = [
        col_feats,
        spatial_shape,
        scale_start_index,
    ]
    return feature_maps
