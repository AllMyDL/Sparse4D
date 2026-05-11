# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)

try:
    from ..ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    # layers: 一个轻量 MLP 模板，反复用于几何编码、质量分支和 camera encoder。
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        # group_dims: 每个 attention group 分到的通道数。
        self.group_dims = int(embed_dims / num_groups)
        # embed_dims: 实例特征总维度。
        self.embed_dims = embed_dims
        # num_levels: 输入多尺度特征图的层数，例如 FPN 的 4 层。
        self.num_levels = num_levels
        # num_groups: 分组注意力的组数，便于把通道拆组做加权融合。
        self.num_groups = num_groups
        # num_cams: 多目相机数量。
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        # use_deformable_func=True 时走自定义 CUDA op，加速采样与融合。
        self.use_deformable_func = use_deformable_func
        # attn_drop: 训练时对权重做随机丢弃的概率。
        self.attn_drop = attn_drop
        # residual_mode: 输出与原实例特征做 add 还是 cat。
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        # kps_generator: 给每个 3D anchor 生成固定点 + 可学习点。
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        # num_pts: 每个 anchor 最终要采样的关键点数。
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        # output_proj: 融合完视觉证据后，再映射回 embed_dims 维。
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            # camera_encoder: 把每个相机的投影参数编码成相机相关的偏置。
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            # 开启 camera embed 后，每个相机先单独注入偏置，因此 weights_fc 不再显式展开 num_cams。
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            # 否则直接显式为每个 camera-level-point-group 预测一套权重。
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        # 初始时把权重预测层置零，避免刚开始训练时某些视角/尺度被异常放大。
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        # bs: batch size；num_anchor: 当前帧待更新的稀疏实例数。
        bs, num_anchor = instance_feature.shape[:2]
        # 先在每个 3D anchor 内部生成若干关键点，后面会把这些点投到图像上采样。
        # key_points shape: (bs, num_anchor, num_pts, 3)。
        key_points = self.kps_generator(anchor, instance_feature)
        # 权重由实例语义 + 几何编码共同预测，决定该信任哪个相机/尺度/采样点。
        # weights shape: (bs, num_anchor, num_cams, num_levels, num_pts, num_groups)。
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            # points_2d: 每个 3D 关键点投影到各相机图像后的 2D 采样坐标。
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            # CUDA op 期望的权重布局是 (bs, num_anchor, num_pts, num_cams, num_levels, num_groups)。
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            # DAF: 一步完成多视角/多尺度采样和加权融合。
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            # 纯 PyTorch 路径更便于理解：先采样，再做多相机/多尺度融合。
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            # features 此时仍保留 camera/level/point 维度，后面才逐步融合掉。
            features = self.multi_view_level_fusion(features, weights)
            # 先融合多视角/多尺度，再对 num_pts 维求和，得到实例级视觉证据。
            features = features.sum(dim=2)  # fuse multi-point features
        # output: 聚合后的视觉特征，shape 为 (bs, num_anchor, embed_dims)。
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            # 当前配置使用 cat，把“新取到的视觉证据”和“原实例特征”同时交给后续 FFN。
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        # feature: 把实例语义和 anchor 几何编码先做一次融合，作为权重预测的输入。
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            # 相机外参与内参被编码成 camera embedding，
            # 让模型知道同一个 anchor 在不同视角下应如何分配权重。
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            # 给每个 anchor 在每个 camera 上都注入一份 camera-specific 偏置。
            feature = feature[:, :, None] + camera_embed[:, None]

        # weights_fc 输出的最后一维会被拆成 num_groups，
        # 倒数第二维会被 softmax，表示 camera-level-point 维度上的归一化权重。
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            # 训练时随机丢弃部分 attention 权重，减少模型过度依赖单一视角或单一点。
            # mask 与 weights 前几维对齐，只在 camera / point 维度上做随机置零。
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        # 给每个 3D 点补齐齐次坐标 1，便于与 4x4 投影矩阵相乘。
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        # 标准齐次坐标投影：3D 点 -> 相机平面 2D 点。
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        # 用深度 z 做透视除法，得到像素坐标。
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            # 若传入图像宽高，则再把像素坐标归一化到 [0, 1]。
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # num_levels: 多尺度层数；num_cams: 相机数。
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        # grid_sample 需要 [-1, 1] 坐标系，因此把 [0, 1] 再映射一次。
        points_2d = points_2d * 2 - 1
        # flatten 后等价于把 (bs, num_cams) 合并，便于逐尺度直接做 grid_sample。
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            # grid_sample 在每个尺度上按投影位置采样，得到该 anchor 的图像证据。
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        # 重排后 features 的语义是：
        # (bs, num_anchor, num_cams, num_levels, num_pts, embed_dims)。
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        # 先把通道拆成 (num_groups, group_dims)，每组使用一套独立的融合权重。
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        # 先对 camera 维求和，再对 level 维求和，得到每个关键点的融合特征。
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        # embed_dims: 输入特征图的通道数。
        self.embed_dims = embed_dims
        # equal_focal: 用于把不同相机焦距缩放到统一标尺的参考焦距。
        self.equal_focal = equal_focal
        # num_depth_layers: 参与深度辅助监督的 feature 层数。
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            # focal 通常按相机展开，reshape(-1) 后与 flatten 的多目特征对齐。
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            # feat.flatten(end_dim=1): 把 (bs, num_cams, C, H, W) 展平成 (bs*num_cams, C, H, W)。
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            # 先预测“参考焦距下”的深度，再按实际 focal 做比例缩放。
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            # pred/gt 最终都拉平成一维，只对有效深度像素计算 L1 误差。
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            # 深度值裁剪到合理范围，避免极端预测主导辅助损失。
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            # pre_norm: 在 FFN 前先做一次归一化，和 Transformer 的 pre-norm 结构一致。
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            # identity_fc: 当输入维和输出维不一致时，先把残差支路投影到 embed_dims。
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            # 默认直接拿当前输入做残差分支。
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
