import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln

__all__ = [
    "SparseBox3DRefinementModule",
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DEncoder",
]


@POSITIONAL_ENCODING.register_module()
class SparseBox3DEncoder(BaseModule):
    def __init__(
        self,
        embed_dims,
        vel_dims=3,
        mode="add",
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        super().__init__()
        assert mode in ["add", "cat"]
        # embed_dims: 每个几何子分支的输出维度，或者一个维度列表。
        self.embed_dims = embed_dims
        # vel_dims: 速度分量的维度数。
        self.vel_dims = vel_dims
        # mode 决定不同几何子分支是相加还是拼接。
        self.mode = mode

        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *linear_relu_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        # pos_fc / size_fc / yaw_fc / vel_fc 分别编码位置、尺寸、朝向和速度。
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(2, embed_dims[2])
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3])
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor):
        # 把 3D 框拆成位置、尺寸、朝向、速度分别编码，
        # 对应论文里 decoupled attention 的几何先验。
        # pos_feat / size_feat / yaw_feat: 几何不同子空间的独立编码结果。
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
        if self.mode == "add":
            output = pos_feat + size_feat + yaw_feat
        elif self.mode == "cat":
            output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
            if self.mode == "add":
                output = output + vel_feat
            elif self.mode == "cat":
                output = torch.cat([output, vel_feat], dim=-1)
        if self.output_fc is not None:
            # output_fc: 对融合后的几何 embedding 再做一次统一映射。
            output = self.output_fc(output)
        return output


@PLUGIN_LAYERS.register_module()
class SparseBox3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
        with_quality_estimation=False,
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        # output_dim: refine 模块输出的 box 状态维度。
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        # refine_state: 哪些状态维采用“增量修正 anchor”的更新策略。
        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        # layers: 主回归头，输出 box 状态增量。
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, 2),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        # feature: 实例语义特征与几何编码的融合表示。
        feature = instance_feature + anchor_embed
        # output: 主回归头预测的 box 状态增量。
        output = self.layers(feature)
        # refine 不是从零回归 box，而是在上一轮 anchor 的基础上做增量修正。
        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state]
        )
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            # translation: 先预测位移，再结合时间间隔恢复成速度。
            translation = torch.transpose(output[..., VX:], 0, -1)
            # 速度分支先预测位移，再除以时间间隔恢复成速度，便于跨不同帧率泛化。
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]

        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            # 分类分支直接读取 instance_feature，而不是 box 增量结果。
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        if return_cls and self.with_quality_estimation:
            # quality 分支输出 centerness / yawness，后续既参与训练也参与推理重排。
            quality = self.quality_layers(feature)
        else:
            quality = None
        return output, cls, quality


@PLUGIN_LAYERS.register_module()
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        # num_learnable_pts: 除固定关键点外，额外学习的点数。
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        # fix_scale: 以 box 尺寸为单位定义的固定相对坐标。
        self.fix_scale = nn.Parameter(
            torch.tensor(fix_scale), requires_grad=False
        )
        # num_pts: 固定点 + 学习点的总数。
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        # bs: batch size；num_anchor: 当前待生成关键点的 anchor 数。
        bs, num_anchor = anchor.shape[:2]
        # size: anchor 中尺寸是 log 空间，这里先 exp 还原真实长宽高。
        size = anchor[..., None, [W, L, H]].exp()
        # key_points: 先生成 box 局部坐标系中的固定关键点。
        key_points = self.fix_scale * size
        if self.num_learnable_pts > 0 and instance_feature is not None:
            # 除固定点外，再让模型学习一部分更灵活的采样点，提高取证据的自由度。
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat(
                [key_points, learnable_scale * size], dim=-2
            )

        # rotation_mat: 每个 anchor 的偏航旋转矩阵。
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])

        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        # 再平移到 box 中心，得到世界坐标系中的关键点。
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        # velocity: anchor 记录的速度分量。
        velocity = anchor[..., VX:]
        for i, t_time in enumerate(temp_timestamps):
            time_interval = cur_timestamp - t_time
            # translation: 当前关键点回到历史时刻应减去的位移。
            translation = (
                velocity
                * time_interval.to(dtype=velocity.dtype)[:, None, None]
            )
            temp_key_points = key_points - translation[:, :, None]
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            # vel / vel_dim: anchor 的速度向量及其维度。
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            # T_src2dst: 源坐标系到目标坐标系的刚体变换。
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = anchor[..., [X, Y, Z]]
            if time_intervals is not None:
                time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(
                    dtype=vel.dtype
                )
            else:
                time_interval = None
            if time_interval is not None:
                # 先按速度补偿时间差，再做坐标系变换。
                translation = vel.transpose(0, -1) * time_interval
                translation = translation.transpose(0, -1)
                center = center - translation
            center = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )
            # size 在刚体变换下不变。
            size = anchor[..., [W, L, H]]
            # yaw 用 sin/cos 表示，因此可直接对二维向量做旋转。
            yaw = torch.matmul(
                T_src2dst[..., :2, :2],
                anchor[..., [COS_YAW, SIN_YAW], None],
            ).squeeze(-1)
            # 速度也要同步旋转到目标坐标系。
            vel = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1)
            # dst_anchor: 目标坐标系中的 anchor 状态向量。
            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            # TODO: Fix bug
            # index = [X, Y, Z, W, L, H, COS_YAW, SIN_YAW] + [VX, VY, VZ][:vel_dim]
            # index = torch.tensor(index, device=dst_anchor.device)
            # index = torch.argsort(index)
            # dst_anchor = dst_anchor.index_select(dim=-1, index=index)
            dst_anchors.append(dst_anchor)
        return dst_anchors

    @staticmethod
    def distance(anchor):
        # 仅用平面中心距离衡量 anchor 离自车的远近。
        return torch.norm(anchor[..., :2], p=2, dim=-1)
