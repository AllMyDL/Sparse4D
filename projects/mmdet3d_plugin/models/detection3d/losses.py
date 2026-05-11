import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES

from projects.mmdet3d_plugin.core.box3d import *


@LOSSES.register_module()
class SparseBox3DLoss(nn.Module):
    def __init__(
        self,
        loss_box,
        loss_centerness=None,
        loss_yawness=None,
        cls_allow_reverse=None,
    ):
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        # loss_box: 主体 3D box 回归损失，一般是 L1。
        self.loss_box = build(loss_box, LOSSES)
        # loss_cns: centerness 质量分支的损失。
        self.loss_cns = build(loss_centerness, LOSSES)
        # loss_yns: yawness 质量分支的损失。
        self.loss_yns = build(loss_yawness, LOSSES)
        # cls_allow_reverse: 某些类别允许前后方向翻转而不算错，例如 barrier。
        self.cls_allow_reverse = cls_allow_reverse

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        # Some categories do not distinguish between positive and negative
        # directions. For example, barrier in nuScenes dataset.
        if self.cls_allow_reverse is not None and cls_target is not None:
            # if_reverse=True 表示当前预测框和 GT 在朝向上近似反向。
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                < 0
            )
            # 只对允许“朝向反转等价”的类别启用这个修正。
            if_reverse = (
                torch.isin(
                    cls_target, cls_target.new_tensor(self.cls_allow_reverse)
                )
                & if_reverse
            )
            # 把 GT 的 sin/cos 一起取反，相当于把朝向翻转 180 度。
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}
        # 主体 box loss 仍然是几何回归误差。
        box_loss = self.loss_box(
            box, box_target, weight=weight, avg_factor=avg_factor
        )
        output[f"loss_box{suffix}"] = box_loss

        if quality is not None:
            # quality[..., CNS]: 模型预测的 centerness logits。
            cns = quality[..., CNS]
            # quality[..., YNS]: 模型预测的 yawness logits，这里先过 sigmoid。
            yns = quality[..., YNS].sigmoid()
            # centerness 监督：中心越接近 GT，质量 target 越接近 1。
            # cns_target 是一个连续值，而不是硬 0/1 标签。
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
            output[f"loss_cns{suffix}"] = cns_loss

            # yawness 监督：判断预测朝向与 GT 是否同向。
            # yns_target=1 表示 yaw 方向一致，=0 表示方向相反。
            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                > 0
            )
            yns_target = yns_target.float()
            yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
            output[f"loss_yns{suffix}"] = yns_loss
        return output
