"""
可变形特征聚合的 PyTorch 自动求导实现
=========================================
该模块实现了 PyTorch 的自定义自动求导函数（autograd Function）
用于包装 CUDA 的可变形聚合前向和反向计算
"""
import torch
from torch.autograd.function import Function, once_differentiable

from . import deformable_aggregation_ext  # CUDA 扩展模块


class DeformableAggregationFunction(Function):
    """
    可变形特征聚合的自动求导函数

    这是一个 PyTorch 自定义算子，连接 Python 代码与 CUDA 高性能计算
    支持前向传播和反向传播（用于梯度计算）
    """

    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,           # 多相机多尺度特征张量
        spatial_shape,        # 各尺度的空间尺寸
        scale_start_index,    # 尺度起始索引
        sampling_location,    # 采样点坐标
        weights,              # 采样权重
    ):
        """
        前向传播：执行特征聚合计算

        参数:
            ctx: PyTorch 上下文对象，用于保存反向传播所需的数据
            mc_ms_feat (Tensor): 多相机多尺度特征 [bs, num_feat, num_embeds]
                - bs: 批次大小
                - num_feat: 压平后的总特征数（多相机×多尺度特征点总数）
                - num_embeds: 特征的通道维度
            spatial_shape (Tensor): 各相机各尺度的空间尺寸 [num_cams, num_scales, 2]
                - 2 表示 [height, width]
            scale_start_index (Tensor): 各尺度在压平张量中的起始索引 [num_cams, num_scales]
            sampling_location (Tensor): 采样点坐标 [bs, num_anchors, num_pts, num_cams, 2]
                - num_anchors: 锚点（对象查询）数量
                - num_pts: 每个锚点的采样点数
                - 2 表示 [x, y] 坐标
            weights (Tensor): 采样权重 [bs, num_anchors, num_pts, num_cams, num_scales, num_groups]
                - num_scales: 使用的特征尺度数
                - num_groups: 权重分组数（通常用于多头注意）

        返回:
            output (Tensor): 聚合后的特征 [bs, num_anchors, num_embeds]
        """
        # 确保所有输入张量在 GPU 内存中且格式正确
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # 调用 CUDA 扩展的前向函数
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )

        # 保存张量用于反向传播（计算梯度时需要这些数据）
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        反向传播：计算各输入对损失函数的梯度

        参数:
            ctx: PyTorch 上下文对象，包含前向传播保存的数据
            grad_output (Tensor): 输出的梯度 [bs, num_anchors, num_embeds]

        返回:
            元组: (grad_mc_ms_feat, None, None, grad_sampling_location, grad_weights)
                - grad_mc_ms_feat: 输入特征的梯度
                - None: spatial_shape 和 scale_start_index 不需要梯度
                - grad_sampling_location: 采样位置的梯度
                - grad_weights: 权重的梯度
        """
        # 恢复前向传播保存的张量
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors

        # 确保所有张量格式正确
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        # 初始化梯度张量
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)           # 特征梯度
        grad_sampling_location = torch.zeros_like(sampling_location)  # 采样位置梯度
        grad_weights = torch.zeros_like(weights)                # 权重梯度

        # 调用 CUDA 扩展的反向函数计算梯度
        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )

        # 返回梯度元组
        # 顺序必须与 forward 的输入参数一致
        return (
            grad_mc_ms_feat,           # mc_ms_feat 的梯度
            None,                      # spatial_shape 不需要梯度
            None,                      # scale_start_index 不需要梯度
            grad_sampling_location,    # sampling_location 的梯度
            grad_weights,              # weights 的梯度
        )
