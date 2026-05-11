# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from torch import nn as nn
from mmcv.runner import save_checkpoint
from mmdet.apis import init_detector


def fuse_conv_bn(conv, bn):
    """During inference, the functionary of batch norm layers is turned off but
    only the mean and var alone channels are used, which exposes the chance to
    fuse it with the preceding conv layers to save computations and simplify
    network structures."""
    # conv_w: 卷积层权重，形状通常为 [out_channels, in_channels, kH, kW]。
    conv_w = conv.weight
    # conv_b: 卷积层偏置；若原卷积无 bias，则补一个与 BN 通道数一致的零向量。
    conv_b = (
        conv.bias
        if conv.bias is not None
        else torch.zeros_like(bn.running_mean)
    )

    # factor: BN 融合到卷积时每个输出通道的缩放系数。
    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    # 将 BN 的缩放合并进卷积权重。
    conv.weight = nn.Parameter(
        conv_w * factor.reshape([conv.out_channels, 1, 1, 1])
    )
    # 将 BN 的平移合并进卷积偏置。
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_module(m):
    # last_conv: 记录当前遍历层级中最近遇到的 Conv2d，用于和后续紧邻的 BN 配对融合。
    last_conv = None
    # last_conv_name: 记录该卷积层在父模块中的名字，便于融合后替换回去。
    last_conv_name = None

    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            # fused_conv: 融合 BN 参数后的卷积层。
            fused_conv = fuse_conv_bn(last_conv, child)
            # 用融合后的卷积替换原来的卷积模块。
            m._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            # 保留模块位置但将 BN 变成恒等映射，避免影响上层结构和 state dict 键名。
            m._modules[name] = nn.Identity()
            # 当前 Conv-BN 对处理完后清空缓存，避免错误跨层融合。
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            # 遇到卷积时先缓存下来，等待看看下一个模块是否是可融合的 BN。
            last_conv = child
            last_conv_name = name
        else:
            # 递归进入子模块，处理更深层的网络结构。
            fuse_module(child)
    return m


def parse_args():
    parser = argparse.ArgumentParser(
        description="fuse Conv and BN layers in a model"
    )
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file path")
    parser.add_argument("out", help="output path of the converted model")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint)
    # fuse conv and bn layers of the model
    fused_model = fuse_module(model)
    save_checkpoint(fused_model, args.out)


if __name__ == "__main__":
    main()
