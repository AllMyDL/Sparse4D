# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
import sys

sys.path.append(".")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset

# from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from tools.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet benchmark a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file")
    parser.add_argument("--samples", default=2000, help="samples to benchmark")
    parser.add_argument(
        "--log-interval", default=50, help="interval of logging"
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    args = parser.parse_args()
    return args


def get_max_memory(model):
    # output_device: MMDataParallel 包装后记录的主输出 GPU 编号。
    device = getattr(model, "output_device", None)
    # max_memory_allocated: 当前进程自启动以来在该 GPU 上申请过的峰值显存。
    mem = torch.cuda.max_memory_allocated(device=device)
    # 转成 MB，便于日志打印。
    mem_mb = torch.tensor(
        [mem / (1024 * 1024)], dtype=torch.int, device=device
    )
    return mem_mb.item()


def main():
    args = parse_args()

    # 读取模型配置文件。
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        # 对固定输入尺寸场景开启 benchmark，可让 cuDNN 自动寻找更快的卷积算法。
        torch.backends.cudnn.benchmark = True
    # 基准测试只做推理，因此关闭预训练权重重复加载和训练模式。
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # 打印测试集配置，方便确认正在 benchmark 的数据集。
    print(cfg.data.test)
    # 构造测试数据集对象。
    dataset = custom_build_dataset(cfg.data.test)
    # 构造 dataloader；benchmark 中固定单卡、单样本顺序推理。
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    # 推理阶段不需要 train_cfg。
    cfg.model.train_cfg = None
    # 根据配置构建检测器模型。
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    # 如果配置开启了 fp16，这里把模型包成半精度版本。
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        # 加载训练好的参数；map 到 cpu 再由 DataParallel 放到 GPU。
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        # 将 Conv 和 BN 融合为单个 Conv，减少推理开销。
        model = fuse_module(model)

    # 用单机单卡的数据并行包装，方便复用 MMDet 的标准调用接口。
    model = MMDataParallel(model, device_ids=[0])

    # 进入 eval 模式，关闭 dropout 并固定 BN 统计量。
    model.eval()

    # the first several iterations may be very slow so skip them
    # num_warmup: 预热轮数，避免首次 cudnn kernel 编译/缓存影响平均速度。
    num_warmup = 5
    # pure_inf_time: 只统计正式 benchmark 阶段的累计推理时间。
    pure_inf_time = 0

    # benchmark with several samples and take the average
    # max_memory: 遍历过程中记录最大显存峰值。
    max_memory = 0
    for i, data in enumerate(data_loader):
        # torch.cuda.synchronize()
        with torch.no_grad():
            # start_time: 单个 batch 推理开始时间。
            start_time = time.perf_counter()
            # return_loss=False 表示走测试前向；rescale=True 表示输出还原到原图尺度。
            model(return_loss=False, rescale=True, **data)

            # 等待 GPU 上的异步 kernel 执行完，再统计真实耗时。
            torch.cuda.synchronize()
            # elapsed: 当前 batch 的端到端推理耗时。
            elapsed = time.perf_counter() - start_time
            # 更新已观测到的最大显存占用。
            max_memory = max(max_memory, get_max_memory(model))

        if i >= num_warmup:
            # 只在预热结束后累计时间。
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                # fps: 当前正式统计阶段的平均吞吐。
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f"Done image [{i + 1:<3}/ {args.samples}], "
                    f"fps: {fps:.1f} img / s, "
                    f"gpu mem: {max_memory} M"
                )

        if (i + 1) == args.samples:
            # 达到指定样本数后打印最终平均速度并退出。
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f"Overall fps: {fps:.1f} img / s")
            break


if __name__ == "__main__":
    main()
