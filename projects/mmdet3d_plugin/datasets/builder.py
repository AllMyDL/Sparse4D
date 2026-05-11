import copy
import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from mmdet.datasets.samplers import GroupSampler
from projects.mmdet3d_plugin.datasets.samplers import (
    # GroupInBatchSampler: IterBasedRunner 下为每个 batch 槽位分配独立序列。
    GroupInBatchSampler,
    # DistributedGroupSampler: 分布式训练时按 group 打散并均匀切分。
    DistributedGroupSampler,
    # DistributedSampler: 分布式测试时按顺序切分样本。
    DistributedSampler,
    # build_sampler: 通过 registry 动态实例化 sampler。
    build_sampler
)


def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dist=True,
    shuffle=True,
    seed=None,
    shuffler_sampler=None,
    nonshuffler_sampler=None,
    runner_type="EpochBasedRunner",
    **kwargs
):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    # rank: 当前进程编号；world_size: 分布式总进程数。
    rank, world_size = get_dist_info()
    # batch_sampler: IterBasedRunner 会显式接管“如何组成一个 batch”。
    batch_sampler = None
    if runner_type == 'IterBasedRunner':
        print("Use GroupInBatchSampler !!!")
        # IterBasedRunner 下通过 batch_sampler 显式控制每一步取样本的方式。
        batch_sampler = GroupInBatchSampler(
            dataset,
            samples_per_gpu,
            world_size,
            rank,
            seed=seed,
        )
        # 使用 batch_sampler 时，DataLoader 的 batch_size 固定设为 1。
        batch_size = 1
        # sampler 与 batch_sampler 二选一，因此这里置空。
        sampler = None
        # num_workers: 每张卡的数据加载进程数。
        num_workers = workers_per_gpu
    elif dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            print("Use DistributedGroupSampler !!!")
            sampler = build_sampler(
                shuffler_sampler
                if shuffler_sampler is not None
                else dict(type="DistributedGroupSampler"),
                dict(
                    dataset=dataset,
                    samples_per_gpu=samples_per_gpu,
                    num_replicas=world_size,
                    rank=rank,
                    seed=seed,
                ),
            )
        else:
            # 分布式测试通常不打乱顺序，因此走非 shuffle 的 sampler。
            sampler = build_sampler(
                nonshuffler_sampler
                if nonshuffler_sampler is not None
                else dict(type="DistributedSampler"),
                dict(
                    dataset=dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    seed=seed,
                ),
            )

        # 分布式模式下一张卡只关心本卡 batch 大小。
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print("WARNING!!!!, Only can be used for obtain inference speed!!!!")
        # 非分布式路径主要保留给单机 benchmark / debug 使用。
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    # init_fn: 每个 worker 启动时设置独立随机种子，避免增强完全同步。
    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    # DataLoader: 最终把 dataset + sampler + collate_fn 拼装成 PyTorch 可迭代对象。
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs
    )

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    # worker_id: 当前 dataloader worker 编号。
    worker_seed = num_workers * rank + worker_id + seed
    # 保证多进程数据增强时的随机性可复现。
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    # RLIMIT_NOFILE: 进程可同时打开的文件数上限。
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    # 数据集含大量图片/点云文件时，适当调高文件句柄限制可减少 I/O 报错。
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

# OBJECTSAMPLERS: 与 mmdet3d 兼容保留的 registry，当前项目里几乎没直接用到。
OBJECTSAMPLERS = Registry("Object sampler")


def custom_build_dataset(cfg, default_args=None):
    # CBGSDataset: class-balanced group sampler 包装器，在 mmdet3d 中常见。
    try:
        from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    except:
        CBGSDataset = None
    from mmdet.datasets.dataset_wrappers import (
        ClassBalancedDataset,
        ConcatDataset,
        RepeatDataset,
    )

    if isinstance(cfg, (list, tuple)):
        # list/tuple 配置会被拼成一个大的 ConcatDataset。
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg]
        )
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg["datasets"]],
            cfg.get("separate_eval", True),
        )
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            custom_build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg["dataset"], default_args),
            cfg["oversample_thr"],
        )
    elif cfg["type"] == "CBGSDataset":
        dataset = CBGSDataset(
            custom_build_dataset(cfg["dataset"], default_args)
        )
    elif isinstance(cfg.get("ann_file"), (list, tuple)):
        # 一个配置下有多个 ann_file 时，复用 MMDet 的 concat 逻辑。
        dataset = _concat_dataset(cfg, default_args)
    else:
        # 普通数据集最终都走 registry 动态构建。
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
