# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from .sampler import SAMPLER
import random
from IPython import embed


@SAMPLER.register_module()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(
        self, dataset, samples_per_gpu=1, num_replicas=None, rank=None, seed=0
    ):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, "flag")
        # flag: 每个样本所属 group，常用于把形状相近或同序列样本放在一起。
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += (
                int(
                    math.ceil(
                        self.group_sizes[i]
                        * 1.0
                        / self.samples_per_gpu
                        / self.num_replicas
                    )
                )
                * self.samples_per_gpu
            )
        # total_size: 所有 rank 加总后，一个 epoch 实际会取到的样本数。
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        # g: 绑定 epoch 的随机数发生器，每个 epoch 采样顺序不同但可复现。
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                # indice: 当前 group 中所有样本的原始下标。
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[
                    list(torch.randperm(int(size), generator=g).numpy())
                ].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas
                    )
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # 每个 group 都补齐到“能被 world_size 和 batch_size 整除”的长度，
                # 这样分布式切分后每张卡都能拿到完整 batch。
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[: extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        # 先以 batch 为单位打乱 group 内样本块，再展开成一维索引序列。
        indices = [
            indices[j]
            for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g
                )
            )
            for j in range(
                i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu
            )
        ]

        # subsample
        # 最后再按 rank 切片，保证不同 GPU 读到互不重叠的数据子集。
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        # epoch 变化后，下一次 __iter__ 会基于新种子重排。
        self.epoch = epoch
