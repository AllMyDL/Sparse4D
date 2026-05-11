import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    # confidence: (bs, N, ...) 或至少前两维是 batch 和候选实例数。
    bs, N = confidence.shape[:2]
    # 对每个 batch 取置信度最高的 k 个实例。
    confidence, indices = torch.topk(confidence, k, dim=1)
    # 把二维下标 (batch_id, instance_id) 展平成一维下标，
    # 便于直接在 flatten 后的张量里做索引。
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        # 对每个伴随张量执行完全相同的 top-k 选择，
        # 例如同步筛选 feature、anchor 或 instance_id。
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
    ):
        super(InstanceBank, self).__init__()
        # embed_dims: 每个稀疏实例特征向量的维度。
        self.embed_dims = embed_dims
        # num_temp_instances: 允许从历史帧保留下来的实例数量。
        self.num_temp_instances = num_temp_instances
        # default_time_interval: 无法可靠计算时间差时使用的默认帧间隔。
        self.default_time_interval = default_time_interval
        # confidence_decay: 历史实例置信度的衰减系数。
        self.confidence_decay = confidence_decay
        # max_time_interval: 若跨帧时间差过大，则认为历史实例不再可靠。
        self.max_time_interval = max_time_interval

        if anchor_handler is not None:
            # anchor_handler 负责跨坐标系投影 anchor，以及必要的几何变换。
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            # 若 anchor 是文件路径，则从磁盘加载预先聚类好的初始 anchors。
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        # num_anchor: 实际使用的基础 anchor 数，不超过提供的 anchor 总数。
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        # anchor_init: 记录初始化值，便于 reset/init_weight 时恢复。
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        # 重新恢复到初始 anchor 分布，常用于模型初始化阶段。
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            # 可学习实例特征采用 Xavier 初始化，使不同 anchor 槽位具备可分性。
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        # cached_feature: 历史帧缓存下来的 top-k 实例特征。
        self.cached_feature = None
        # cached_anchor: 与 cached_feature 对应的历史 3D anchor 状态。
        self.cached_anchor = None
        # metas: 生成 cached_* 时那一帧的元信息，用于下一帧做坐标变换。
        self.metas = None
        # mask: 当前 batch 中哪些样本可以安全复用历史实例。
        self.mask = None
        # confidence: 上一轮最终缓存下来的历史实例置信度。
        self.confidence = None
        # temp_confidence: 当前帧临时的置信度，用于更新 instance id。
        self.temp_confidence = None
        # instance_id: 缓存实例对应的追踪 id。
        self.instance_id = None
        # prev_id: 下一个新建 tracking id 的起始编号。
        self.prev_id = 0

    def get(self, batch_size, metas=None, dn_metas=None):
        # 把 learnable 的基础 instance feature 复制到 batch 维。
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        # 把 learnable 的基础 anchors 复制到 batch 维。
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            # history_time: 历史缓存对应帧的时间戳。
            history_time = self.metas["timestamp"]
            # time_interval: 当前帧与历史缓存帧的时间差，shape 通常为 (bs,)。
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            # mask=True 表示该样本的历史实例仍然足够新，可以参与当前帧推理。
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                # 把历史帧坐标系下的 anchor 投影到当前帧坐标系，
                # 这样 temp_gnn 用到的历史实例才和当前图像处在同一个几何空间里。
                # T_temp2cur: 历史帧 -> 当前帧的 4x4 坐标变换矩阵。
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                # cached_anchor 会被原地更新为“已经对齐到当前帧”的版本。
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                # num_dn_group: dn 分组数；num_dn: 每组 dn anchor 数量。
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                # temporal dn 也必须同步从历史帧投影到当前帧坐标系。
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            # 若时间差非法或历史实例无效，则回退到默认时间间隔，避免后续除法/速度补偿异常。
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            # 无历史缓存或 batch 形状不匹配时，直接清空历史状态，从纯 learnable anchors 开始。
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        # 若没有历史缓存，则当前帧实例保持原样，不做时序融合。
        if self.cached_feature is None:
            return instance_feature, anchor

        # num_dn: 当前 instance 列表尾部额外拼接进来的 dn query 数量。
        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            # dn_* 先暂存，避免和正常实例一起参与时序 top-k 选择。
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        # N: 当前帧要保留的“新实例”数量，其余槽位让给历史实例。
        N = self.num_anchor - self.num_temp_instances
        # 多类别分类分数取最大值，作为实例级置信度。
        confidence = confidence.max(dim=-1).values
        # 当前帧保留一部分高置信实例，另一部分槽位留给上一帧传播过来的历史实例。
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        # 按顺序拼成：[历史实例 | 当前高置信实例]。
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        # 只有 mask=True 的样本才真的用历史实例覆盖当前列表；否则保留当前帧纯新实例。
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        if self.instance_id is not None:
            # 对无法使用历史实例的样本，把旧 id 置为 -1，防止错误继承。
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        if num_dn > 0:
            # 把之前剥离出去的 dn query 再拼回末尾，保持后续 loss 所需的布局不变。
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        # detach: 历史缓存只作为下一帧的输入，不应该反向传播回当前帧图。
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        # metas: 保存当前帧的位姿与时间戳，供下一帧做历史实例投影。
        self.metas = metas
        # 每个实例取最大类别分数作为实例级置信度，并映射到 [0, 1]。
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            # 历史置信度做衰减，防止旧实例长期压制当前帧的新证据。
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        # temp_confidence: 当前帧所有基础实例的临时置信度，稍后更新 instance_id 时会用到。
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)
        # cached_* 只保留 top-k 高置信实例，作为下一帧的时序记忆。

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        # 推理时同样把多类别分数压成实例级置信度。
        confidence = confidence.max(dim=-1).values.sigmoid()
        # -1 表示该位置当前还没有合法的 tracking id。
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            # 先继承历史缓存实例的 id，形成跟踪的时间连续性。
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        # mask=True 表示该位置还没有 id，需要考虑分配新 id。
        mask = instance_id < 0
        if threshold is not None:
            # 只给足够高置信的实例分配新 id，减少短命误检轨迹。
            mask = mask & (confidence >= threshold)
        # 对还没有继承到历史 id 的高置信实例分配新 id。
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        instance_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        if self.num_temp_instances > 0:
            self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        # temp_confidence 优先使用 cache() 中记录的当前帧置信度，
        # 保证选出的 id 与下一帧真正会缓存的历史实例一致。
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        # 只保留 top-k 历史实例对应的 id，和 cached_feature/cached_anchor 的布局对齐。
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        instance_id = instance_id.squeeze(dim=-1)
        # 后半部分未被缓存的槽位统一补成 -1，表示这些位置没有稳定历史 id。
        self.instance_id = F.pad(
            instance_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )
