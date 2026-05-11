import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import BBOX_SAMPLERS

from projects.mmdet3d_plugin.core.box3d import *
from ..base_target import BaseTargetWithDenoising


__all__ = ["SparseBox3DTarget"]


@BBOX_SAMPLERS.register_module()
class SparseBox3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
    ):
        super(SparseBox3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        # cls_weight / box_weight: Hungarian matching 中分类项和框项的全局权重。
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        # alpha / gamma / eps: focal-style 分类代价的超参数。
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        # reg_weights: 各个 box 状态维度在匹配和损失中的基础权重。
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        # cls_wise_reg_weights: 对某些类别覆盖默认回归维度权重。
        self.cls_wise_reg_weights = cls_wise_reg_weights
        # dn_noise_scale: 给 denoising query 加噪时，各状态维的扰动尺度。
        self.dn_noise_scale = dn_noise_scale
        # max_dn_gt: 每帧参与 dn 构造的 GT 上限。
        self.max_dn_gt = max_dn_gt
        # add_neg_dn: 是否额外构造明显偏离 GT 的负 dn 查询。
        self.add_neg_dn = add_neg_dn

    def encode_reg_target(self, box_target, device=None):
        outputs = []
        for box in box_target:
            # 训练统一在编码空间回归：尺度取 log，朝向改成 sin/cos，避免角度周期不连续。
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]].log(),
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    box[..., YAW + 1 :],
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def sample(
        self,
        cls_pred,
        box_pred,
        cls_target,
        box_target,
    ):
        # bs: batch size；num_pred: 每帧 query 数；num_cls: 类别数。
        bs, num_pred, num_cls = cls_pred.shape

        # cls_cost: 分类匹配代价，长度为 bs 的列表。
        cls_cost = self._cls_cost(cls_pred, cls_target)

        # 把 GT box 编码到与预测相同的回归空间。
        box_target = self.encode_reg_target(box_target, box_pred.device)

        # instance_reg_weights[i]: 第 i 个样本中，每个 GT 在每个状态维上的有效权重。
        instance_reg_weights = []
        for i in range(len(box_target)):
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    # 某些类别可禁用部分维度回归，例如不预测速度。
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights,
                    )
            instance_reg_weights.append(weights)
        # box_cost: 几何匹配代价，长度为 bs 的列表。
        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        indices = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                # 标准 Hungarian matching：分类代价 + 几何代价，做一对一分配。
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                assign = linear_sum_assignment(cost)
                indices.append(
                    [cls_pred.new_tensor(x, dtype=torch.int64) for x in assign]
                )
            else:
                indices.append([None, None])

        # 默认全设为背景类 index=num_cls，匹配到 GT 的位置稍后再覆盖。
        output_cls_target = (
            cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )
        # output_box_target / output_reg_weights 只在正匹配位置有意义。
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][
                target_idx
            ]
        return output_cls_target, output_box_target, output_reg_weights

    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        # 先映射到 sigmoid 概率，再计算与 focal loss 同形态的代价。
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                # 这里的代价形式刻意贴近 focal loss，本质上是在估计“分到该类有多划算”。
                neg_cost = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                )
                pos_cost = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                )
                # 得到 (num_pred, num_gt) 的分类代价矩阵。
                cost.append(
                    (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                    * self.cls_weight
                )
            else:
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            if len(box_target[i]) > 0:
                # 不同 box 维度可配置不同权重，例如位置通常比速度更重要。
                # 输出形状为 (num_pred, num_gt)，可以直接与 cls_cost 相加。
                cost.append(
                    torch.sum(
                        torch.abs(box_pred[i, :, None] - box_target[i][None])
                        * instance_reg_weights[i][None]
                        * box_pred.new_tensor(self.reg_weights),
                        dim=-1,
                    )
                    * self.box_weight
                )
            else:
                cost.append(None)
        return cost

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            # 若不做 temporal dn，则无需保留 GT 的实例 id。
            gt_instance_id = None

        if self.max_dn_gt > 0:
            # 限制每帧参与 dn 的 GT 数，避免极端拥挤样本引起显存爆炸。
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        # max_dn_gt: batch 内 GT 数的最大值，用于 pad 对齐。
        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                for x in cls_target
            ]
        )
        box_target = self.encode_reg_target(box_target, cls_target.device)
        box_target = torch.stack(
            [F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0])) for x in box_target]
        )
        box_target = torch.where(
            cls_target[..., None] == -1, box_target.new_tensor(0), box_target
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        # bs: batch size；num_gt: pad 后每帧的 GT 数；state_dims: box 状态维度数。
        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            # 同一帧 GT 复制多组，构造成多个彼此独立的 dn group。
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)

        # 在 GT 周围加噪，构造“被扰动过的查询”，训练模型把它们拉回正确位置。
        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            # 负 dn 样本偏移更大，提醒模型学会区分“离 GT 很远的错误查询”。
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        box_cost = self._box_cost(
            dn_anchor, box_target, torch.ones_like(box_target)
        )
        dn_box_target = torch.zeros_like(dn_anchor)
        # -3 表示“有效但未分到正 GT 的 dn 条目”，主要给分类监督用。
        dn_cls_target = -torch.ones_like(cls_target) * 3
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        for i in range(dn_anchor.shape[0]):
            cost = box_cost[i].cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_id is not None:
                # temporal dn 需要知道 noisy query 背后对应的真实实例 id。
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]
        # 下面把“按 group 平铺”的布局整理回 (bs, num_all_dn, ...) 的统一形式。
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None
        # valid_mask=True 表示该 dn 条目不是 padding。
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )  # valid denotes the items is not from pad.
        attn_mask = dn_box_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            # 只允许同一 dn group 内部互相可见，避免不同噪声组之间信息泄漏。
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        dn_cls_target = dn_cls_target.long()
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_noraml_anchor,
        temporal_valid_mask,
    ):
        # bs: batch size；num_anchor: normal query + dn query 总数。
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            # 没有历史 dn 缓存或当前没有 dn query 时，直接返回原输入。
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # split instance_feature and anchor into non-dn and dn
        # num_dn: 当前帧总 dn query 数。
        num_dn = num_anchor - num_noraml_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, :num_noraml_anchor]
        anchor = anchor[:, :num_noraml_anchor]

        # reshape all dn metas from (bs,num_all_dn,xxx)
        # to (bs, dn_group, num_dn_per_group, xxx)
        num_dn_groups = self.num_dn_groups
        # 这里重新把 num_dn 解释为“每组 dn query 数”。
        num_dn = num_dn // num_dn_groups
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)
        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)

        # update temp_dn_metas by instance_id
        # temp_dn_feat / temp_dn_id: 上一帧缓存下来的 temporal dn 组。
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]

        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        # match=True 表示历史 dn 和当前 dn 对应的是同一个真实实例。
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )
        temp_valid_mask = self.dn_metas["valid_mask"]
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        # handle the misalignment the length of temp_dn to dn caused by the
        # change of num_gt, then concat the temp_dn and dn
        # temp_dn_metas: 历史帧 temporal dn 的各个张量。
        temp_dn_metas = [
            temp_dn_feat,
            temp_dn_anchor,
            temp_reg_target,
            temp_cls_target,
            temp_valid_mask,
            temp_dn_id,
        ]
        # dn_metas: 当前帧 dn 的各个张量。
        dn_metas = [
            dn_feat,
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            valid_mask,
            dn_id,
        ]
        output = []
        for i, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):
            if num_temp_dn < num_dn:
                # 历史 dn 较短时补零，使长度与当前 dn 对齐。
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad
                else:
                    assert temp_meta.dim() == 3
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                # 历史 dn 较长时截断到当前长度。
                temp_meta = temp_meta[:, :, :num_dn]
            mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)
            # temporal_valid_mask=True 的样本使用历史 dn；否则回退到当前 dn。
            temp_meta = torch.where(
                mask, temp_meta, meta[:, :num_temp_dn_groups]
            )
            # 拼回 [temporal dn | 当前剩余 dn]。
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            meta = meta.flatten(1, 2)
            output.append(meta)
        # output[0]/output[1] 分别是更新后的 dn feature 和 dn anchor，要重新接回 normal query。
        output[0] = torch.cat([instance_feature, output[0]], dim=1)
        output[1] = torch.cat([anchor, output[1]], dim=1)
        return output

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        if self.num_temp_dn_groups < 0:
            return
        num_dn_groups = self.num_dn_groups
        # bs: batch size；num_dn: 当前帧总 dn query 数。
        bs, num_dn = dn_instance_feature.shape[:2]
        # num_temp_dn: 每个 dn group 内部的 query 数。
        num_temp_dn = num_dn // num_dn_groups
        # 随机选若干 dn group 作为下一帧 temporal dn 的记忆。
        temp_group_mask = (
            torch.randperm(num_dn_groups) < self.num_temp_dn_groups
        )
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device)
        # 历史 dn 只作为下一帧输入，因此这里先 detach。
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(
                bs, num_dn_groups, num_temp_dn
            )[:, temp_group_mask]
        # dn_metas: 下一帧 update_dn() 所需的全部 temporal dn 状态。
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature,
            dn_anchor=dn_anchor,
            dn_cls_target=dn_cls_target,
            valid_mask=valid_mask,
            dn_id_target=dn_id_target,
        )
