# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *


@BBOX_CODERS.register_module()
class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        super(SparseBox3DDecoder, self).__init__()
        # num_output: 每帧最终最多保留的候选框数量。
        self.num_output = num_output
        # score_threshold: 可选的分数阈值，低于该阈值的候选会被过滤。
        self.score_threshold = score_threshold
        # sorted=True 时，top-k 返回的候选会按分数排序。
        self.sorted = sorted

    def decode_box(self, box):
        # box: 编码空间下的预测框，包含 [x,y,z, log(w),log(l),log(h), sin(yaw),cos(yaw), ...]。
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        box = torch.cat(
            [
                box[:, [X, Y, Z]],
                # 尺度在训练时使用 log 空间回归，这里恢复到真实长宽高。
                box[:, [W, L, H]].exp(),
                yaw[:, None],
                box[:, VX:],
            ],
            dim=-1,
        )
        return box

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        qulity=None,
        output_idx=-1,
    ):
        # squeeze_cls=True 表示当前处于 tracking 场景，
        # 每个 query 最终只保留一个最佳类别，而不是保留 query x class 的全展开候选。
        squeeze_cls = instance_id is not None

        # 取指定 decoder 层输出，并把分类 logits 转成概率。
        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            # tracking 场景下先对类别维取最大值，得到每个 query 的最佳类别和得分。
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        # bs: batch size；num_pred: query 数；num_cls: 类别数或 1。
        bs, num_pred, num_cls = cls_scores.shape
        # 先从所有 query、所有类别中挑出 top-k 候选，再做后续质量重排。
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            # 非 tracking 场景下，indices 来自 flatten(query, class)，
            # 因此对 num_cls 取模可还原类别 id。
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            # mask=True 表示该 top-k 候选原始分类分数过阈值。
            mask = cls_scores >= self.score_threshold

        if qulity is not None:
            # centerness: 只取出 top-k 对应 query 的质量预测。
            centerness = qulity[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            # cls_scores_origin: 保留质量重排前的原始分类分数，tracking 过滤时会用到。
            cls_scores_origin = cls_scores.clone()
            # 最终排序分数 = 分类置信度 * 几何中心质量。
            cls_scores *= centerness.sigmoid()
            # idx: 质量重排后的新排序索引。
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            # category_ids: 第 i 个样本里，每个候选框对应的类别 id。
            category_ids = cls_ids[i]
            if squeeze_cls:
                # tracking 模式下 cls_ids 是“按 query 存”的，因此要再用 indices 映射回 top-k 顺序。
                category_ids = category_ids[indices[i]]
            # scores: 最终输出分数，若有 quality 则已完成质量重排。
            scores = cls_scores[i]
            # indices // num_cls 可从 flatten(query, class) 的索引还原出 query 下标。
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if qulity is not None:
                # scores_origin: 质量重排前的原始分类分数。
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            # 把编码空间下的 box 还原成真实物理量，供评测和可视化使用。
            box = self.decode_box(box)
            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
            if qulity is not None:
                # 额外保留原始分类分数，tracking 阈值过滤时可能会用到。
                output[-1]["cls_scores"] = scores_origin.cpu()
            if instance_id is not None:
                # instance_id 与 query 对齐，因此同样按 indices 映射到 top-k 候选。
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids
        return output
