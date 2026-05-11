import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class Grid(object):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        # use_h / use_w: 是否分别沿高/宽方向生成条带状遮挡。
        self.use_h = use_h
        self.use_w = use_w
        # rotate: mask 允许的最大随机旋转角度范围。
        self.rotate = rotate
        # offset=True 时，被遮挡区域不直接置零，而是加随机偏移噪声。
        self.offset = offset
        # ratio: 每个网格周期中，被遮挡条带所占的比例。
        self.ratio = ratio
        # mode=1 时反转 mask，保留条带、遮挡其余区域。
        self.mode = mode
        # st_prob: 初始设定概率；prob: 当前实际使用概率。
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        # 训练前期弱一些，后期逐步把增强概率拉到 st_prob。
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        # h / w: 单张图像的高宽。
        h = img.size(1)
        w = img.size(2)
        # d1 / d2: 网格周期 d 的采样范围。
        self.d1 = 2
        self.d2 = min(h, w)
        # hh / ww 扩到 1.5 倍，是为了旋转后再中心裁剪时不丢边缘。
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        # d: 网格周期；l: 每个周期里被遮挡条带的宽度。
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                # s / t: 当前横向遮挡条带的起止位置。
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        # 旋转后能得到更丰富的遮挡方向，而不仅是纯水平/垂直条带。
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        # 再从中心裁回原图大小。
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        # expand_as(img): 将单通道 2D mask 扩展到与图像相同的通道数。
        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


class GridMask(nn.Module):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        super(GridMask, self).__init__()
        # 下面这些参数和上面的 Grid 类含义一致，这里是 nn.Module 版本供模型中直接调用。
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        # 随 epoch 线性调节实际生效概率。
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        # n / c / h / w: batch 大小、通道数、高、宽。
        n, c, h, w = x.size()
        # 先把 batch 和通道维合并，对每个通道应用同一张 2D mask。
        x = x.view(-1, h, w)
        # hh / ww 扩大到 1.5 倍，方便旋转后再中心裁剪。
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        # d: 网格周期；self.l: 遮挡条带宽度。
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                # 沿高度方向生成一组条带遮挡。
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                # 沿宽度方向生成一组条带遮挡。
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        # 从放大后的画布中心裁回原图大小。
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask.copy()).float().cuda()
        if self.mode == 1:
            mask = 1 - mask
        # mask 会广播到合并后的 (n*c, h, w) 形状。
        mask = mask.expand_as(x)
        if self.offset:
            offset = (
                torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
                .float()
                .cuda()
            )
            # 被遮挡区域注入随机偏移，而不是简单清零。
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        # 恢复回原始的 (n, c, h, w) 形状。
        return x.view(n, c, h, w)
