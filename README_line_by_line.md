# Sparse4D 逐行导读版

这份文档是 [`README_study.md`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/README_study.md) 的进阶版。

如果说 `README_study.md` 解决的是“这套代码大概在干什么”，那么这份文档解决的是：

- 关键函数的输入输出分别是什么
- 代码是按什么顺序一步步推进的
- 每一步里最值得盯住的变量有哪些
- 这些实现和 paper 的哪一部分最相关

这份文档不追求覆盖仓库全部代码，而是优先覆盖最核心、最值得你反复读的函数。

## 1. 推荐使用方式

建议你开两个窗口：

1. 左边打开源码
2. 右边打开这份导读

然后按下面顺序读：

1. [`projects/mmdet3d_plugin/models/sparse4d.py`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py:62)
2. [`projects/mmdet3d_plugin/models/sparse4d_head.py`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:163)
3. [`projects/mmdet3d_plugin/models/instance_bank.py`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/instance_bank.py:83)
4. [`projects/mmdet3d_plugin/models/blocks.py`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/blocks.py:111)
5. [`projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py:23)
6. [`projects/mmdet3d_plugin/models/detection3d/target.py`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:48)
7. [`projects/mmdet3d_plugin/models/detection3d/losses.py`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/losses.py:31)
8. [`projects/mmdet3d_plugin/models/detection3d/decoder.py`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/decoder.py:37)

## 2. 主链总览

先把一次训练前向浓缩成一句话：

1. 多相机图像提特征
2. 取出当前 anchor 和历史实例
3. 一轮轮 decoder 做时序交互、实例交互、图像采样和 box refine
4. 用 Hungarian matching 和 denoising 构造监督
5. 计算分类、回归、质量损失

如果你中间读迷糊了，就回到这一句。

---

## 3. 顶层入口：`Sparse4D.extract_feat`

文件：

- [`projects/mmdet3d_plugin/models/sparse4d.py:62`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py:62)

### 3.1 函数职责

这个函数只做一件事：

- 把多目图像变成后续 sparse head 可以消费的多尺度特征

你可以把它理解成 “dense 图像世界 -> sparse 实例世界” 之前的准备阶段。

### 3.2 输入和输出

输入：

- `img`: 通常形状是 `(bs, num_cams, C, H, W)`
- `return_depth`: 是否顺便输出深度辅助分支
- `metas`: 相机参数、图像大小等元信息

输出：

- `feature_maps`: 多尺度图像特征
- `depths`: 可选的深度预测，仅辅助训练使用

### 3.3 逐步看代码

#### 第 1 步：判断是不是多相机输入

看这几行：

- [`sparse4d.py:64-69`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py:64)

阅读提示：

- `img.dim() == 5` 说明输入包含多相机维度。
- `img.flatten(end_dim=1)` 把 `(bs, num_cams, C, H, W)` 压成 `(bs * num_cams, C, H, W)`，这样 backbone 就能按普通 2D 图像处理。

#### 第 2 步：图像增强与 backbone

看：

- [`sparse4d.py:70-75`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py:70)

阅读提示：

- `grid_mask` 是图像增强，不影响算法主干理解。
- `self.img_backbone(img)` 就是普通 CNN 提特征。
- 这里你不用纠结 backbone 细节，重点是后面如何使用这些特征。

#### 第 3 步：FPN 多尺度特征

看：

- [`sparse4d.py:76-80`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py:76)

阅读提示：

- `img_neck` 把 backbone 输出统一成多尺度、同通道数的特征图。
- reshape 后每层特征都回到 `(bs, num_cams, C, H_l, W_l)` 的形式。

这一步很关键，因为后面的 deformable aggregation 要在“每个相机、每个尺度”上采样。

#### 第 4 步：辅助深度分支

看：

- [`sparse4d.py:82-85`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py:82)

阅读提示：

- `depth_branch` 只为训练提供几何辅助监督。
- 它不是最终检测输出的一部分，所以读主干时不必在这里停太久。

#### 第 5 步：给自定义 CUDA deformable op 做格式转换

看：

- [`sparse4d.py:86-90`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py:86)

阅读提示：

- 这一步是工程优化，不是论文核心思想。
- 读算法时只需要知道：后面的 aggregation 会消费这些 feature maps。

### 3.4 读这一段时最该盯住的变量

- `bs`
- `num_cams`
- `feature_maps`
- `depths`

### 3.5 和论文的对应关系

它对应论文中的图像编码部分，也就是给 sparse decoder 提供多视角、多尺度视觉证据。

---

## 4. 训练总入口：`Sparse4D.forward_train`

文件：

- [`projects/mmdet3d_plugin/models/sparse4d.py:99`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py:99)

### 4.1 逐步看代码

看：

- [`sparse4d.py:99-107`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d.py:99)

这个函数非常短，但主链特别清楚：

1. `extract_feat(img, True, data)`：先提图像特征
2. `self.head(feature_maps, data)`：再跑 sparse head 主体
3. `self.head.loss(model_outs, data)`：最后算损失
4. 如果有 `depth_branch`，额外加深度损失

阅读提示：

- 真正难的内容都被折叠进 `self.head(...)` 和 `self.head.loss(...)` 里了。
- 所以你读到这里时要立刻跳去看 `Sparse4DHead.forward`。

---

## 5. 核心主脑：`Sparse4DHead.forward`

文件：

- [`projects/mmdet3d_plugin/models/sparse4d_head.py:163`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:163)

这是全仓库最重要的函数之一。

### 5.1 先记住它在做什么

这个函数做了五件事：

1. 取当前实例和历史实例
2. 构造 denoising queries
3. 按 `operation_order` 一层层执行 decoder
4. 收集分类、回归、质量输出
5. 缓存当前高置信实例供下一帧使用

### 5.2 第 1 段：统一 feature map 形式

看：

- [`sparse4d_head.py:168-170`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:168)

阅读提示：

- 这里是在防御性处理输入，保证后面统一按 `List[Tensor]` 使用多尺度特征。

### 5.3 第 2 段：从 `InstanceBank` 取实例

看：

- [`sparse4d_head.py:172-186`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:172)

这里返回五个量：

- `instance_feature`
- `anchor`
- `temp_instance_feature`
- `temp_anchor`
- `time_interval`

你可以这样理解：

- `instance_feature`：当前帧基础实例特征
- `anchor`：当前帧基础 3D anchor 状态
- `temp_instance_feature`：历史帧缓存下来的实例特征
- `temp_anchor`：历史实例投影到当前坐标系后的 anchor
- `time_interval`：当前帧与历史帧的时间差

阅读提示：

- 这一段就是“时序信息从哪里来”的入口。
- 如果你不知道 `temp_*` 是怎么来的，就去看 `InstanceBank.get()`。

### 5.4 第 3 段：构造 denoising anchors

看：

- [`sparse4d_head.py:188-245`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:188)

这一段是 v3 很关键的地方。

可以按下面顺序理解：

1. 如果在训练，并且 sampler 支持 dn，就调用 `get_dn_anchors(...)`
2. 得到 noisy anchors、对应 GT、有效 mask、注意力 mask
3. 把这些 dn anchors 拼到普通 anchors 后面
4. 为 dn 部分补零初始化的 feature
5. 构造 `attn_mask`，限制不同 dn group 之间的注意力交互

阅读提示：

- `num_free_instance` 表示原本正常的 learnable instances 数量。
- `num_dn_anchor` 表示后面拼进来的 dn queries 数量。
- 你可以把整个张量理解成：
  “前面是正常 query，后面是训练时额外加入的 noisy query”。

### 5.5 第 4 段：anchor 编码

看：

- [`sparse4d_head.py:247-251`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:247)

阅读提示：

- `anchor_embed` 是后续 attention 和 refine 的位置几何提示。
- 论文里的 decoupled attention，就是通过这个编码器展开的。

### 5.6 第 5 段：最核心的 decoder 循环

看：

- [`sparse4d_head.py:253-341`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:253)

这段一定要按 `op` 分支来理解，不要一行行硬啃。

#### `temp_gnn`

看：

- [`sparse4d_head.py:260-271`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:260)

含义：

- 当前实例和历史实例做 attention 交互。

你可以把它理解成：

- “我这一帧的 query，去问上一帧保留下来的重要实例，拿一点时间上下文回来。”

#### `gnn`

看：

- [`sparse4d_head.py:272-279`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:272)

含义：

- 当前帧实例之间做 self-attention。

你可以理解成：

- “同一帧里不同实例之间互相交流，避免重复和冲突。”

#### `norm` / `ffn`

看：

- [`sparse4d_head.py:280-281`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:280)

含义：

- 普通 transformer 风格的规范化和前馈变换。

#### `deformable`

看：

- [`sparse4d_head.py:282-289`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:282)

含义：

- 每个 anchor 去多视角、多尺度图像特征上找证据。

这是论文里的视觉信息注入核心。

#### `refine`

看：

- [`sparse4d_head.py:290-339`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:290)

含义：

- 根据当前特征更新 anchor
- 预测分类分数 `cls`
- 预测质量分数 `qt`

这一段还有两个非常重要的动作：

1. 到了 `num_single_frame_decoder` 之后，把历史实例和当前高置信实例重新组织。
2. 如果开启 temporal dn，还会更新 dn 对齐关系。

阅读提示：

- `prediction.append(anchor)` 说明这里保存的是“每一轮 refine 后的 box 状态”。
- `classification.append(cls)` 和 `quality.append(qt)` 分别保存每一轮分类和质量输出。
- 所以最后 loss 会对多层 decoder 输出都做监督。

### 5.7 第 6 段：拆分正常 query 和 dn query

看：

- [`sparse4d_head.py:345-390`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:345)

阅读提示：

- 因为训练时 normal query 和 dn query 是拼在一起跑的，所以输出后必须再拆开。
- `classification[:num_free_instance]` 是正常实例。
- `classification[num_free_instance:]` 是 dn 实例。

这里做完以后，normal 分支进入正常 loss，dn 分支进入 denoising loss。

### 5.8 第 7 段：缓存当前高置信实例

看：

- [`sparse4d_head.py:399-408`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/sparse4d_head.py:399)

阅读提示：

- `self.instance_bank.cache(...)` 是时序建模闭环的最后一步。
- 推理时 `get_instance_id(...)` 则把当前高置信实例串成 tracking id。

### 5.9 读这一整个函数时最该盯住的变量

- `anchor`
- `instance_feature`
- `temp_anchor`
- `attn_mask`
- `prediction`
- `classification`
- `quality`
- `num_free_instance`

### 5.10 和论文的对应关系

这个函数是论文主方法最集中、最完整的代码映射：

- 时序建模：`temp_gnn + instance_bank`
- 稀疏实例更新：`gnn + refine`
- 图像特征取证：`deformable`
- Temporal Instance Denoising：`get_dn_anchors / update_dn`
- Quality Estimation：`quality`

---

## 6. 时序记忆：`InstanceBank.get / update / cache`

文件：

- [`projects/mmdet3d_plugin/models/instance_bank.py:83`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/instance_bank.py:83)

### 6.1 `get()`

看：

- [`instance_bank.py:83-145`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/instance_bank.py:83)

这个函数做三件事：

1. 复制 learnable anchors 和 learnable instance features
2. 如果有历史缓存，把历史 anchor 投影到当前帧坐标系
3. 计算当前帧和历史帧的 `time_interval`

最关键的阅读点：

- [`instance_bank.py:93-112`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/instance_bank.py:93)

这里的 `T_temp2cur` 很关键，它把上一帧坐标系下的实例映射到当前帧。

阅读提示：

- 这一步是 Sparse4D 时序传播的几何基础。
- 如果没有这一步，历史实例就只是“上一帧的框”，不能直接拿来当前帧使用。

### 6.2 `update()`

看：

- [`instance_bank.py:147-187`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/instance_bank.py:147)

这个函数发生在 decoder 中途。

它的意思是：

- 从当前帧候选实例里挑 top-k 高置信结果
- 再和缓存下来的历史实例拼起来
- 组成后续 decoder 继续使用的实例集合

最关键的点：

- `N = self.num_anchor - self.num_temp_instances`
- 说明总 anchor 槽位里，专门留出一部分给历史实例。

### 6.3 `cache()`

看：

- [`instance_bank.py:189-215`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/instance_bank.py:189)

它做的是：

- 从当前帧实例里选 top-k
- 保存成下一帧的 `cached_feature` 和 `cached_anchor`

阅读提示：

- `confidence_decay` 体现了历史实例置信度衰减。
- 这让历史信息有“记忆”，但不会无限强势。

### 6.4 `get_instance_id()`

看：

- [`instance_bank.py:217-254`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/instance_bank.py:217)

它的作用是：

- 给当前高置信实例分配稳定的 id
- 让模型输出 tracking 结果

阅读提示：

- 这里不是传统 MOT 的复杂关联器，而是基于时序缓存的轻量实例延续。

---

## 7. 视觉证据抽取：`DeformableFeatureAggregation.forward`

文件：

- [`projects/mmdet3d_plugin/models/blocks.py:111`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/blocks.py:111)

### 7.1 先抓一句话

这个模块的核心就是：

- “让每个 3D anchor 去多相机、多尺度特征图上，按一组关键点采样证据，再把这些证据融合回来。”

### 7.2 第 1 段：生成 key points 和融合权重

看：

- [`blocks.py:120-122`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/blocks.py:120)

阅读提示：

- `key_points = self.kps_generator(anchor, instance_feature)`：在每个 3D box 内部生成若干采样点。
- `weights = self._get_weights(...)`：预测这些采样点在不同相机、不同尺度上的融合权重。

这两句就是 deformable aggregation 的输入准备。

### 7.3 第 2 段：3D 点投影到 2D 图像

看：

- [`blocks.py:124-148`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/blocks.py:124)

阅读提示：

- `projection_mat` 负责把 3D 点投到各相机平面。
- reshape 之后的 `points_2d` 维度可以理解成：
  `(bs, num_anchor, num_pts, num_cams, 2)`

这说明模型已经把“3D anchor 内部点”变成了“各个相机图像上的采样位置”。

### 7.4 第 3 段：特征采样与融合

如果不开 CUDA op，会走：

- [`blocks.py:149-157`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/blocks.py:149)

理解顺序：

1. `feature_sampling(...)`：在多尺度特征图上采样
2. `multi_view_level_fusion(...)`：按 learned weights 融合 view / level
3. `sum(dim=2)`：把多个点的特征再聚合

### 7.5 第 4 段：残差输出

看：

- [`blocks.py:158-163`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/blocks.py:158)

阅读提示：

- `residual_mode == "cat"` 在当前 config 下很关键。
- 这意味着视觉证据不是简单加回原特征，而是和原实例特征拼接，留给后面的 FFN 继续整合。

### 7.6 `_get_weights()`

看：

- [`blocks.py:165-197`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/blocks.py:165)

这是另一个很值得读的函数。

理解重点：

- `instance_feature + anchor_embed`：融合当前实例语义和几何信息
- `camera_encoder(...)`：把相机参数也编码进去
- `softmax(dim=-2)`：对 view / level / point 相关维度做归一化，得到可解释的融合权重

这段非常像在回答一个问题：

- “这个 anchor 更应该信任哪个相机、哪个尺度、哪个关键点？”

---

## 8. 几何编码和 refine：`SparseBox3DEncoder / SparseBox3DRefinementModule`

文件：

- [`projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py:23`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py:23)

### 8.1 `SparseBox3DEncoder.forward`

看：

- [`detection3d_blocks.py:57-74`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py:57)

理解顺序：

1. 编码位置
2. 编码尺寸
3. 编码朝向
4. 编码速度
5. 按 `add` 或 `cat` 方式融合

阅读提示：

- 当前 config 里是 `mode="cat"`，所以更接近论文里的 decoupled attention。
- 也就是不同属性先分别保留，再在后面一起参与注意力交互。

### 8.2 `SparseBox3DRefinementModule.forward`

看：

- [`detection3d_blocks.py:123-156`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py:123)

逐步理解：

1. `feature = instance_feature + anchor_embed`
2. `output = self.layers(feature)`：预测新的状态增量
3. `output[..., self.refine_state] += anchor[..., self.refine_state]`
4. 如果有速度分量，再根据 `time_interval` 做速度修正
5. 输出 `cls`
6. 输出 `quality`

阅读提示：

- 这里不是从零预测 box，而是“在旧 anchor 基础上修正”。
- 这是 decoder 逐层 refinement 的核心。

最值得盯住的地方：

- [`detection3d_blocks.py:133-145`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py:133)

因为这里体现了：

- 中心、尺寸、朝向是迭代 refine 的
- 速度和时间间隔是耦合的

### 8.3 为什么 `quality` 很重要

看：

- [`detection3d_blocks.py:152-155`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py:152)

这几行告诉你：

- quality 不是后处理时临时拍脑袋算的
- 它和 box、cls 一样，是 decoder 正式输出的一部分

---

## 9. GT 编码与匹配：`SparseBox3DTarget`

文件：

- [`projects/mmdet3d_plugin/models/detection3d/target.py:48`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:48)

### 9.1 `encode_reg_target()`

看：

- [`target.py:48-64`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:48)

这一步非常基础但很重要。

它把 GT box 变成训练时统一使用的编码形式：

- `(x, y, z)`
- `log(w), log(l), log(h)`
- `sin(yaw), cos(yaw)`
- 其他速度等状态

阅读提示：

- 这解释了为什么 decoder 里最后要用 `exp()` 恢复尺度、用 `atan2()` 恢复 yaw。

### 9.2 `sample()`

看：

- [`target.py:66-119`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:66)

理解顺序：

1. 先算分类 cost
2. 再算 box cost
3. 用 Hungarian matching 做一对一匹配
4. 生成每个 query 对应的分类目标、box 目标、回归权重

阅读提示：

- `output_cls_target = ... * num_cls` 表示默认都先赋成背景类。
- 只有匹配上的 query 才会被填入 GT 类别和 GT box。

### 9.3 `_cls_cost()`

看：

- [`target.py:121-143`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:121)

这段本质上是按 focal loss 风格构造匹配代价。

阅读提示：

- 不是直接拿 `-score` 当 cost，而是更贴近训练损失形式。

### 9.4 `_box_cost()`

看：

- [`target.py:145-161`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:145)

理解重点：

- L1 距离
- 按维度加权
- 按类别还可附加不同回归权重

这体现了 3D 检测里不同状态维度的重要性不完全一样。

### 9.5 `get_dn_anchors()`

看：

- [`target.py:163-292`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:163)

这是理解 v3 `Temporal Instance Denoising` 的核心函数。

可以按下面顺序读：

#### 第 1 步：裁剪 GT 数量并对齐 batch 内形状

看：

- [`target.py:169-198`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:169)

含义：

- 每个 batch 里的 GT 数量不同，先 pad 到同样长度。

#### 第 2 步：复制 dn group

看：

- [`target.py:199-205`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:199)

含义：

- 同一批 GT 会复制成多个 denoising group，让训练更稳定。

#### 第 3 步：加噪

看：

- [`target.py:206-219`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:206)

含义：

- `dn_anchor = GT + noise`
- 如果 `add_neg_dn=True`，还会构造偏移更大的负样本 dn anchors

你可以理解成：

- 让模型学习“把被扰动的框拉回正确位置”

#### 第 4 步：给 noisy anchors 再分配回 GT

看：

- [`target.py:221-241`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:221)

含义：

- 即使是 dn anchors，也要明确哪个 noisy query 对哪个 GT 负责。

#### 第 5 步：重排形状并生成有效 mask、注意力 mask

看：

- [`target.py:242-291`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/target.py:242)

阅读提示：

- `valid_mask` 用来区分 pad 和有效 dn 条目。
- `attn_mask` 用来阻止不同 dn group 相互泄漏信息。

### 9.6 读这个类时最该盯住的变量

- `box_target`
- `cls_cost`
- `box_cost`
- `dn_anchor`
- `valid_mask`
- `attn_mask`

---

## 10. 损失闭环：`SparseBox3DLoss.forward`

文件：

- [`projects/mmdet3d_plugin/models/detection3d/losses.py:31`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/losses.py:31)

### 10.1 第 1 段：处理可反向类别的 yaw

看：

- [`losses.py:42-63`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/losses.py:42)

理解重点：

- 某些类别朝向正反无关，例如 `barrier`
- 所以如果预测和 GT 的 yaw 方向相反，但语义上等价，就把 GT yaw 翻过来

阅读提示：

- 这是典型的 3D 检测任务细节，论文里不一定会写这么细，但工程上很重要。

### 10.2 第 2 段：box loss

看：

- [`losses.py:65-69`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/losses.py:65)

这是最标准的主回归损失：

- 预测框 vs 编码后的 GT 框

### 10.3 第 3 段：quality loss

看：

- [`losses.py:71-91`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/losses.py:71)

这一段特别值得读，因为它完整展示了 `Quality Estimation`。

#### `centerness`

看：

- [`losses.py:72-79`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/losses.py:72)

含义：

- 预测中心越接近 GT，target 越接近 1
- 这里用 `exp(-distance)` 构造连续质量标签

#### `yawness`

看：

- [`losses.py:81-90`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/losses.py:81)

含义：

- 如果预测 yaw 和 GT yaw 方向一致，target 为 1
- 否则为 0

阅读提示：

- 这说明 quality 不是黑盒打分，而是拆成“中心质量”和“朝向质量”两部分监督。

---

## 11. 推理输出：`SparseBox3DDecoder.decode`

文件：

- [`projects/mmdet3d_plugin/models/detection3d/decoder.py:37`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/decoder.py:37)

### 11.1 第 1 段：拿最后一层 decoder 输出

看：

- [`decoder.py:45-57`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/decoder.py:45)

阅读提示：

- `cls_scores = cls_scores[output_idx].sigmoid()`
- 默认使用最后一层 decoder 结果

### 11.2 第 2 段：top-k 选候选框

看：

- [`decoder.py:53-61`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/decoder.py:53)

含义：

- 从所有 query、所有类别里选 top-k 候选输出。

### 11.3 第 3 段：用 quality 调整排序

看：

- [`decoder.py:63-73`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/decoder.py:63)

这是 quality estimation 在推理时真正发挥作用的地方。

理解顺序：

1. 取出 `centerness`
2. 保留原始分类分数 `cls_scores_origin`
3. 用 `cls_scores *= centerness.sigmoid()` 调整最终得分
4. 再重新排序

阅读提示：

- 这一步说明分类分数不再单独决定最终排序。
- 几何质量也开始影响输出优先级。

### 11.4 第 4 段：恢复 box 物理量

看：

- [`decoder.py:75-105`](/Users/aqiu/Documents/1_study/00_AllMyXX/AllMyDL/Sparse4D/projects/mmdet3d_plugin/models/detection3d/decoder.py:75)

理解重点：

- `decode_box()` 会把：
  - `log(w), log(l), log(h)` 还原成真实尺度
  - `sin/cos yaw` 还原成真实角度

如果有 `instance_id`，这里还会一起输出 tracking id。

---

## 12. 读完这些函数后，你应该建立的心智模型

如果你已经顺着这份文档看完了上面的关键函数，建议你把 Sparse4D 在脑中压缩成下面这句话：

“Sparse4D 维护一组稀疏 3D 实例，每一帧先取出当前实例和历史实例，再让这些实例一边彼此做时序和空间交互，一边去多相机图像特征里采样证据，接着逐层 refine 出更好的 box、分类和质量分数；训练时再通过 Hungarian matching 和 denoising 稳定优化，推理时用 quality 重排候选框，并用 instance bank 延续 tracking id。”

## 13. 下一步你最值得继续深挖的点

如果你准备继续往下读，我建议按这个顺序继续深入：

1. 只追一条 `quality` 线：
   `RefinementModule -> SparseBox3DLoss -> SparseBox3DDecoder`
2. 再只追一条 `temporal` 线：
   `InstanceBank.get -> temp_gnn -> cache -> get_instance_id`
3. 最后再细抠 `dn` 线：
   `get_dn_anchors -> head.forward 中的拼接 -> prepare_for_dn_loss`

这样每次只追一条线，不容易被大段 tensor reshape 吓住。

