# datasets 目录导读

这份文档面向刚开始阅读 Sparse4D 数据部分实现的同学，帮助你快速建立一个整体图：

- 数据是从哪里读进来的
- 图像和点云做了哪些处理
- 多相机投影矩阵是怎么准备的
- sampler 为什么这么设计
- 每个脚本分别负责哪一层

---

## 1. `datasets/` 目录整体是干什么的

如果说：

- `tools/train.py` 是训练启动器
- `tools/test.py` 是测试启动器

那么 `projects/mmdet3d_plugin/datasets/` 就是“数据进入模型之前的总装线”。

它主要负责四件事：

1. 定义数据集类
2. 从磁盘读取多相机图像和点云
3. 做图像/3D 框/投影矩阵的同步增强
4. 决定训练时样本如何组成 batch

对新手来说，这个目录很重要，因为你看懂这里之后，基本就能回答下面这些问题：

- 一帧样本最后长什么样
- 多相机图像是怎么组织成张量的
- LiDAR 到图像的投影矩阵在哪里算
- 深度监督是怎么从点云生成的
- 时序训练为什么要用特殊 sampler

---

## 2. 最推荐的阅读顺序

建议按这条顺序看：

1. `nuscenes_3d_det_track_dataset.py`
2. `pipelines/loading.py`
3. `pipelines/augment.py`
4. `pipelines/transform.py`
5. `builder.py`
6. `samplers/`
7. `utils.py`

原因很简单：

- 先看主数据集类，知道“总流程”
- 再看 pipeline，知道“样本怎么一步步加工”
- 再看 sampler，知道“batch 为什么这样取”
- 最后看 utils，知道“可视化和几何辅助工具”

---

## 3. 主文件：`nuscenes_3d_det_track_dataset.py`

这是整个目录里最核心的文件。

你可以把它理解成：

“把 `infos.pkl` 里的单帧记录，组织成 Sparse4D 真正能吃的训练/测试样本。”

它主要做这些事：

- 从 `ann_file` 读取预处理好的 `infos.pkl`
- 按时间戳排序样本
- 根据 `cams`、`lidar2ego`、`ego2global` 等字段拼出输入信息
- 计算每个相机的 `lidar2img`
- 读取 GT 框、GT 类别、GT 速度、instance id
- 在评估时把模型输出转成 nuScenes 官方格式
- 在可视化时把预测框画到图像和 BEV 上

这个文件最值得你盯住的几个函数是：

### `load_annotations()`

作用：

- 读取 `infos.pkl`
- 按时间排序
- 做抽帧

意义：

- 时序模型非常依赖时间顺序，这一步是整个时间线的入口

### `get_data_info()`

作用：

- 把单帧 `info` 转成 pipeline 要吃的 `input_dict`

这是最关键的桥梁函数。

它会准备：

- `pts_filename`
- `img_filename`
- `lidar2img`
- `cam_intrinsic`
- `lidar2global`
- `timestamp`

如果你在做自定义数据适配，这个函数基本就是“你最终要对齐的目标格式”。

### `get_ann_info()`

作用：

- 从 `info` 中提取 GT 监督

它会整理出：

- `gt_bboxes_3d`
- `gt_labels_3d`
- `gt_names`
- `instance_inds`

如果开启 `with_velocity`，还会把速度拼到 3D box 后面。

### `evaluate()`

作用：

- 把模型预测导出
- 调 nuScenes 官方评测
- 汇总 detection / tracking 指标

所以这个文件不仅是“训练输入组织器”，也是“评估出口”。

---

## 4. `pipelines/` 目录是干什么的

`pipelines/` 可以理解成“样本加工流水线”。

主数据集类只是先准备一个原始 `input_dict`，真正把它变成模型输入的是 pipeline。

### `pipelines/loading.py`

这个文件负责“从磁盘读取原始数据”。

包含两个核心类：

#### `LoadMultiViewImageFromFiles`

作用：

- 读取多相机图像
- 把多视角图像堆起来
- 再拆成 list，方便后续逐相机增强

你可以把它理解成：

“先把多目图片读进来，再变成适合 pipeline 逐视角处理的格式。”

#### `LoadPointsFromFile`

作用：

- 读取 LiDAR 点云文件
- 按 `load_dim` 和 `use_dim` 截取需要的特征维度

在这个 Sparse4D 实现里，点云并不是直接送进主干网络的。
它更重要的用途是：

- 为多目图像生成辅助深度监督

这是一个很关键的认识点。

### `pipelines/augment.py`

这个文件负责“数据增强”。

它的最大特点是：

- 不只是改图像
- 还会同步修改投影矩阵和 3D 框

这是多传感器 3D 任务里最容易出错的地方。

包含三个核心类：

#### `ResizeCropFlipImage`

作用：

- 对每个相机图像做 resize / crop / flip / rotate
- 同时左乘更新 `lidar2img`
- 必要时同步调整内参

你可以把它理解成：

“图像平面怎么变，投影矩阵也必须怎么变。”

#### `BBoxRotation`

作用：

- 在 LiDAR 坐标系下对整帧做 3D 旋转增强
- 同步更新 `lidar2img`
- 同步旋转 GT 框和速度

你可以把它理解成：

“不是只转框，而是整个 3D 世界一起转。”

#### `PhotoMetricDistortionMultiViewImage`

作用：

- 做亮度、对比度、饱和度、色调等颜色增强

特点：

- 它只改颜色，不改几何
- 所以不需要改投影矩阵

### `pipelines/transform.py`

这个文件负责“把原始数据整理成 Sparse4D 直接要用的张量和字段”。

包含几个非常关键的模块：

#### `MultiScaleDepthMapGenerator`

作用：

- 把 LiDAR 点投到每个相机图像上
- 生成多尺度深度监督图 `gt_depth`

这一步非常重要，因为它解释了：

- 为什么训练 pipeline 明明是相机模型，还要加载点云

答案就是：

- 点云在这里主要是给图像分支提供辅助深度监督

#### `NuScenesSparse4DAdaptor`

作用：

- 把通用字段整理成 Sparse4D 直接需要的格式

比如：

- `projection_mat`
- `image_wh`
- `T_global`
- `T_global_inv`
- `focal`
- `img`

这是“通用 dataset 字段”到“模型前向输入字段”的最后一道桥。

#### `InstanceNameFilter`

作用：

- 只保留当前实验定义的类别

#### `CircleObjectRangeFilter`

作用：

- 按类别设置不同的感兴趣半径
- 过滤过远目标

#### `NormalizeMultiviewImage`

作用：

- 对多视角图像做标准化

---

## 5. `builder.py` 是干什么的

这个文件负责两件事：

1. 构建 DataLoader
2. 构建 Dataset

### `build_dataloader()`

作用：

- 根据训练方式选择不同 sampler
- 设置 batch size、worker 数、collate 函数
- 最终生成 PyTorch DataLoader

它最重要的价值是：

- 把“普通训练 / 分布式训练 / IterBasedRunner”几种情况统一起来

### `custom_build_dataset()`

作用：

- 根据配置递归构建数据集
- 支持 `ConcatDataset`、`RepeatDataset`、`ClassBalancedDataset` 等包装器

你可以把它理解成：

“配置文件写一个 dataset 结构，这里把它真正变成 Python 对象。”

---

## 6. `samplers/` 目录是干什么的

这是新手最容易跳过，但时序模型里其实非常关键的一层。

sampler 决定的不是“一个样本长什么样”，而是：

- 每一步训练到底拿哪些样本
- 它们之间的顺序是什么
- 多卡之间如何切分

### `samplers/group_sampler.py`

核心类：

- `DistributedGroupSampler`

作用：

- 把同 group 的样本放在一起
- 让每张卡拿到完整 batch
- 再按 rank 切给不同 GPU

适合普通分布式训练。

### `samplers/distributed_sampler.py`

核心类：

- `DistributedSampler`

作用：

- 分布式测试时按顺序切分样本
- 尽量以完整序列为单位给不同 rank 分配数据

这个实现里它特别照顾时序场景，不只是简单均分下标。

### `samplers/group_in_batch_sampler.py`

这是最值得注意的 sampler。

核心类：

- `GroupInBatchSampler`

作用：

- 给一个 global batch 中的每个“槽位”分配一条独立序列
- 每个槽位沿着自己的序列往前走
- 同一序列可共享增强参数

你可以把它理解成：

“一个 batch 不是简单随机抽几帧，而是并行维护多条时间序列。”

这对时序模型很重要，因为它能让：

- 时间邻近关系保留下来
- batch 内又能覆盖多条不同序列

### `samplers/sampler.py`

作用：

- 定义 sampler registry
- 提供 `build_sampler()`

它本身逻辑很简单，但属于 glue code，负责把配置和具体 sampler 类接起来。

---

## 7. `utils.py` 是干什么的

这个文件主要是可视化和几何辅助函数。

它不直接参与训练主流程，但非常适合新手理解几何。

里面比较重要的函数有：

### `box3d_to_corners()`

作用：

- 把 3D box 的中心、尺寸、yaw 变成 8 个角点

这是 3D 框可视化的基础。

### `draw_lidar_bbox3d_on_img()`

作用：

- 把 LiDAR 坐标系里的 3D 框投到图像上并画出来

这在调试自定义数据时特别有用，因为它能直接帮你检查：

- 标定对不对
- 坐标系对不对
- `lidar2img` 对不对

### `draw_lidar_bbox3d_on_bev()`

作用：

- 在鸟瞰图上画 3D 框

### `draw_lidar_bbox3d()`

作用：

- 同时生成“多视角图像 + BEV”的总览图

很适合做 demo 和调试截图。

---

## 8. 从“输入文件”到“模型输入”的完整链路

如果你想只记住一条主线，最推荐记下面这条：

1. `infos.pkl` 被 `NuScenes3DDetTrackDataset.load_annotations()` 读进来
2. `get_data_info()` 把单帧信息整理成 `input_dict`
3. `LoadMultiViewImageFromFiles` / `LoadPointsFromFile` 读取图像和点云
4. `ResizeCropFlipImage` / `BBoxRotation` / `PhotoMetricDistortion...` 做增强
5. `MultiScaleDepthMapGenerator` 生成深度监督
6. `NuScenesSparse4DAdaptor` 整理成模型直接要用的字段
7. `Collect` 把最终张量和 meta 打包
8. `build_dataloader()` + sampler 决定 batch 怎么组成

这条链一旦看顺了，后面理解自定义数据适配会轻松很多。

---

## 9. 如果你正在做自定义数据适配，最该看哪几个点

如果你的目标是“把自己的多相机 + LiDAR 数据接进 Sparse4D”，最该盯住的是：

1. `nuscenes_3d_det_track_dataset.py` 里的 `get_data_info()`
2. `pipelines/loading.py`
3. `pipelines/augment.py`
4. `pipelines/transform.py` 里的 `NuScenesSparse4DAdaptor`
5. `utils.py` 里的投影可视化函数

原因是：

- `get_data_info()` 定义了最终要准备哪些字段
- `loading.py` 定义了图像/点云文件怎么读
- `augment.py` 定义了增强后哪些几何量必须同步更新
- `NuScenesSparse4DAdaptor` 定义了模型真正要吃的输入格式
- `utils.py` 能帮你快速验证标定和投影有没有错

---

## 10. 一句话总结

`datasets/` 目录本质上就是三层：

- `nuscenes_3d_det_track_dataset.py`：定义“一个样本长什么样”
- `pipelines/`：定义“样本怎么被加工成模型输入”
- `samplers/`：定义“训练时一批样本怎么被选出来”

而 `builder.py` 和 `utils.py` 则分别负责：

- 把这些模块拼起来
- 提供调试和可视化辅助能力

如果你愿意，我下一步可以继续把这份导读再往前推进一层，专门给你写一版：

`infos.pkl -> get_data_info() -> pipeline -> Collect -> Sparse4D.forward()`

也就是“数据真正进入模型前向”的完整调用链导读。
