# tools 目录导读

这份文档面向刚开始接触 Sparse4D 代码库的同学，帮助你快速理解 `tools/` 目录里每个脚本是做什么的、它们之间是什么关系，以及建议先看哪些文件。

## 1. `tools/` 目录整体是干什么的

可以把 `tools/` 理解成这个项目的“操作台”。

- 模型主体通常定义在 `projects/` 中。
- `tools/` 不负责发明模型结构，而是负责把训练、测试、数据预处理、性能测试这些流程串起来。
- 你平时在命令行里运行的很多入口脚本，基本都在这个目录里。

如果把整个项目类比成一个完整系统：

- `projects/` 更像“发动机和机械结构”
- `tools/` 更像“点火、驾驶、检测和维护面板”

---

## 2. 一个新手最容易理解的整体流程

在这个项目里，一个比较常见的使用顺序如下：

1. 先把 nuScenes 原始数据转换成项目需要的 `info pkl` 文件。
2. 如果模型需要 anchor 先验，再从标注中聚类生成 anchor。
3. 用训练脚本启动模型训练。
4. 训练完成后，用测试脚本评估模型效果。
5. 如果关心推理速度，再做 benchmark。
6. 如果准备部署或想进一步加速推理，可以做 Conv-BN 融合。

对应到 `tools/` 下的文件，大致关系是：

1. `nuscenes_converter.py`
2. `anchor_generator.py`
3. `train.py` 或 `dist_train.sh`
4. `test.py` 或 `dist_test.sh`
5. `benchmark.py`
6. `fuse_conv_bn.py`

---

## 3. 每个脚本分别做什么

### `nuscenes_converter.py`

这是数据预处理脚本。

它的核心作用是：把 nuScenes 官方原始格式，转换成当前项目训练和测试时更容易直接读取的 `pkl info` 文件。

它主要做这些事情：

- 读取 nuScenes 数据集和官方划分
- 过滤掉本地并不存在的数据场景
- 遍历每个 sample
- 记录激光雷达路径
- 记录 6 个相机的信息
- 收集历史 sweeps
- 计算不同传感器到顶置 LiDAR 的坐标变换
- 在非 test 模式下整理 GT 框、类别、速度、点数等标注

你可以把它理解成：

“把官方原始数据，整理成训练代码真正吃得下去的结构化输入。”

如果你是新手，这个文件很值得认真看，因为：

- 3D 检测非常依赖坐标系理解
- 多传感器任务非常依赖外参和时间序列组织方式
- 这个文件能帮你知道训练数据最终长什么样

### `anchor_generator.py`

这是 anchor 生成脚本。

它的作用是：从标注文件里取出所有真实目标框，然后对目标中心位置做 KMeans 聚类，生成一组 anchor 的空间先验。

它主要做这些事情：

- 读取 `ann_file` 对应的 `pkl` 标注文件
- 收集所有样本中的 `gt_boxes`
- 按距离过滤掉过远目标
- 对目标中心点的 `x/y/z` 位置做聚类
- 用聚类中心生成 anchor 的位置先验
- 用全部 GT 的平均尺寸生成 anchor 的尺寸先验

你可以把它理解成：

“让模型一开始就从一些更合理的位置去猜目标，而不是完全随机初始化。”

如果你刚接触 anchor-based 思想，这个脚本能帮助你建立一个朴素直觉：

- anchor 不是凭空来的
- 它往往来自数据分布统计

### `train.py`

这是训练入口脚本，也是 `tools/` 里最关键的文件之一。

它本身不定义模型细节，而是负责把配置、模型、数据集、日志、分布式环境这些东西拼起来，然后正式启动训练。

它主要做这些事情：

- 解析命令行参数
- 读取配置文件
- 支持命令行覆盖配置项
- 导入自定义 plugin 模块
- 配置工作目录、日志、随机种子
- 初始化单机或分布式训练环境
- 构建模型
- 构建训练集和验证集
- 调用训练流程开始训练

你可以这样理解它：

“`train.py` 不是模型本身，而是整个训练系统的启动器。”

对新手来说，这个文件非常重要，因为它能帮你建立整张“项目运行总图”：

- 配置文件是怎么进来的
- 模型是在哪里被 build 的
- 数据集是在哪里被 build 的
- 训练是从哪一行真正开始的

### `dist_train.sh`

这是多卡训练的 shell 启动脚本。

它本质上是对 `train.py` 的一个封装，用来更方便地启动分布式训练。

它主要做这些事情：

- 接收配置文件路径
- 接收 GPU 数量
- 设置分布式通信端口
- 把仓库根目录加入 `PYTHONPATH`
- 使用 `torch.distributed.launch` 启动 `train.py`

你可以把它理解成：

- `train.py` 是训练发动机
- `dist_train.sh` 是多卡启动按钮

平时如果你要用多张 GPU 训练，通常就是运行这个脚本。

### `test.py`

这是测试和评估入口脚本。

它负责在给定配置文件和 checkpoint 的情况下，完成模型推理、结果保存、结果评估、结果格式化和可视化。

它主要做这些事情：

- 读取配置和权重文件
- 构建测试数据集和 dataloader
- 构建模型并加载 checkpoint
- 在单卡或多卡环境中执行推理
- 保存推理结果
- 调用 `dataset.evaluate()` 计算指标
- 或者调用 `format_results()` 生成提交格式
- 或者调用 `show()` 做结果可视化

你可以把它理解成：

“训练完以后，用它来回答模型到底效果怎么样。”

这个文件很适合和 `train.py` 对照着看，因为它能让你同时理解：

- 训练时系统怎么组织
- 测试时系统怎么组织

### `dist_test.sh`

这是多卡测试的 shell 启动脚本。

它和 `dist_train.sh` 的关系类似，只不过这里封装的是 `test.py`。

它主要做这些事情：

- 接收配置文件
- 接收 checkpoint 路径
- 接收 GPU 数量
- 设置分布式测试端口
- 启动 `test.py`

你可以把它理解成：

“让测试也能在多张 GPU 上并行跑起来。”

当测试集比较大、单卡推理太慢时，这个脚本就很有用。

### `benchmark.py`

这是推理性能测试脚本。

它的关注重点不是精度，而是速度和资源占用。

它主要会统计：

- 推理吞吐量，也就是 `FPS`
- 推理过程中 GPU 峰值显存

它主要做这些事情：

- 读取配置文件
- 构建测试集和模型
- 加载 checkpoint
- 多次执行前向推理
- 跳过前几次 warmup
- 统计平均推理时间和显存占用

你可以把它理解成：

“不问模型准不准，只问它跑得快不快。”

这个脚本适合在这些场景使用：

- 比较不同模型版本的推理速度
- 比较是否 fuse Conv-BN 前后的性能差异
- 做部署前的速度摸底

### `fuse_conv_bn.py`

这是推理优化脚本。

它的作用是把卷积层 `Conv` 和它后面的归一化层 `BN` 融合起来，生成一个更适合推理的模型。

这样做的常见收益是：

- 减少推理时的计算开销
- 简化网络结构
- 略微提升推理速度

它主要做这些事情：

- 读取配置文件和 checkpoint
- 构建模型
- 遍历模型结构
- 查找相邻的 Conv 和 BN
- 把 BN 参数折叠进 Conv
- 导出融合后的新模型

你可以把它理解成：

“训练时保留 BN 方便优化，推理时把 BN 折叠掉来提升效率。”

它通常不是训练阶段使用的，而是更偏部署前优化。

---

## 4. 这些文件之间的关系

如果只从“工作流”角度去看，它们之间的关系可以简单记成下面这样：

`nuscenes_converter.py`
-> 生成训练/测试要用的 `info pkl`

`anchor_generator.py`
-> 从标注统计 anchor 先验

`train.py` / `dist_train.sh`
-> 启动训练

`test.py` / `dist_test.sh`
-> 加载训练好的模型并评估

`benchmark.py`
-> 评估推理速度

`fuse_conv_bn.py`
-> 生成更适合推理部署的模型

所以它们并不是彼此独立的一堆零散工具，而是围绕“数据准备 -> 训练 -> 测试 -> 加速”这条主线组织起来的。

---

## 5. 新手建议先看哪几个

如果你现在的目标是“真正开始读懂 Sparse4D 代码”，推荐按这个顺序看：

### 第一优先级

`train.py`

原因：

- 它最能帮助你建立全局视角
- 你会知道项目是怎样从一个 config 真正跑起来的

### 第二优先级

`test.py`

原因：

- 你会看到模型推理和评估是怎么接起来的
- 方便和 `train.py` 形成对照

### 第三优先级

`nuscenes_converter.py`

原因：

- 数据和坐标系是 3D 检测理解的基础
- 这个文件能帮你看清楚训练输入是怎么准备出来的

### 第四优先级

`anchor_generator.py`

原因：

- 帮你理解 anchor 先验来自哪里
- 帮你建立“模型先验和数据分布相关”的直觉

### 第五优先级

`benchmark.py` 和 `fuse_conv_bn.py`

原因：

- 它们更偏工程优化
- 对“先跑通和先读懂主体逻辑”不是最前置

---

## 6. 如果你只想先记住一句话

`tools/` 目录不是“模型定义区”，而是“项目运行入口区”。

- `nuscenes_converter.py`：整理数据
- `anchor_generator.py`：生成 anchor 先验
- `train.py`：启动训练
- `test.py`：启动测试和评估
- `dist_train.sh` / `dist_test.sh`：多卡启动器
- `benchmark.py`：测试推理速度
- `fuse_conv_bn.py`：做推理优化

---

## 7. 推荐的下一步

如果你已经看完 `tools/`，下一步最推荐的是继续顺着下面这条链路往里读：

`train.py`
-> 配置文件 `configs/...`
-> `projects/mmdet3d_plugin/...`
-> 模型、数据集、head、loss、训练流程

这条链路最适合新手从“会运行脚本”过渡到“真正理解模型实现”。

---

## 8. 如何使用 `custom_converter.py`

如果你的本地数据不是 nuScenes，而是你自己的多相机 + LiDAR + pose + 标定数据，推荐的做法不是直接大改数据集类，而是先把原始数据转换成 Sparse4D 当前实现已经能读取的 `infos.pkl` 格式。

### 相关文件

仓库里现在已经提供了下面几个辅助文件：

- `tools/custom_converter.py`
- `tools/custom_converter_example.json`
- `projects/configs/custom_sparse4d_temporal_r50_1x8_bs6_256x704.py`

它们的分工分别是：

- `custom_converter.py`：把你的原始数据转换成 Sparse4D 需要的 `infos.pkl`
- `custom_converter_example.json`：给你一个最小可参考的原始标注中间格式示例
- `custom_sparse4d_temporal_r50_1x8_bs6_256x704.py`：给你一份可改的自定义数据配置模板

### 推荐工作流

建议按下面这个顺序做：

1. 先把你的原始标注整理成类似 `custom_converter_example.json` 的结构
2. 用 `custom_converter.py` 转成 `custom_infos_train.pkl` / `custom_infos_val.pkl`
3. 修改自定义配置模板中的路径、类别数、相机数、图像尺寸、anchor 路径
4. 先做单样本投影检查，确认几何关系没有错
5. 再开始训练

### `custom_converter_example.json` 里最重要的字段

每个样本通常至少需要包含：

- `token`
- `timestamp`
- `lidar_path`
- `lidar2ego_rotation_matrix`
- `lidar2ego_translation`
- `ego2global_rotation_matrix`
- `ego2global_translation`
- `cameras`

如果你要训练，还要有：

- `anns`

每个相机通常至少需要包含：

- `name`
- `image_path`
- `cam_intrinsic`
- `cam_to_lidar_rotation` 和 `cam_to_lidar_translation`

每个标注框通常至少需要包含：

- `name`
- `center_global` 或 `center_lidar`
- `size`
- `yaw`

可选字段包括：

- `velocity_global` 或 `velocity_lidar`
- `num_lidar_pts`
- `num_radar_pts`
- `instance_id`

### 最简单的转换命令

假设你的数据根目录是 `data/custom/`，原始中间标注文件是 `tools/custom_converter_example.json`，那么可以像这样先生成一个测试版 `pkl`：

```bash
python tools/custom_converter.py \
  --root-path data/custom \
  --ann-path tools/custom_converter_example.json \
  --out-path data/custom_infos/custom_infos_train.pkl \
  --version custom
```

如果你是在生成测试集 `pkl`，不希望写入 GT，可以加上：

```bash
python tools/custom_converter.py \
  --root-path data/custom \
  --ann-path your_test_annotations.json \
  --out-path data/custom_infos/custom_infos_test.pkl \
  --version custom \
  --test-mode
```

### 你最可能需要改的函数

`custom_converter.py` 里最值得优先改的是这几个函数：

- `load_raw_samples()`
- `build_one_info()`
- `convert_boxes_to_lidar()`
- `build_camera_info()`

原因分别是：

- `load_raw_samples()`：决定你怎么把原始 JSON/CSV/数据库读进来
- `build_one_info()`：决定单帧最终长什么样
- `convert_boxes_to_lidar()`：决定你的框、速度、类别怎么映射进训练格式
- `build_camera_info()`：决定你的相机标定怎么映射到 `sensor2lidar_*`

### 最容易出错的地方

新手最常在下面几个地方出错：

- 3D 框坐标系不是 LiDAR 坐标系
- `yaw` 定义和当前 LiDAR 坐标轴方向不一致
- `cam_to_lidar` 和 `lidar_to_cam` 弄反
- 图像内参没和分辨率对应上
- 多相机顺序训练时不一致
- 时间戳没有按时间递增排序

所以最稳妥的调试方式是：

1. 先只取 1 帧
2. 先只取 1 个相机
3. 检查 LiDAR 点能不能正确投到图像上
4. 检查 GT 3D 框投影后是不是落在目标位置附近
5. 确认无误后再扩大到全量数据

### 配置模板里最优先要改的部分

`projects/configs/custom_sparse4d_temporal_r50_1x8_bs6_256x704.py` 里，最优先要改的是：

- `data_root`
- `anno_root`
- `class_names`
- `num_classes`
- `num_cams`
- `custom_anchor`
- `custom_image_h`
- `custom_image_w`

其中：

- 如果你的类别和 nuScenes 不同，必须同步修改 `class_names`
- 如果你的相机不是 6 个，必须同步修改 `num_cams`
- 如果你重新做了 anchor 聚类，必须改 `custom_anchor`
- 如果你图像原始分辨率不是 1600x900，必须改 `custom_image_h/custom_image_w`

### 一个很实际的建议

先不要一开始就追求“完全适配时序、多相机、全类别、全评估”。

更推荐的推进顺序是：

1. 先做单帧版本
2. 先确认投影几何完全正确
3. 再补 sweeps
4. 再补 instance id
5. 最后再补 tracking 和完整评估

这样排查问题会轻松很多。
