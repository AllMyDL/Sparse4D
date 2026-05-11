/*
 * C++ 桥接层 - PyTorch 与 CUDA 内核的接口
 * ======================================
 * 该文件定义了 PyTorch 张量与 CUDA 内核之间的接口
 * 负责从 PyTorch 张量提取原始数据指针，调用 CUDA 内核计算，返回结果
 */

#include <c10/cuda/CUDAGuard.h>  // CUDA 设备管理
#include <torch/extension.h>     // PyTorch C++ 扩展 API

/*
 * CUDA 内核前向函数声明
 * 在 .cu 文件中实现，这里仅声明
 * 参数说明见下方的包装函数
 */
void deformable_aggregation(
    float*       output,             // 输出特征 [bs, num_anchors, num_embeds]
    const float* mc_ms_feat,         // 多相机多尺度特征 [bs, num_feat, num_embeds]
    const int*   spatial_shape,      // 各尺度空间尺寸 [num_cams, num_scales, 2]
    const int*   scale_start_index,  // 尺度起始索引 [num_cams, num_scales]
    const float* sample_location,    // 采样点坐标 [bs, num_anchors, num_pts, num_cams, 2]
    const float* weights,            // 采样权重 [bs, num_anchors, num_pts, num_cams, num_scales, num_groups]
    int          batch_size,         // 批次大小
    int          num_cams,           // 相机数量
    int          num_feat,           // 压平后的特征数
    int          num_embeds,         // 特征通道数
    int          num_scale,          // 特征尺度数
    int          num_anchors,        // 锚点数量
    int          num_pts,            // 每个锚点的采样点数
    int          num_groups          // 权重分组数
);

/* 张量形状说明 */
/* feat: [bs, num_feat, channels] */
/* spatial_shape: [num_cams, num_scales, 2] */
/* scale_start_index: [num_cams, num_scales] */
/* sampling_location: [bs, num_anchors, num_pts, num_cams, 2] */
/* weights: [bs, num_anchors, num_pts, num_cams, num_scales, num_groups] */
/* output: [bs, num_anchors, channels] */

/*
 * 前向传播包装函数 - PyTorch 自动求导调用的入口
 *
 * 功能：
 * 1. 从 PyTorch 张量中提取原始数据指针
 * 2. 提取张量的维度参数
 * 3. 调用 CUDA 内核执行计算
 * 4. 返回计算结果给 PyTorch
 *
 * 参数: 都是 PyTorch 张量（ATensor），需要转换为原始指针
 * 返回: 聚合后的特征张量
 */
at::Tensor deformable_aggregation_forward(const at::Tensor& _mc_ms_feat,         // 输入特征张量
                                          const at::Tensor& _spatial_shape,      // 空间形状张量
                                          const at::Tensor& _scale_start_index,  // 尺度索引张量
                                          const at::Tensor& _sampling_location,  // 采样位置张量
                                          const at::Tensor& _weights             // 权重张量
)
{
    // 设置 CUDA 设备上下文，确保后续操作在正确的 GPU 上执行
    at::DeviceGuard                   guard(_mc_ms_feat.device());
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));

    // 从张量维度中提取问题尺寸参数
    int batch_size  = _mc_ms_feat.size(0);         // 批次大小
    int num_feat    = _mc_ms_feat.size(1);         // 特征总数
    int num_embeds  = _mc_ms_feat.size(2);         // 特征通道数
    int num_cams    = _spatial_shape.size(0);      // 相机数
    int num_scale   = _spatial_shape.size(1);      // 尺度数
    int num_anchors = _sampling_location.size(1);  // 锚点数
    int num_pts     = _sampling_location.size(2);  // 采样点数
    int num_groups  = _weights.size(5);            // 权重分组数

    // 从 PyTorch 张量中提取原始数据指针（指向 GPU 内存）
    const float* mc_ms_feat        = _mc_ms_feat.data_ptr<float>();
    const int*   spatial_shape     = _spatial_shape.data_ptr<int>();
    const int*   scale_start_index = _scale_start_index.data_ptr<int>();
    const float* sampling_location = _sampling_location.data_ptr<float>();
    const float* weights           = _weights.data_ptr<float>();

    // 创建输出张量，初始化为零
    // 使用与输入相同的 device 和 dtype（数据类型）
    auto output = at::zeros({batch_size, num_anchors, num_embeds}, _mc_ms_feat.options());

    // 调用 CUDA 内核执行前向计算
    deformable_aggregation(output.data_ptr<float>(), mc_ms_feat, spatial_shape, scale_start_index, sampling_location,
                           weights, batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts,
                           num_groups);

    // 返回计算结果给 PyTorch
    return output;
}

/*
 * CUDA 内核反向函数声明
 * 在 .cu 文件中实现，用于计算梯度
 */
void deformable_aggregation_grad(const float* mc_ms_feat,              // 原始特征
                                 const int*   spatial_shape,           // 空间形状
                                 const int*   scale_start_index,       // 尺度索引
                                 const float* sample_location,         // 采样位置
                                 const float* weights,                 // 权重
                                 const float* grad_output,             // 输出的梯度
                                 float*       grad_mc_ms_feat,         // 特征梯度（输出）
                                 float*       grad_sampling_location,  // 采样位置梯度（输出）
                                 float*       grad_weights,            // 权重梯度（输出）
                                 int          batch_size,
                                 int          num_cams,
                                 int          num_feat,
                                 int          num_embeds,
                                 int          num_scale,
                                 int          num_anchors,
                                 int          num_pts,
                                 int          num_groups);

/*
 * 反向传播包装函数 - 计算梯度
 *
 * 功能：
 * 1. 从 PyTorch 张量提取数据指针
 * 2. 调用 CUDA 反向内核计算梯度
 * 3. 结果直接写入梯度张量中
 */
void deformable_aggregation_backward(const at::Tensor& _mc_ms_feat,         // 原始特征
                                     const at::Tensor& _spatial_shape,      // 空间形状
                                     const at::Tensor& _scale_start_index,  // 尺度索引
                                     const at::Tensor& _sampling_location,  // 采样位置
                                     const at::Tensor& _weights,            // 权重
                                     const at::Tensor& _grad_output,      // 输出梯度（损失对输出的导数）
                                     at::Tensor&       _grad_mc_ms_feat,  // 特征梯度（输出参数）
                                     at::Tensor&       _grad_sampling_location,  // 采样位置梯度（输出参数）
                                     at::Tensor&       _grad_weights             // 权重梯度（输出参数）
)
{
  // 设置 CUDA 设备上下文
  at::DeviceGuard                   guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));

  // 提取问题尺寸参数
  int batch_size  = _mc_ms_feat.size(0);
  int num_feat    = _mc_ms_feat.size(1);
  int num_embeds  = _mc_ms_feat.size(2);
  int num_cams    = _spatial_shape.size(0);
  int num_scale   = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts     = _sampling_location.size(2);
  int num_groups  = _weights.size(5);

  // 提取前向传播数据的原始指针
  const float* mc_ms_feat        = _mc_ms_feat.data_ptr<float>();
  const int*   spatial_shape     = _spatial_shape.data_ptr<int>();
  const int*   scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights           = _weights.data_ptr<float>();
  const float* grad_output       = _grad_output.data_ptr<float>();

  // 提取梯度张量的原始指针（这些是输出参数，会被修改）
  float* grad_mc_ms_feat        = _grad_mc_ms_feat.data_ptr<float>();
  float* grad_sampling_location = _grad_sampling_location.data_ptr<float>();
  float* grad_weights           = _grad_weights.data_ptr<float>();

  // 调用 CUDA 内核计算反向传播梯度
  deformable_aggregation_grad(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, grad_output,
                              grad_mc_ms_feat, grad_sampling_location, grad_weights, batch_size, num_cams, num_feat,
                              num_embeds, num_scale, num_anchors, num_pts, num_groups);
}

/*
 * Python 绑定 - 使用 pybind11 将 C++ 函数暴露给 Python
 * 允许 Python 代码调用这些 C++ 函数
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 注册前向函数，Python 可以通过 deformable_aggregation_ext.deformable_aggregation_forward 调用
  m.def(
    "deformable_aggregation_forward",
    &deformable_aggregation_forward,
    "deformable_aggregation_forward"
  );

  // 注册反向函数，Python 可以通过 deformable_aggregation_ext.deformable_aggregation_backward 调用
  m.def(
    "deformable_aggregation_backward",
    &deformable_aggregation_backward,
    "deformable_aggregation_backward"
  );
}
