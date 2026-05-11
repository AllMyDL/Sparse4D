/*
 * CUDA 核心计算内核
 * ==================
 * 该文件在 GPU 上并行执行可变形特征聚合的计算
 * 使用双线性插值进行特征采样和聚合
 */

#include <ATen/ATen.h>              // PyTorch ATen 库
#include <ATen/cuda/CUDAContext.h>  // CUDA 上下文管理
#include <THC/THCAtomics.cuh>       // CUDA 原子操作
#include <cuda.h>                   // CUDA API
#include <cuda_runtime.h>           // CUDA 运行时

#include <iostream>

#include <stdlib.h>

/*
 * 双线性插值采样函数 - 从特征图中采样一个浮点坐标处的值
 *
 * 双线性插值原理：
 * 对于浮点坐标 (x, y)，先找到周围四个整数坐标的像素值
 * 然后根据距离权重组合这四个值，得到平滑的插值结果
 */
__device__ float bilinear_sampling(const float*& bottom_data,  // 输入特征数据指针
                                   const int&    height,       // 特征图高度
                                   const int&    width,        // 特征图宽度
                                   const int&    num_embeds,   // 特征通道数
                                   const float&  h_im,         // 采样点的高度坐标（浮点）
                                   const float&  w_im,         // 采样点的宽度坐标（浮点）
                                   const int&    base_ptr      // 该特征图在内存中的基址
)
{
    // 向下取整得到四个临界点的整数坐标
    const int h_low  = floorf(h_im);  // 下方整数坐标
    const int w_low  = floorf(w_im);  // 左方整数坐标
    const int h_high = h_low + 1;     // 上方整数坐标
    const int w_high = w_low + 1;     // 右方整数坐标

    // 计算浮点坐标到整数坐标的距离
    const float lh = h_im - h_low;         // 距离下方的距离（高度方向）
    const float lw = w_im - w_low;         // 距离左方的距离（宽度方向）
    const float hh = 1 - lh, hw = 1 - lw;  // 相反方向的距离（用于权重）

    // 计算内存步长
    const int w_stride          = num_embeds;                   // 宽度方向的步长
    const int h_stride          = width * w_stride;             // 高度方向的步长
    const int h_low_ptr_offset  = h_low * h_stride;             // 下方行的偏移
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;  // 上方行的偏移
    const int w_low_ptr_offset  = w_low * w_stride;             // 左方列的偏移
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;  // 右方列的偏移

    // 采样四个角的像素值
    float v1 = 0;  // 左上角
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1             = bottom_data[ptr1];
    }
    float v2 = 0;  // 右上角
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2             = bottom_data[ptr2];
    }
    float v3 = 0;  // 左下角
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3             = bottom_data[ptr3];
    }
    float v4 = 0;  // 右下角
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4             = bottom_data[ptr4];
    }

    // 计算双线性插值的权重
    // 权重为距离的乘积（距离越近权重越大）
    const float w1 = hh * hw;  // 左上权重
    const float w2 = hh * lw;  // 右上权重
    const float w3 = lh * hw;  // 左下权重
    const float w4 = lh * lw;  // 右下权重

    // 组合四个值得到最终的插值结果
    const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

/*
 * 双线性插值的反向传播梯度计算函数
 *
 * 功能：计算采样位置和权重对于双线性插值结果的梯度
 * 这些梯度用于在反向传播时更新采样位置和权重参数
 */
__device__ void bilinear_sampling_grad(const float*& bottom_data,             // 输入特征数据
                                       const float&  weight,                  // 当前的权重系数
                                       const int&    height,                  // 特征图高度
                                       const int&    width,                   // 特征图宽度
                                       const int&    num_embeds,              // 特征通道数
                                       const float&  h_im,                    // 采样点高度坐标
                                       const float&  w_im,                    // 采样点宽度坐标
                                       const int&    base_ptr,                // 特征图基址
                                       const float&  grad_output,             // 损失对输出的梯度
                                       float*&       grad_mc_ms_feat,         // 特征梯度（输出）
                                       float*        grad_sampling_location,  // 采样位置梯度（输出）
                                       float*        grad_weights             // 权重梯度（输出）
)
{
    // 获取四个临界点的整数坐标
    const int h_low  = floorf(h_im);
    const int w_low  = floorf(w_im);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    // 计算距离和权重
    const float lh = h_im - h_low;
    const float lw = w_im - w_low;
    const float hh = 1 - lh, hw = 1 - lw;

    // 计算内存步长
    const int w_stride          = num_embeds;
    const int h_stride          = width * w_stride;
    const int h_low_ptr_offset  = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset  = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

    // 双线性插值权重
    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    // 通过权重缩放后向梯度
    const float top_grad_mc_ms_feat = grad_output * weight;

    // 初始化采样位置梯度
    float grad_h_weight = 0, grad_w_weight = 0;

    // 计算各个角的梯度贡献
    float v1 = 0;  // 左上角
    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1             = bottom_data[ptr1];
        // 对采样位置的梯度：特征值乘以对应的权重微分
        grad_h_weight -= hw * v1;  // ∂(w1*v1)/∂lh
        grad_w_weight -= hh * v1;  // ∂(w1*v1)/∂lw
        // 对特征的梯度：权重乘以后向梯度
        atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
    }
    float v2 = 0;  // 右上角
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2             = bottom_data[ptr2];
        grad_h_weight -= lw * v2;
        grad_w_weight += hh * v2;
        atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
    }
    float v3 = 0;  // 左下角
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3             = bottom_data[ptr3];
        grad_h_weight += hw * v3;
        grad_w_weight -= lh * v3;
        atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
    }
    float v4 = 0;  // 右下角
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4             = bottom_data[ptr4];
        grad_h_weight += lw * v4;
        grad_w_weight += lh * v4;
        atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
    }

    // 计算双线性插值的值（用于权重梯度）
    const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    // 使用原子操作累加梯度（因为多个线程可能写入同一内存位置）
    atomicAdd(grad_weights, grad_output * val);                                           // 权重梯度
    atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);       // 宽度方向
    atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);  // 高度方向
}

/*
 * 前向传播 CUDA 内核
 *
 * 每个线程处理一个输出特征点的一个通道
 * 通过 atomicAdd 累加来自各个采样点的贡献
 */
__global__ void deformable_aggregation_kernel(const int    num_kernels,        // 总的计算任务数（线程数）
                                              float*       output,             // 输出特征 [bs, num_anchors, num_embeds]
                                              const float* mc_ms_feat,         // 多相机多尺度特征
                                              const int*   spatial_shape,      // 各尺度的空间尺寸
                                              const int*   scale_start_index,  // 尺度的起始索引
                                              const float* sample_location,    // 采样点坐标
                                              const float* weights,            // 采样权重
                                              int          batch_size,
                                              int          num_cams,
                                              int          num_feat,
                                              int          num_embeds,
                                              int          num_scale,
                                              int          num_anchors,
                                              int          num_pts,
                                              int          num_groups)
{
    // 计算当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) {
        return;  // 超出范围的线程直接返回
    }

    // 从线性索引反解各个维度的索引
    const float weight        = *(weights + idx / (num_embeds / num_groups));
    const int   channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    // 构建锚点的全局索引
    anchor_index = batch_index * num_anchors + anchor_index;

    // 计算采样位置数据的偏移
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 读取采样坐标（归一化到 [0, 1]）
    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) {
        return;  // 超出范围的采样点无效
    }
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) {
        return;
    }

    // 计算特征数据的基址
    int       cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    // 获取该尺度的特征图尺寸
    cam_scale_index = cam_scale_index << 1;
    const int h     = spatial_shape[cam_scale_index];
    const int w     = spatial_shape[cam_scale_index + 1];

    // 将归一化坐标转换为像素坐标
    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    // 使用双线性插值采样，累加到输出
    atomicAdd(output + anchor_index * num_embeds + channel_index,
              bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight);
}

/*
 * 反向传播 CUDA 内核
 *
 * 计算特征、采样位置和权重对损失函数的梯度
 */
__global__ void deformable_aggregation_grad_kernel(const int    num_kernels,             // 总的计算任务数
                                                   const float* mc_ms_feat,              // 原始特征
                                                   const int*   spatial_shape,           // 空间尺寸
                                                   const int*   scale_start_index,       // 尺度索引
                                                   const float* sample_location,         // 采样位置
                                                   const float* weights,                 // 权重
                                                   const float* grad_output,             // 输出梯度
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
                                                   int          num_groups)
{
    // 计算线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) {
        return;
    }

    // 反解各维度索引
    const int weights_ptr   = idx / (num_embeds / num_groups);
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index         = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 检查采样位置有效性
    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) {
        return;
    }
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) {
        return;
    }

    // 获取该输出位置的梯度
    const float grad = grad_output[anchor_index * num_embeds + channel_index];

    // 计算特征基址
    int       cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    // 获取特征图尺寸
    cam_scale_index = cam_scale_index << 1;
    const int h     = spatial_shape[cam_scale_index];
    const int w     = spatial_shape[cam_scale_index + 1];

    // 计算像素坐标
    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    // 获取权重值
    const float weight = weights[weights_ptr];

    // 指向梯度数据的指针
    float* grad_weights_ptr  = grad_weights + weights_ptr;
    float* grad_location_ptr = grad_sampling_location + loc_offset;

    // 调用双线性采样梯度函数计算各项梯度
    bilinear_sampling_grad(mc_ms_feat, weight, h, w, num_embeds, h_im, w_im, value_offset, grad, grad_mc_ms_feat,
                           grad_location_ptr, grad_weights_ptr);
}

/*
 * 前向传播主函数 - 启动 CUDA 内核
 */
void deformable_aggregation(float*       output,
                            const float* mc_ms_feat,
                            const int*   spatial_shape,
                            const int*   scale_start_index,
                            const float* sample_location,
                            const float* weights,
                            int          batch_size,
                            int          num_cams,
                            int          num_feat,
                            int          num_embeds,
                            int          num_scale,
                            int          num_anchors,
                            int          num_pts,
                            int          num_groups)
{
    // 计算总的计算任务数
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;

    // 启动 CUDA 内核
    // 网格大小：(num_kernels/128) 个块，每块 128 个线程
    deformable_aggregation_kernel<<<(int)ceil(((double)num_kernels / 128)), 128>>>(
        num_kernels, output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, batch_size,
        num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups);
}

/*
 * 反向传播主函数 - 启动梯度计算内核
 */
void deformable_aggregation_grad(const float* mc_ms_feat,
                                 const int*   spatial_shape,
                                 const int*   scale_start_index,
                                 const float* sample_location,
                                 const float* weights,
                                 const float* grad_output,
                                 float*       grad_mc_ms_feat,
                                 float*       grad_sampling_location,
                                 float*       grad_weights,
                                 int          batch_size,
                                 int          num_cams,
                                 int          num_feat,
                                 int          num_embeds,
                                 int          num_scale,
                                 int          num_anchors,
                                 int          num_pts,
                                 int          num_groups)
{
    // 计算总的计算任务数
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;

    // 启动梯度计算内核
    deformable_aggregation_grad_kernel<<<(int)ceil(((double)num_kernels / 128)), 128>>>(
        num_kernels, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights, grad_output,
        grad_mc_ms_feat, grad_sampling_location, grad_weights, batch_size, num_cams, num_feat, num_embeds, num_scale,
        num_anchors, num_pts, num_groups);
}

__global__ void deformable_aggregation_kernel(const int    num_kernels,        // 总的计算任务数（线程数）
                                              float*       output,             // 输出特征 [bs, num_anchors, num_embeds]
                                              const float* mc_ms_feat,         // 多相机多尺度特征
                                              const int*   spatial_shape,      // 各尺度的空间尺寸
                                              const int*   scale_start_index,  // 尺度的起始索引
                                              const float* sample_location,    // 采样点坐标
                                              const float* weights,            // 采样权重
                                              int          batch_size,
                                              int          num_cams,
                                              int          num_feat,
                                              int          num_embeds,
                                              int          num_scale,
                                              int          num_anchors,
                                              int          num_pts,
                                              int          num_groups)
{
    // 计算当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) {
        return;  // 超出范围的线程直接返回
    }

    // 从线性索引反解各个维度的索引
    const float weight        = *(weights + idx / (num_embeds / num_groups));
    const int   channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    // 构建锚点的全局索引
    anchor_index = batch_index * num_anchors + anchor_index;

    // 计算采样位置数据的偏移
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 读取采样坐标（归一化到 [0, 1]）
    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) {
        return;  // 超出范围的采样点无效
    }
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) {
        return;
    }

    // 计算特征数据的基址
    int       cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    // 获取该尺度的特征图尺寸
    cam_scale_index = cam_scale_index << 1;
    const int h     = spatial_shape[cam_scale_index];
    const int w     = spatial_shape[cam_scale_index + 1];

    // 将归一化坐标转换为像素坐标
    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    // 使用双线性插值采样，累加到输出
    atomicAdd(output + anchor_index * num_embeds + channel_index,
              bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight);
}

__global__ void deformable_aggregation_grad_kernel(const int    num_kernels,             // 总的计算任务数
                                                   const float* mc_ms_feat,              // 原始特征
                                                   const int*   spatial_shape,           // 空间尺寸
                                                   const int*   scale_start_index,       // 尺度索引
                                                   const float* sample_location,         // 采样位置
                                                   const float* weights,                 // 权重
                                                   const float* grad_output,             // 输出梯度
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
                                                   int          num_groups)
{
    // 计算线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) {
        return;
    }

    // 反解各维度索引
    const int weights_ptr   = idx / (num_embeds / num_groups);
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index         = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    // 检查采样位置有效性
    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) {
        return;
    }
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) {
        return;
    }

    // 获取该输出位置的梯度
    const float grad = grad_output[anchor_index * num_embeds + channel_index];

    // 计算特征基址
    int       cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    // 获取特征图尺寸
    cam_scale_index = cam_scale_index << 1;
    const int h     = spatial_shape[cam_scale_index];
    const int w     = spatial_shape[cam_scale_index + 1];

    // 计算像素坐标
    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;

    // 获取权重值
    const float weight = weights[weights_ptr];

    // 指向梯度数据的指针
    float* grad_weights_ptr  = grad_weights + weights_ptr;
    float* grad_location_ptr = grad_sampling_location + loc_offset;

    // 调用双线性采样梯度函数计算各项梯度
    bilinear_sampling_grad(mc_ms_feat, weight, h, w, num_embeds, h_im, w_im, value_offset, grad, grad_mc_ms_feat,
                           grad_location_ptr, grad_weights_ptr);
}

/*
 * 前向传播主函数 - 启动 CUDA 内核
 */
void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    // 计算总的计算任务数
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;

    // 启动 CUDA 内核
    // 网格大小：(num_kernels/128) 个块，每块 128 个线程
    deformable_aggregation_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
}

/*
 * 反向传播主函数 - 启动梯度计算内核
 */
void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    // 计算总的计算任务数
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;

    // 启动梯度计算内核
    deformable_aggregation_grad_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
}
