"""
CUDA 扩展编译配置
==================
该文件配置 PyTorch CUDA 扩展模块的编译参数
将 C++/CUDA 源代码编译成可被 Python 调用的共享库
"""
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,      # PyTorch 的 C++ 扩展构建工具
    CppExtension,        # C++ 扩展配置
    CUDAExtension,       # CUDA 扩展配置
)


def make_cuda_ext(
    name,                   # 扩展模块名称
    module,                 # 模块路径
    sources,                # C++ 源文件列表
    sources_cuda=[],        # CUDA 源文件列表
    extra_args=[],          # 额外的编译参数
    extra_include_path=[],  # 额外的头文件搜索路径
):
    """
    创建 CUDA 扩展配置

    这个函数根据是否有可用的 CUDA 设备来生成相应的编译配置
    - 如果有 CUDA：编译为完整的 CUDA 扩展（包含 .cu 文件）
    - 如果没有 CUDA：降级编译为纯 C++ 扩展（仅使用 .cpp 文件）

    参数:
        name (str): 扩展模块名称
        module (str): 模块路径（用于生成最终的模块名称）
        sources (list): C++ 源文件列表
        sources_cuda (list): CUDA 源文件列表（可选）
        extra_args (list): 额外编译参数
        extra_include_path (list): 额外头文件路径

    返回:
        Extension: C++/CUDA 扩展配置对象
    """
    # 定义预处理宏
    define_macros = []

    # 初始化编译参数字典（分别为 C++ 编译器和 CUDA 编译器）
    extra_compile_args = {"cxx": [] + extra_args}

    # 检查是否有 CUDA 设备可用，或者通过环境变量强制启用 CUDA 编译
    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        # 添加 WITH_CUDA 宏定义，用于条件编译
        define_macros += [("WITH_CUDA", None)]

        # 使用 CUDA 扩展
        extension = CUDAExtension

        # 设置 CUDA 编译参数
        # 这些参数禁用了一些 CUDA 的半精度浮点操作以提高兼容性
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",        # 禁用 HALF 操作符重载
            "-D__CUDA_NO_HALF_CONVERSIONS__",      # 禁用 HALF 隐式转换
            "-D__CUDA_NO_HALF2_OPERATORS__",       # 禁用 HALF2 操作符重载
        ]
        # 将 CUDA 源文件加入编译源列表
        sources += sources_cuda
    else:
        # 如果没有 CUDA，打印警告信息并使用纯 C++ 编译
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    # 返回扩展配置对象
    return extension(
        # 最终的模块名称（导入时使用）
        name="{}.{}".format(module, name),
        # 源文件的绝对路径列表
        sources=[os.path.join(*module.split("."), p) for p in sources],
        # 头文件搜索路径
        include_dirs=extra_include_path,
        # 预处理宏定义
        define_macros=define_macros,
        # 编译参数
        extra_compile_args=extra_compile_args,
    )


# 脚本入口：setuptools 将调用这个函数来配置和编译扩展
if __name__ == "__main__":
    setup(
        # 包名称
        name="deformable_aggregation_ext",
        # 扩展模块列表
        ext_modules=[
            make_cuda_ext(
                # 扩展名称
                "deformable_aggregation_ext",
                # 模块路径（当前目录）
                module=".",
                # C++ 源文件
                sources=[
                    f"src/deformable_aggregation.cpp",
                ],
                # CUDA 源文件
                sources_cuda=[
                    f"src/deformable_aggregation_cuda.cu",
                ],
            ),
        ],
        # 构建命令处理器
        cmdclass={"build_ext": BuildExtension},
    )
