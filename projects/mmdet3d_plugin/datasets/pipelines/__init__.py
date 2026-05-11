# 变换类：负责筛框、归一化、打包成 Sparse4D 需要的输入格式。
from .transform import (
    InstanceNameFilter,
    CircleObjectRangeFilter,
    NormalizeMultiviewImage,
    NuScenesSparse4DAdaptor,
    MultiScaleDepthMapGenerator,
)
# 增强类：负责图像几何增强、3D 旋转增强、颜色扰动。
from .augment import (
    ResizeCropFlipImage,
    BBoxRotation,
    PhotoMetricDistortionMultiViewImage,
)
# 加载类：负责从磁盘读取多目图像和点云。
from .loading import LoadMultiViewImageFromFiles, LoadPointsFromFile

__all__ = [
    # 统一声明 pipeline registry 中可复用的公开模块名。
    "InstanceNameFilter",
    "ResizeCropFlipImage",
    "BBoxRotation",
    "CircleObjectRangeFilter",
    "MultiScaleDepthMapGenerator",
    "NormalizeMultiviewImage",
    "PhotoMetricDistortionMultiViewImage",
    "NuScenesSparse4DAdaptor",
    "LoadMultiViewImageFromFiles",
    "LoadPointsFromFile",
]
