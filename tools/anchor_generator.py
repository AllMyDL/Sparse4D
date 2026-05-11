import numpy as np
from sklearn.cluster import KMeans
import mmcv

from projects.mmdet3d_plugin.core.box3d import *


def get_kmeans_anchor(
    ann_file,
    num_anchor=900,
    detection_range=55,
    output_file_name="nuscenes_kmeans900.npy",
    verbose=False,
):
    # ann_file: 由数据预处理阶段生成的 pkl 标注文件路径。
    # num_anchor: 希望聚类得到的 anchor 数量。
    # detection_range: 仅统计该半径范围内的目标，避免远距离稀疏目标干扰聚类中心。
    # output_file_name: 输出的 anchor numpy 文件名。
    # verbose: 是否打印 sklearn KMeans 的详细日志。

    # 读取 pkl 标注文件，data["infos"] 中每个元素对应一个样本的标注信息。
    data = mmcv.load(ann_file, file_format="pkl")
    # 将所有样本中的 gt_boxes 纵向拼接成一个大矩阵，形状通常为 [总目标数, 10/11]。
    gt_boxes = np.concatenate([x["gt_boxes"] for x in data["infos"]], axis=0)
    # 计算每个 3D 框中心点到雷达原点的欧氏距离，只使用 xyz 三个坐标。
    distance = np.linalg.norm(gt_boxes[:, :3], axis=-1, ord=2)
    # 仅保留检测范围内的目标框。
    mask = distance <= detection_range
    gt_boxes = gt_boxes[mask]
    # 对目标中心点做 KMeans，聚类中心将作为 anchor 的空间先验位置。
    clf = KMeans(n_clusters=num_anchor, verbose=verbose)
    print("===========Starting kmeans, please wait.===========")
    # 只使用 X/Y/Z 三个维度做聚类，不把尺寸和朝向直接纳入聚类空间。
    clf.fit(gt_boxes[:, [X, Y, Z]])
    # 初始化 anchor 张量。
    # 11 维格式由 box3d 定义，后面只填充当前位置、平均尺寸和默认朝向。
    anchor = np.zeros((num_anchor, 11))
    # 将聚类中心写入 anchor 的中心坐标部分。
    anchor[:, [X, Y, Z]] = clf.cluster_centers_
    # 尺寸部分不按 cluster 分别统计，而是统一使用全体 GT 的平均尺寸对数值。
    # 这样可以让 anchor 的位置由聚类决定，尺寸由全局统计决定。
    anchor[:, [W, L, H]] = np.log(gt_boxes[:, [W, L, H]].mean(axis=0))
    # 将 cos(yaw) 默认设为 1，等价于初始朝向为 0。
    anchor[:, COS_YAW] = 1
    # 保存生成好的 anchor 先验。
    np.save(output_file_name, anchor)
    print(f"===========Done! Save results to {output_file_name}.===========")


if __name__ == "__main__":
    import argparse

    # 该脚本是一个独立工具，用命令行参数指定输入标注和输出 anchor 文件。
    parser = argparse.ArgumentParser(description="anchor kmeans")
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--num_anchor", type=int, default=900)
    parser.add_argument("--detection_range", type=float, default=55)
    parser.add_argument(
        "--output_file_name", type=str, default="_nuscenes_kmeans900.npy"
    )
    parser.add_argument("--verbose", action="store_true")
    # args: 汇总后的命令行参数对象。
    args = parser.parse_args()
    # 将命令行参数逐个传入主函数执行聚类。
    get_kmeans_anchor(
        args.ann_file,
        args.num_anchor,
        args.detection_range,
        args.output_file_name,
        args.verbose,
    )
