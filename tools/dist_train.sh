#!/usr/bin/env bash

# CONFIG: 训练配置文件路径。
CONFIG=$1
# GPUS: 启动的 GPU / 进程数量。
GPUS=$2
# PORT: 分布式通信端口；若外部未设置，则使用默认值 28650。
PORT=${PORT:-28650}

# 将仓库根目录加入 PYTHONPATH，保证训练脚本能导入自定义模块。
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# 用 torch.distributed.launch 启动多卡训练，`${@:3}` 透传额外训练参数。
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
