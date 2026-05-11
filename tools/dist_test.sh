#!/usr/bin/env bash

# CONFIG: 测试配置文件路径。
CONFIG=$1
# CHECKPOINT: 待评测模型权重路径。
CHECKPOINT=$2
# GPUS: 启动的 GPU / 进程数量。
GPUS=$3
# PORT: 分布式通信端口；若外部未设置，则使用默认值 29610。
PORT=${PORT:-29610}

# 将仓库根目录加入 PYTHONPATH，确保可以导入 projects/ 下的自定义模块。
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# 使用 torch.distributed.launch 启动多进程测试脚本，`${@:4}` 透传剩余命令行参数。
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
