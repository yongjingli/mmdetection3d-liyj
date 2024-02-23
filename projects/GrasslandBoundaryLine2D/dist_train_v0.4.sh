#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=6,7

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/../..

# GPUS代表的是GPU的数量，通过visible的方式指定gpu的id
export CUDA_VISIBLE_DEVICES=4,5,6

CONFIG=${SCRIPT_DIR}/configs/gbld_config_v0.4_overfit.py
GPUS=3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
#PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

echo "ROOT_DIR:"$ROOT_DIR
echo "CONFIG:"$CONFIG
echo "GPUS:"$GPUS
echo "NNODES:"$NNODES
echo "PORT:"$PORT
echo "MASTER_ADDR:"$MASTER_ADDR
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/debug_discrimate/gbld_v0.4_20231212_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb_with_discrimate
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/normal/gbld_v0.4_20231214_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/debug_rotate/gbld_v0.4_20231218_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point
work_dir=${SCRIPT_DIR}/work_dirs/on_server/normal/gbld_v0.4_20240104_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter

# 过拟合调试
#work_dir=${SCRIPT_DIR}/work_dirs/debug_point_emb/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale2_fix_crop_segment_label3

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $ROOT_DIR/tools/train.py \
    $CONFIG \
    --work-dir ${work_dir} \
    --launcher pytorch ${@:3} \
#    --resume
echo "end"