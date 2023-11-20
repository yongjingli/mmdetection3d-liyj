#cd ../../
##python tools/train.py ${CONFIG_FILE} [optional arguments]
##CONFIG_FILE="./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py"
##CONFIG_FILE="./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_v1.0-min.py"
#
## debug gbld
#CONFIG_FILE="./projects/GrasslandBoundaryLine2D/configs/gbld_overfit_20230811.py"
#
#python tools/train.py $CONFIG_FILE
#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=4

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/../..


# 加入可见属性等
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/gbld_v0.3_20231024_batch_12_no_crop_split_line

# batch_size=12, no crop aug, split line by type, seg_emb=10.0
#debug_crop
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/debug_no_spllit/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_stage2
work_dir=${SCRIPT_DIR}/work_dirs/on_server/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3

python ${ROOT_DIR}/tools/train.py \
    ${SCRIPT_DIR}/configs/gbld_config_v0.3.py \
    --work-dir ${work_dir}