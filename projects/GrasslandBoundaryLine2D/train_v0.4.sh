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
#export CUDA_VISIBLE_DEVICES=4,5,6
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/../..


# 加入可见属性等
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/gbld_v0.3_20231024_batch_12_no_crop_split_line

# batch_size=12, no crop aug, split line by type, seg_emb=10.0
#debug_crop
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/debug_no_spllit/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_stage2
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/all_discrimate_aux/gbld_v0.4_20231121_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_overfit_aux128

#采用1123最新的数据进行辅助任务的训练
# 原来的版本
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/all_discrimate_aux_1123/gbld_v0.4_20231121_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3

# 将其作为辅助任务的分支
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/all_discrimate_aux_1123/gbld_v0.4_20231121_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_aux_discrimate

# 将其作为辅助任务的分支与emb共享分支
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/all_discrimate_aux_1123/gbld_v0.4_20231125_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_aux_discrimate_same_branch

# 过拟合调试
work_dir=${SCRIPT_DIR}/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale1_fix_crop_segment_label_rotate

# 2023-11-25 filter data by self
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/debug_upscale/gbld_v0.4_20231202_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_filter_upscale1


python ${ROOT_DIR}/tools/train.py \
    ${SCRIPT_DIR}/configs/gbld_config_v0.4_overfit.py \
    --work-dir ${work_dir} \
#    --local_rank 0 \
#    --resume      # 是否resume训练