#!/usr/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/../..

#work_dir=${SCRIPT_DIR}/work_dirs/debug_visible_hanging_covered8
#checkpoint=${work_dir}/epoch_200.pth
#show_dir=${work_dir}/visual_epoch_200

# debub crop
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/debug_crop/gbld_v0.3_20231024_batch_12_with_crop_split_line_10_emb_weight
#work_dir=${SCRIPT_DIR}/work_dirs/gbld_v0.3_20231026_batch_12_with_crop_split_line_10_emb_weight
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/gbld_v0.3_20231102_batch_12_with_crop_split_line_10_emb_weight
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/filter_data_2023_11_25/gbld_v0.4_20231125_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_filter_by_self
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3
#work_dir=${SCRIPT_DIR}/work_dirs/on_server/debug_discrimate/gbld_v0.4_20231208_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb_with_discrimate


#work_dir=${SCRIPT_DIR}/work_dirs/on_server/debug_discrimate/gbld_v0.4_20231208_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb_with_discrimate

#work_dir=${SCRIPT_DIR}/work_dirs/on_server/debug_discrimate/gbld_v0.4_20231218_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb_discrimate_16
work_dir=${SCRIPT_DIR}/work_dirs/on_server/normal/gbld_v0.4_20240102_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_discrimate_rotate

checkpoint=${work_dir}/epoch_250.pth
show_dir=${work_dir}/visual_epoch_250

#${SCRIPT_DIR}/configs/gbld_debug_config_no_dcn_datasetv2 \
# val
python ${ROOT_DIR}/tools/test.py \
    ${work_dir}/gbld_config_v0.4_overfit.py \
    ${checkpoint} \
    --task mono_det \
    --cfg-options test_evaluator.output_dir=${show_dir}_test/format_results \
    --work-dir ${work_dir} \
    --wait-time 1 \
#    --show                                     # 设置是否显示 # 设置显示的间距
#    --show-dir ${show_dir}_test \              # 设置是否保存显示内容

#
## convert to video
#${SCRIPT_DIR}/tools/merge_img_to_video.sh ${show_dir}_val/vis_data/vis_image 1
#
## test
#python ${ROOT_DIR}/tools/test.py \
#    ${SCRIPT_DIR}/configs/ddrnet_23_test.py \
#    ${checkpoint} \
#    --work-dir ${work_dir} \
#    --show-dir ${show_dir}_test \
#    --cfg-options test_evaluator.output_dir=${show_dir}_test/format_results \
#    --task seg
#
## convert to vidio
#${SCRIPT_DIR}/tools/merge_img_to_video.sh ${show_dir}_test/vis_data/vis_image 10
