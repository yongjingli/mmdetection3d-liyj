#!/usr/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/../..

work_dir=${SCRIPT_DIR}/work_dirs/gbld_overfit_20230815
checkpoint=${work_dir}/epoch_250.pth
show_dir=${work_dir}/visual_epoch_250

# val
python ${ROOT_DIR}/tools/test.py \
    ${SCRIPT_DIR}/configs/gbld_debug_config_no_dcn.py \
    ${checkpoint} \
    --task mono_det \
    --cfg-options test_evaluator.output_dir=${show_dir}_test/format_results \
    --work-dir ${work_dir} \
    --wait-time 1 \
    --show                                     # 设置是否显示 # 设置显示的间距
#    --show-dir ${show_dir}_test \        # 设置是否保存显示内容

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
