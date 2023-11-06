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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/../..

work_dir=${SCRIPT_DIR}/work_dirs/gbld_overfit_20230927_v0.2_fit_line_crop_batch_12

python ${ROOT_DIR}/tools/train.py \
    ${SCRIPT_DIR}/configs/gbld_debug_config_no_dcn_v0.2.py \
    --work-dir ${work_dir}
