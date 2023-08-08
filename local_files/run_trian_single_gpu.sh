#python tools/train.py ${CONFIG_FILE} [optional arguments]

cd ../
#CONFIG_FILE="./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py"
#CONFIG_FILE="./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_v1.0-min.py"

# debug gbld
CONFIG_FILE="./projects/GlasslandBoundaryLine2D/configs/gbld_debug_config.py"

python tools/train.py $CONFIG_FILE