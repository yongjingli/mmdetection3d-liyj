cd ..

#https://mmdetection3d.readthedocs.io/en/latest/user_guides/inference.html
# 1.Point cloud demo
#python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]

# 2.Monocular 3D demo
#python demo/mono_det_demo.py ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--cam-type ${CAM_TYPE}] [--score-thr ${SCORE-THR}] [--out-dir ${OUT_DIR}] [--show]

# (1)focs3d
#CONFIG_FILE="./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
#CHECKPOINT_FILE="./checkpoints/focal3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"

# retrain
# 由于只是在v1.0-mini进行训练,在demo图片的效果并不好
#CONFIG_FILE="./configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_v1.0-min.py"
#CHECKPOINT_FILE="./work_dirs/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_v1.0-min/epoch_12.pth"

#echo "CONFIG_FILE:"$CONFIG_FILE
#echo "CHECKPOINT_FILE:"$CHECKPOINT_FILE
#
#img_name="n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg"
#img_info_name="n015-2018-07-24-11-22-45+0800.pkl"
#
#python demo/mono_det_demo.py demo/data/nuscenes/$img_name \
#demo/data/nuscenes/$img_info_name  $CONFIG_FILE \
# $CHECKPOINT_FILE  --show --cam-type CAM_BACK #--score-thr 0.01

# 如果在设置阈值内没有检测结果, 则不会显示结果图片


# 3.Multi-modality demo
#python demo/multi_modality_demo.py ${PCD_FILE} ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]

#(1) bevfuse
#CONFIG_FILE="./projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
##CHECKPOINT_FILE="./checkpoints/bevfuse/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
##https://github.com/open-mmlab/mmdetection3d/issues/2584
## 需要将权重进行shape的转换,要不然导入不正常
#CHECKPOINT_FILE="./checkpoints/bevfuse/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af_convert.pth"
#
#python demo/multi_modality_demo.py \
# demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin\
# demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
# $CONFIG_FILE $CHECKPOINT_FILE --cam-type all --score-thr 0.2 --show


#(2) mvxnet
CONFIG_FILE="./configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py"
CHECKPOINT_FILE="./checkpoints/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a_convert.pth"

python demo/multi_modality_demo.py \
 demo/data/kitti/000008.bin\
 demo/data/kitti/000008.png demo/data/kitti/000008.pkl \
 $CONFIG_FILE $CHECKPOINT_FILE \
 --cam-type CAM2 --score-thr 0.2 --show

#python demo/multi_modality_demo.py demo/data/kitti/kitti_000008.bin demo/data/kitti/kitti_000008.png demo/data/kitti/kitti_000008_infos.pkl configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth

# 4.3D Segmentation
#python demo/pcd_seg_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out-dir ${OUT_DIR}] [--show]