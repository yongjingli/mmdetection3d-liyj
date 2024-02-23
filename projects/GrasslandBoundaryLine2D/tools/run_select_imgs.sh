test_root="/home/liyongjing/Egolee/hdd-data/test_data/20231217/rosbag2_2023_12_16-16_28_19_dyg/images"
s_vis_root="/home/liyongjing/Egolee/hdd-data/test_data/20231217/rosbag2_2023_12_16-16_28_19_dyg/1"

#/home/liyongjing/anaconda3/envs/mmdet3/bin/python \
/data-hdd/liyj/packages/anaconda3/envs/mmdet3/bin/python \
/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/tools/demo_infer_for_select_imgs.py \
--test_root $test_root --s_vis_root $s_vis_root
