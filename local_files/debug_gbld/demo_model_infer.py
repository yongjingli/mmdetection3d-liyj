# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
from mmengine.dataset import Compose, pseudo_collate
from copy import deepcopy
from mmdet3d.apis import inference_mono_3d_detector, init_model
from mmdet3d.registry import VISUALIZERS
import numpy as np
import mmdet
import cv2
import os
import torch
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from mmengine.dataset import Compose, pseudo_collate
from debug_utils import decode_gt_lines, cal_points_orient, draw_orient, filter_near_same_points


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]


def main():
    # build the model from a config file and a checkpoint file
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/configs/gbld_debug_config.py"
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug/20230804_164059/vis_data/config.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug/20230804_164059/epoch_250.pth"

    # no dcn
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug_no_dcn/20230812_182852/vis_data/config.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug_no_dcn/epoch_250.pth"

    # no dcn v0.2
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230905_v0.2/gbld_debug_config_no_dcn_v0.2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230905_v0.2/epoch_250.pth"

    # no dcn v0.2 with orient
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230906_v0.2_fit_line_crop/gbld_debug_config_no_dcn_v0.2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230906_v0.2_fit_line_crop/epoch_200.pth"

    # no dcn v0.2 with orient2
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230927_v0.2_fit_line_crop/gbld_debug_config_no_dcn_v0.2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230927_v0.2_fit_line_crop/epoch_250.pth"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230927_v0.2_fit_line_crop_batch_12/epoch_250.pth"

    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230927_v0.2_fit_line_crop/gbld_debug_config_no_dcn_v0.2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230927_v0.2_fit_line_crop/epoch_250.pth"

    # debug
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20231013_v0.2_fit_line_crop_batch_6/gbld_debug_config_no_dcn_v0.2_1013.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20231013_v0.2_fit_line_crop_batch_6/epoch_250.pth"

    # 服务器
    # config_path = "/data-hdd/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20231013_v0.2_fit_line_crop_batch_6/gbld_debug_config_no_dcn_v0.2_1013.py"
    # checkpoint_path = "/data-hdd/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20231013_v0.2_fit_line_crop_batch_6/epoch_250.pth"

    # visible hanging covered
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered/gbld_debug_config_no_dcn_datasetv2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered/epoch_250.pth"

    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered6/gbld_debug_config_no_dcn_datasetv2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered6/epoch_250.pth"

    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered8/gbld_debug_config_no_dcn_datasetv2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered8/epoch_200.pth"

    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered8/gbld_debug_config_no_dcn_datasetv2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered8/epoch_200.pth"

    # 服务器
    # config_path = "/data-hdd/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered8/gbld_debug_config_no_dcn_datasetv2.py"
    # checkpoint_path = "/data-hdd/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered8/epoch_200.pth"

    # 20231024
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231023_2/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231023_2/epoch_250.pth"

    # 将crop数据增强去除
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231024_no_crop/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231024_no_crop/epoch_100.pth"

    # 将seg-emg weight提升为10,采用split-line的方式
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231024_batch_12_no_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231024_batch_12_no_crop_split_line_10_emb_weight/epoch_200.pth"


    # debug crop
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_crop/gbld_v0.3_20231024_batch_12_no_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_crop/gbld_v0.3_20231024_batch_12_no_crop_split_line_10_emb_weight/epoch_250.pth"

    # 20231026
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231026_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231026_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 20231030
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231030_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231030_batch_12_with_crop_split_line_10_emb_weight/epoch_200.pth"

    # 20231031
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231031_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231031_batch_12_with_crop_split_line_10_emb_weight/epoch_100.pth"

    # 20231031
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231101_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231101_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 20231102
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231102_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231102_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 20231108
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls/epoch_200.pth"

    # 20231110
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls_dist_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls_dist_weight/epoch_200.pth"

    # 20231113
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/epoch_250.pth"

    # 20231113
    # 测试不按照类别分段
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/epoch_250.pth"

    # 20231116
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/epoch_250.pth"

    # 20231121-overfit
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/overfit/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_overfit/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/overfit/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_overfit/epoch_250.pth"

    # 20231116
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/epoch_250.pth"

    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231116_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/epoch_250.pth"

    # 20231127将数据过滤一遍
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231125_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_filter_by_self/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231125_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_filter_by_self/epoch_250.pth"

    # 将模型进行上采样
    # no up scale
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_no_upscale/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_no_upscale/epoch_250.pth"

    # 在neck的后面加上上采样
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_upscale/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_upscale/epoch_250.pth"

    # 对最后的特征图进行上采样
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale2/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale2/epoch_200.pth"

    # 对最后的特征图进行上采样,并且最大分辨率的绘制宽度为2
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale2_line_thinkness_2/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale2_line_thinkness_2/epoch_250.pth"

    # 对最后的特征图进行2倍上采样,并且最大分辨力的绘制宽度为2,并且修正crop数据时的bug
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale2_fix_crop/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale2_fix_crop/epoch_200.pth"

    # 对最后的特征图进行2倍上采样,并且最大分辨力的绘制宽度为2,并且修正crop数据时的bug, 并且进行分段标注
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale2_fix_crop_segment_label/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale2_fix_crop_segment_label/epoch_250.pth"

    # 对最后的特征图进行1倍上采样,并且最大分辨力的绘制宽度为2,并且修正crop数据时的bug, 并且进行分段标注
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale1_fix_crop_segment_label/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale1_fix_crop_segment_label/epoch_250.pth"


    # 12-02修改后的全量数据
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231202_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_filter_upscale1/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231202_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_filter_upscale1/epoch_250.pth"


    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231205_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231205_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/epoch_250.pth"

    # over-fit debug
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale1_fix_crop_segment_label_rotate/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_overfit/gbld_v0.4_20231201_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3_debug_finnal_upscale1_fix_crop_segment_label_rotate/epoch_250.pth"

    # 12-08 全量数据训练
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231208_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231208_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/epoch_250.pth"

    # 12-19  全量数据训练 + 行人数据 + 外边界遮挡 + 挑选corner-case数据
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231218_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231218_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/epoch_250.pth"

    # 01-04  全量数据训练 + 行人数据 + 外边界遮挡 + 挑选corner-case数据 + rotate + discrimate16
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/normal/gbld_v0.4_20240102_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_discrimate_rotate/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/normal/gbld_v0.4_20240102_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_discrimate_rotate/epoch_250.pth"

    # 01-06  全量数据训练 + 行人数据 + 外边界遮挡 + 挑选corner-case数据 （去除rotate + discrimate16）
    config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/normal/gbld_v0.4_20240104_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter/gbld_config_v0.4_overfit.py"
    checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/normal/gbld_v0.4_20240104_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter/epoch_250.pth"

    model = init_model(config_path, checkpoint_path, device='cuda:0')

    # 采用官方的方式进行推理
    cfg = model.cfg
    # build the data pipeline
    test_pipeline = deepcopy(cfg.val_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)

    # 服务器
    # test_root = "/data-hdd/XW_Data/collect_data/20231010/long_rosbag2_2023_10_10-15_04_43_test_1/parse_data/image"
    # test_root = "/data-hdd/XW_Data/collect_data/20231010/short_rosbag2_2023_10_10-15_49_36_test_2/parse_data_liyj/image"
    # test_root = "/data-hdd/XW_Data/collect_data/20231026am/big_rosbag2_2023_10_26-11_09_36/images"
    # test_root = "/data-hdd/XW_Data/collect_data/20231026am/small_rosbag2_2023_10_26-11_24_50/images"

    # lwj-2023-10-25
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231023/rosbag2_2023_10_11-10_07_59/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231023/rosbag2_2023_10_11-10_13_56/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231023/rosbag2_2023_10_11-10_24_14/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231023/rosbag2_2023_10_11-10_28_36/images"
    # test_root = "/data-hdd/XW_Data/collect_data/20231026am/big_rosbag2_2023_10_26-11_09_36/images"
    # 2023-11-14
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231114/rosbag2_2023_11_14-16_11_14/images"

    # 对测试集进行验证,这样比较容易发现问题（对验证集进行验证也更加容易发现问题）
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231025_mmdet3d_spline/test/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231108_mmdet3d_spline_by_cls/test/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231110_mmdet3d_spline_by_cls/test/images"

    # 20231031采集车
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231101/rosbag2_2023_10_31-16_09_00/camera_front_images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231101/rosbag2_2023_10_31-16_09_00/camera_front_left_images"

    # 测试近距离
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231104/rosbag2_camera_2023_11_04-17_06_47/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231104/rosbag2_camera_2023_11_04-17_07_30/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231104/rosbag2_camera_2023_11_04-17_09_30/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231104/rosbag2_camera_2023_11_04-17_15_34/images"

    # 原型机的采集
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_06_57_yanbian/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_12_06_daoludaquan/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_29_19_gongzidaolu/images"

    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_22_29_gongzitianjixian/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_42_25_gongzidaolu2/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_06_57_yanbian/images"

    # 采集车外采集
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231108/rosbag2_2023_11_03-15_24_51/front_camera"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231108/rosbag2_2022_09_08-18_07_28/front_camera"

    # 公司采集
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_06_57_yanbian/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231110/rosbag2_2023_11_10-11_44_23/images_camera_front"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231110/images_front_right"

    # 在公司对面上方的第一块草地
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231113/rosbag2_2023_11_13-09_53_33/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231113/rosbag2_2023_11_13-10_02_35/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231113/rosbag2_2023_11_13-10_53_55/images"

    # 万科采集
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/collect_car/20231103_wankejinsemengxiang/rosbag2_2022_09_08-18_07_28_deal/rosbag2_2022_09_08-18_07_28_imgs_202311114"

    # 2013-11-15
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/select_test_20231114"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231114/rosbag2_2023_11_14-16_11_14/images"

    # 石头和树周围土堆的数据
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231116/rosbag2_2023_11_16-10_51_42/images"

    # 测试集
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231110_mmdet3d_spline_by_cls/test/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/select_test_20231117"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/debug"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231116/select_imgs_2023116/t0"

    # cornet case
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_corner_case/231113_missed_and_split_lines"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_corner_case/231113_stone_edges"

    # 对面草地阴影
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231119/shitoushangmianbianyanxianwujian-1118/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231119/yinyingwujain-shitoushangwujianbianyanxian-1118/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231119/rosbag2_2023_11_18-11_13_49/images"

    # overfit
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231116_mmdet3d_spline_by_cls_overfit/train/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231116_mmdet3d_spline_by_cls/train/images"

    # 2023-11-22
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231122/20231121/rosbag2_2023_11_21-15_00_34_collect/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231122/20231121/rosbag2_2023_11_21-15_33_55_collect/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231122/20231121/rosbag2_2023_11_21-16_29_28_jinruguanmu/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231122/20231121/rosbag2_2023_11_21-16_32_14_wujianbianyanxian/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231122/20231121/rosbag2_2023_11_21-16_47_42_bianyanxianxiaoshi/images"

    # 室内的操作
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231122/20231121/rosbag2_2023_11_21-15_00_34_collect/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_corner_case/2023_11_16_curse_line"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_corner_case/2023_11_16_curse_line"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231130/rosbag2_2023_11_30-11_34_12/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231205/rosbag2_2023_12_05-17_19_07/images"


    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/debug_overfit/train/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231201/rosbag2_2023_12_01-12_05_19/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231204/rosbag2_2023_12_04-17_42_23/images"

    # corner-case test
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_corner_case/2023_11_24_before_test"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_corner_case/2013_11_24_rosbag2_2023_11_21-15_00_34_collect"

    # 假草
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231206/total_outdoor/images"
    # test_root = "/home/liyongjing/Downloads/2/indoor"

    # test_roots = ["/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_22-16_54_34/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-09_01_40/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-09_07_55/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-09_25_21/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-09_34_05/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-09_41_38/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-09_54_55/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_00_23/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_01_54/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_04_43/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_10_10/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_20_50/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_27_52/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_32_06/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_35_51/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_38_58/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_48_30/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_58_35/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-11_11_31/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-11_17_47/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-11_26_15/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_08_35/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_25_53/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_26_43/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_31_03/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_33_08/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_44_44/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_47_38/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_48_00/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_49_55/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-14_54_01/images",
    #               "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-15_00_17/images",
    #            ]

    test_roots = [1]
    for test_root in test_roots:

        # 验证测试数据
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231211_test_public_model/rosbag2_2023_12_06-11_16_25/images"
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231211_test_public_model/rosbag2_2023_12_06-11_27_01/images"
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231211_test_public_model/rosbag2_2023_12_06-11_44_37/images"

        # 楼梯灯光数据
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231214/rosbag2_2023_12_13-20_44_43_human_control/images"

        # 2023-12-16
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231217/rosbag2_2023_12_16-16_28_19_dyg/images"
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231217/rosbag2_2023_12_16-16_28_19_dyg/select_images"

        # 1M相机
        # test_root = "/home/liyongjing/Downloads/1M样品实景原图"

        # 夜晚场景
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231219/rosbag2_2023_12_13-17_32_03/images"

        # 2号场地出现误检
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231220/rosbag2_2023_12_20-17_52_01_2号草坪/images"

        # 3号场地夜晚数据
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231220/rosbag2_2023_12_20-16_38_39/images"

        # 土堆边界线
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231225/planning_bug/images"

        # 2号场地阴影
        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231226/rosbag2_2023_12_25-10_00_23/images"

        # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231227/2023-12-26_16-15-14_planning_jincaodui/images"

        # test_root = "/home/liyongjing/Downloads/front_camera_rect"

        # 排查重复点的问题
        test_root = "/home/liyongjing/Downloads/front_camera"
        print(test_root)

        save_root = test_root + "_vis"
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.mkdir(save_root)

        vis_select_root = test_root + "_vis_select"
        if os.path.exists(vis_select_root):
            shutil.rmtree(vis_select_root)
        os.mkdir(vis_select_root)

        img_names = [name for name in os.listdir(test_root) if name[-4:] in ['.jpg', '.png', '.bmp']]
        jump = 1
        count = 0
        for img_name in tqdm(img_names):
            # img_name = img_names[1308]
            # print(test_root, img_name)
            count = count + 1
            if count % jump != 0:
                continue

            # img_name = "1695031549185622139.jpg"
            # img_name = "1696991348.057879.jpg"
            # img_name = "1699840777.207751.jpg"  # 井盖的问题

            path_0 = os.path.join(test_root, img_name)
            # path_0 = "/home/liyongjing/Downloads/1/1700550042.691849.jpg"
            # path_0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.jpg"
            # path_0 = "/media/dell/Elements SE/liyj/data/collect_data_20230720/rosbag2_2023_07_20-18_24_24/glass_edge_overfit_20230721/1689848678804013195.jpg"
            # path_0 = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230728_mmdet3d/test/images/1689848678804013195.jpg"
            # path_0 = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230728_mmdet3d/train/images/1689848680876640844.jpg"

            img_paths = [path_0]
            batch_data = []
            for img_path in img_paths:    # 这里可以是列表,然后推理采用batch的形式
                data_info = {}
                data_info['img_path'] = img_path
                data_input = test_pipeline(data_info)
                batch_data.append(data_input)

            # 这里设置为单个数据, 可以是batch的形式
            collate_data = pseudo_collate(batch_data)

            # forward the model
            with torch.no_grad():
                results = model.test_step(collate_data)
                # 得到输入图像
                # collate_data = model.data_preprocessor(collate_data, False)
                # results = model.predict(batch_inputs=collate_data["inputs"],
                #                         batch_data_samples=collate_data["data_samples"], rescale=True)

                for batch_id, result in enumerate(results):
                    img_origin = cv2.imread(img_paths[batch_id])

                    # img = collate_data["inputs"]["img"][batch_id]
                    # img = img.cpu().detach().numpy()
                    # img = np.transpose(img, (1, 2, 0))
                    # mean = np.array([103.53, 116.28, 123.675, ])
                    # std = np.array([1.0, 1.0, 1.0, ])

                    # 模型的内部进行归一化的处理data_preprocessor,从dataset得到的数据实际是未处理前的
                    # 如果是从result里拿到的img,则需要进行这样的还原
                    # img = img * std + mean
                    # img = img.astype(np.uint8)

                    stages_result = result.pred_instances.stages_result[0]
                    # stages_result = result.pred_instances.stages_result[1]
                    # meta_info = result.metainfo
                    # batch_input_shape = meta_info["batch_input_shape"]
                    # pred_line_map = np.zeros(batch_input_shape, dtype=np.uint8)

                    single_stage_result = stages_result[0]
                    for i, curve_line in enumerate(single_stage_result):
                        # if i != 4:
                        #     continue

                        curve_line = np.array(curve_line)

                        poitns_cls = curve_line[:, 4]

                        # line_cls = np.argmax(np.bincount(poitns_cls.astype(np.int32)))
                        # points[:, 4] = cls

                        point_num = len(curve_line)
                        pre_point = curve_line[0]

                        line = curve_line[:, :2]
                        # line = np.concatenate([line, line[0:1]], axis=0)
                        # line_set = np.array(list(set(tuple(p) for p in line)))
                        # print(line.shape[0], line_set.shape[0])
                        # if line.shape[0] != line_set.shape[0]:
                        #     print(i)
                        #     print("Fuck")
                        #     exit(1)

                        # line_cls = pre_point[4]
                        # color = color_list[int(line_cls)]
                        # x1, y1 = int(pre_point[0]), int(pre_point[1])

                        # color = [0, 0, 255]
                        # cv2.circle(img_origin, (int(x1), int(y1)), 10, (0, 0, 255), -1)

                        for i, cur_point in enumerate(curve_line[1:]):
                            x1, y1 = int(pre_point[0]), int(pre_point[1])
                            x2, y2 = int(cur_point[0]), int(cur_point[1])

                            if len(pre_point) >= 9:
                                point_visible = pre_point[6]
                                point_hanging = pre_point[7]
                                point_covered = pre_point[8]
                            else:
                                point_visible = -1
                                point_hanging = -1
                                point_covered = -1
                            # print("point_covered:", point_covered)

                            point_cls = pre_point[4]
                            color = color_list[int(point_cls)]

                            thickness = 3
                            cv2.line(img_origin, (x1, y1), (x2, y2), color, thickness, 8)
                            line_orient = cal_points_orient(pre_point, cur_point)

                            if -1 not in [point_visible, point_covered]:
                                if point_visible < 0.2 and point_covered < 0.2:
                                    cv2.circle(img_origin, (x2, y2), thickness * 2, (0, 0, 0), thickness=2)

                            if i % 50 == 0:
                                orient = pre_point[5]
                                if orient != -1:
                                    reverse = False   # 代表反向是否反了
                                    orient_diff = abs(line_orient - orient)
                                    if orient_diff > 180:
                                        orient_diff = 360 - orient_diff

                                    if orient_diff > 90:
                                        reverse = True

                                    # color = (0, 255, 0)
                                    # if reverse:
                                    #     color = (0, 0, 255)

                                    # 绘制预测的方向
                                    # 转个90度,指向草地
                                    orient = orient + 90
                                    if orient > 360:
                                        orient = orient - 360

                                    # img_origin = draw_orient(img_origin, pre_point, orient, arrow_len=30, color=color)

                            if i == point_num//2:
                                line_orient = line_orient + 90
                                if line_orient > 360:
                                    line_orient = line_orient - 360
                                img_origin = draw_orient(img_origin, pre_point, line_orient, arrow_len=50, color=color)

                            pre_point = cur_point

                    s_img_path = os.path.join(save_root, img_name)

                    if img_origin is not None:
                        img_h, img_w, _ = img_origin.shape
                        img_origin = cv2.resize(img_origin, (img_w//2, img_h//2))
                        cv2.imwrite(s_img_path, img_origin)


                    plt.imshow(img_origin[:, :, ::-1])
                    plt.show()
                    exit(1)

            # print("ff")


if __name__ == "__main__":
    # 进行模型预测, 给出图片的路径, 即可进行模型预测
    print("Start")
    main()
    print("End")