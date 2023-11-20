# 用来调试生成模型

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import cv2
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
import torch

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
import numpy as np
from mmdet3d.utils import replace_ceph_backend


#----------------------------------------------调试模型构建----------------------------------------------
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Sequence, Union
import copy
import shutil
from functools import partial
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.dist import (broadcast, get_dist_info, get_rank, init_dist,
                           is_distributed, master_only)
from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import TORCH_VERSION
from mmdet3d.apis import init_model
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from mmengine.dataset import Compose, pseudo_collate
from debug_utils import decode_gt_lines, cal_points_orient, draw_orient, filter_near_same_points, parse_ann_info, parse_ann_infov2


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


from gbld_mono2d_decode_numpy import GlasslandBoundaryLine2DDecodeNumpy


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def draw_pred_result(img_show, results):
    stages_result = results.pred_instances.stages_result[0]
    meta_info = results.metainfo
    batch_input_shape = meta_info["batch_input_shape"]

    pred_line_map = np.zeros(batch_input_shape, dtype=np.uint8)
    single_stage_result = stages_result[0]
    thickness = 1
    color = (0, 0, 255)

    for curve_line in single_stage_result:
        curve_line = np.array(curve_line)
        pre_point = curve_line[0]

        line_cls = pre_point[4]
        # color = color_list[int(line_cls)]
        # x1, y1 = int(pre_point[0]), int(pre_point[1])

        point_num = len(curve_line)
        for i, cur_point in enumerate(curve_line[1:]):
            x1, y1 = int(pre_point[0]), int(pre_point[1])
            x2, y2 = int(cur_point[0]), int(cur_point[1])

            cv2.line(pred_line_map, (x1, y1), (x2, y2), (1), thickness, 8)
            cv2.line(img_show, (x1, y1), (x2, y2), color, thickness, 8)

            line_orient = cal_points_orient(pre_point, cur_point)

            if i % 40 == 0:
                orient = pre_point[5]
                if orient != -1:
                    reverse = False  # 代表反向是否反了
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

                    # img_show = draw_orient(img_show, pre_point, orient, arrow_len=30, color=color)

            if i == point_num // 2:
                line_orient = line_orient + 90
                if line_orient > 360:
                    line_orient = line_orient - 360
                img_show = draw_orient(img_show, pre_point, line_orient, arrow_len=50, color=color)

            pre_point = cur_point
    # plt.imshow(img_show[:, :, ::-1])
    # # plt.imshow(pred_line_map)
    # plt.show()
    # exit(1)
    return img_show


def draw_pred_result_numpy(img_show, single_stage_result):
    # stages_result = results.pred_instances.stages_result[0]
    # meta_info = results.metainfo
    # batch_input_shape = meta_info["batch_input_shape"]
    #
    # pred_line_map = np.zeros(batch_input_shape, dtype=np.uint8)
    # single_stage_result = stages_result[0]
    thickness = 1
    color = (0, 0, 255)

    for curve_line in single_stage_result:
        curve_line = np.array(curve_line)

        pre_point = curve_line[0]
        line_cls = pre_point[4]
        # color = color_list[int(line_cls)]
        # x1, y1 = int(pre_point[0]), int(pre_point[1])

        point_num = len(curve_line)
        for i, cur_point in enumerate(curve_line[1:]):
            x1, y1 = int(pre_point[0]), int(pre_point[1])
            x2, y2 = int(cur_point[0]), int(cur_point[1])

            # cv2.line(pred_line_map, (x1, y1), (x2, y2), (1), thickness, 8)
            cv2.line(img_show, (x1, y1), (x2, y2), color, thickness, 8)

            line_orient = cal_points_orient(pre_point, cur_point)

            if i % 40 == 0:
                orient = pre_point[5]
                if orient != -1:
                    reverse = False  # 代表反向是否反了
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

                    # img_show = draw_orient(img_show, pre_point, orient, arrow_len=30, color=color)

            if i == point_num // 2:
                line_orient = line_orient + 90
                if line_orient > 360:
                    line_orient = line_orient - 360
                img_show = draw_orient(img_show, pre_point, line_orient, arrow_len=50, color=color)

            pre_point = cur_point
    # plt.imshow(img_show[:, :, ::-1])
    # # plt.imshow(pred_line_map)
    # plt.show()
    # exit(1)
    return img_show


def draw_gt_result(meta_info):
    batch_input_shape = meta_info["batch_input_shape"]
    gt_line_map = np.zeros(batch_input_shape, dtype=np.uint8)

    # 得到gt-map
    if "gt_lines" in meta_info:
        gt_lines = meta_info["gt_lines"]
        # gt_lines = meta_info["eval_gt_lines"]
        for gt_line in gt_lines:
            gt_label = gt_line["label"]
            points = gt_line["points"]
            category_id = gt_line["category_id"]

            pre_point = points[0]
            for cur_point in points[1:]:
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                thickness = 3
                cv2.line(gt_line_map, (x1, y1), (x2, y2), (1), thickness, 8)
                pre_point = cur_point

    return gt_line_map


def heatmap_nms(seg_pred):
    max_pooling = torch.nn.MaxPool2d((3, 3), stride=(1, 1), padding=[1, 1])
    max_pooling_col = torch.nn.MaxPool2d((3, 1), stride=(1, 1), padding=[1, 0])
    max_pooling_row = torch.nn.MaxPool2d((1, 3), stride=(1, 1), padding=[0, 1])
    max_pooling_dilate = torch.nn.MaxPool2d([3, 3], stride=1, padding=[1, 1])  # 去锯齿


    seg_max_pooling_col = max_pooling_col(seg_pred)
    seg_max_pooling_row = max_pooling_row(seg_pred)
    seg_max_pooling = max_pooling(seg_pred)

    mask_col = seg_pred == seg_max_pooling_col
    mask_row = seg_pred == seg_max_pooling_row
    mask_row_row = seg_pred == seg_max_pooling
    mask = torch.bitwise_or(mask_col, mask_row)
    # seg_pred[~mask] = -1e6
    # seg_pred[~mask_row_row] = -1e6
    # seg_pred[~mask_col] = -1e6
    # seg_pred[~mask_row] = -1e6
    # seg_pred = max_pooling_dilate(seg_pred)

    return seg_pred


def debug_model_infer_decode(config_path, checkpoint_path):
    # test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231013_mmdet3d/train/images"
    # test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231017_mmdet3d_debug/train/images"
    # test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231017_mmdet3d_debug/test/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231023/rosbag2_2023_10_11-10_07_59/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231102_mmdet3d_spline/test/images"

    # lwj-2023-10-25
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231023/rosbag2_2023_10_11-10_07_59/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231023/rosbag2_2023_10_11-10_13_56/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231023/rosbag2_2023_10_11-10_24_14/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231023/rosbag2_2023_10_11-10_28_36/images"

    # 采集车采集2023-11-08
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_06_57_yanbian/images"

    # 公司对面草地采集
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231110/rosbag2_2023_11_10-11_44_23/images_camera_front"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_06_57_yanbian/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231113/rosbag2_2023_11_13-09_53_33/images"

    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/select_test_20231114"
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231110_mmdet3d_spline_by_cls/test/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231110_mmdet3d_spline_by_cls/test/images"

    # 树边带坑
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231114/rosbag2_2023_11_14-16_11_14/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231110_mmdet3d_spline_by_cls/test/images"

    test_root = "/home/liyongjing/Egolee/hdd-data/test_data/select_test_20231117"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/debug"
    print(test_root)
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    cfg = model.cfg

    test_pipeline = deepcopy(cfg.val_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)

    save_root = test_root + "_debug_decode"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)
    img_names = [name for name in os.listdir(test_root) if name[-4:] in ['.jpg', '.png']]
    jump = 1
    count = len(img_names)

    glass_land_boundary_line_2d_decode_numpy = GlasslandBoundaryLine2DDecodeNumpy(confident_t=0.2)

    for img_name in tqdm(img_names, desc="img_names"):

        # img_name = "1696990080.166705.jpg"   # 由于聚类的效果出现漏检
        # img_name = "1696990080.699983.jpg"   # 由于聚类的效果出现漏检
        # img_name = "1696990082.499848.jpg"   # 由于聚类的效果出现漏检
        # img_name = "1696990108.297873.jpg"   # 在障碍的周围产生响应,区域生长出现问题
        # img_name = "1696990114.397417.jpg"   # 由于聚类的效果出现漏检
        # img_name = "1696990119.297038.jpg"   # 聚类效果还可以,但是由于分割效果不好而变得零散
        # img_name = "1696990121.196903.jpg"   # 聚类的问题，长的那段与下面那段归为一类，由于长度比较长从而而释放出来
        # img_name = "1696990179.092455.jpg"   # 距离近而且聚类连在一起,从而导致出现问题
        # img_name = "1696990188.258413.jpg"   # 分割的问题,但误识别部分的置信度比周围低,通过数据解决
        # img_name = "1696990195.691176.jpg"     # 分割的问题

        #测试集验证
        # img_name = "1689848691117702411.jpg"  # 远处距离聚类出错，而且更远的点比较靠近从而出现连接错误
        # img_name = "1689848788578806401.jpg"  # 聚类的问题
        # img_name = "1689848819889360075.jpg"  # 采用分类聚类的方式
        # img_name = "1695031238613012474.jpg"  # Heatmap变粗，垂直的时候区域生长的时候出现点顺序的问题
        # img_name = "c0bfbf61-2adf-40d3-85b1-ed9332a5fedb_front_camera_7181.jpg"  # 聚类、分割问题
        # img_name = "c0bfbf61-2adf-40d3-85b1-ed9332a5fedb_front_camera_7451.jpg"  # 聚类、分割问题
        # img_name = "dff47067-ede3-4ac0-b903-ef04df89a291_front_camera_9091.jpg"    # 分割问题

        # 将线段进行细化
        # img_name = "1695030883240994950.jpg"    # 灌木丛线
        # img_name = "c0bfbf61-2adf-40d3-85b1-ed9332a5fedb_front_camera_9201.jpg"   # 灌木丛线
        # img_name = "1689848713868127775.jpg"
        # img_name = "1695031405813262853.jpg"
        # img_name = "1695031238613012474.jpg"

        # 细化导致的问题
        # img_name = "c0bfbf61-2adf-40d3-85b1-ed9332a5fedb_front_camera_6761.jpg"   # 聚类效果问题
        # img_name = "c0bfbf61-2adf-40d3-85b1-ed9332a5fedb_front_camera_8731.jpg"   # 聚类效果问题
        # img_name = "dff47067-ede3-4ac0-b903-ef04df89a291_front_camera_7091.jpg"     # 聚类效果问题

        # 检查test数据集
        # img_name = "1689848734654320005.jpg"

        # lwj 验证
        # img_name = "1696990107.197989.jpg"
        # img_name = "1696990184.992007.jpg"
        # img_name = "1696990178.992459.jpg"
        # img_name = "1696990178.992459.jpg"
        # img_name = "1696990178.592509.jpg"
        # img_name = "1696990178.592509.jpg"      # 在远处的线会叠加到一起

        # 点排序的验证
        # img_name = "1695031549185622139.jpg"

        # 采集车采集 2023-11-08
        # img_name = "1699430863.116359.jpg"
        # img_name = "1699430900.64799.jpg"
        # img_name = "1699430840.150728.jpg"

        # 采集车公司对面采集
        # img_name = "1699587863.896535.jpg"
        # img_name = "1699430890.448459.jpg"

        # 分析圆形的检测效果不好的情况
        # img_name = "1699840642.113791.jpg"  # 地灯
        # img_name = "1699840777.207751.jpg"  # 井盖

        # img_name = "1696990178.592509.jpg"
        # img_name = "1699949529.037394.jpg"
        # img_name = "1699949526.737498.jpg"
        # img_name = "1699949527.637458.jpg"
        # img_name = "1699949526.737498.jpg"
        # img_name = "1699949527.637458.jpg"
        # img_name = "1699949528.837429.jpg"
        # img_name = "1699949529.437395.jpg"
        # img_name = "1699949664.932432.jpg"
        # img_name = "1699949722.730314.jpg"

        # 分析测试集的情况
        # img_name = "1689848734654320005.jpg"     # 分段，原因为中间小的piece-line过滤了
        # img_name = "1699949723.730288.jpg"

        # 分析树边带坑
        # img_name = "1699949477.739274.jpg"
        # img_name = "1699949682.731791.jpg"

        # 测试集
        # img_name = "1695031238613012474.jpg"
        # img_name = "1695030227549081836.jpg"    # 用了骨架细化会有后退

        # 20231117
        # img_name = "1699840777.207751.jpg"    # 井盖的问题
        # img_name = "1695031238613012474.jpg"  # 道路中间垂直
        img_name = "1699840593.682635.jpg"       # 道路边垂直

        glass_land_boundary_line_2d_decode_numpy.debub_emb = 1
        glass_land_boundary_line_2d_decode_numpy.debug_existing_points = 1
        glass_land_boundary_line_2d_decode_numpy.debug_piece_line = 0
        glass_land_boundary_line_2d_decode_numpy.debug_exact_line = 0

        # glass_land_boundary_line_2d_decode_numpy.debub_emb = False
        # glass_land_boundary_line_2d_decode_numpy.debug_piece_line = False

        count = count + 1
        if count % jump != 0:
            continue

        # img_name = "1695031543693612696.jpg"
        # img_name = "1696991118.675607.jpg"
        path_0 = os.path.join(test_root, img_name)

        # path_0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.jpg"

        img = cv2.imread(path_0)

        img_paths = [path_0]

        batch_data = []
        for img_path in img_paths:    # 这里可以是列表,然后推理采用batch的形式
            data_info = {}
            data_info['img_path'] = img_path

            ann_path = img_path.replace("images", "jsons")[:-4] + ".json"
            if os.path.exists(ann_path):
                data_info["ann_path"] = ann_path

            data_input = test_pipeline(data_info)
            batch_data.append(data_input)

        # 这里设置为单个数据, 可以是batch的形式
        collate_data = pseudo_collate(batch_data)

        # forward the model
        with torch.no_grad():
            # results = model.test_step(collate_data)
            bath_size = 0
            stages = 0

            # 得到输入图像
            collate_data = model.data_preprocessor(collate_data, False)
            # show pred result
            img = collate_data["inputs"]["imgs"][bath_size]
            img = img.cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            mean = np.array([103.53, 116.28, 123.675, ])
            std = np.array([1.0, 1.0, 1.0, ])

            img = img * std + mean
            img = img.astype(np.uint8)

            img_show = copy.deepcopy(img.copy())
            img_show2 = copy.deepcopy(img.copy())

            # 与gt进行计算losss
            if 0:
                loss = model.loss(batch_inputs=collate_data["inputs"],
                                     batch_data_samples=collate_data["data_samples"])
                all_loss = 0
                for _loss in loss.keys():
                    print(_loss, loss[_loss].detach().cpu().numpy())
                    all_loss = all_loss + loss[_loss].detach().cpu().numpy()
            else:
                all_loss = None

            if 1:
                # 得到预测解析后的结果
                results = model.predict(batch_inputs=collate_data["inputs"],
                                        batch_data_samples=collate_data["data_samples"], rescale=False)
                results = results[0]

                img_show = draw_pred_result(img_show, results)

                meta_info = results.metainfo
                gt_line_map = draw_gt_result(meta_info)
            else:
                img_show = None
                gt_line_map = None

            if 1:
                # 为多分辨率多尺度的预测结果,取最大的分辨率
                # dir_seg_pred, dir_offset_pred, dir_seg_emb_pred, dir_connect_emb_pred, \
                # dir_cls_pred, dir_orient_pred, dir_visible_pred, dir_hanging_pred, dir_covered_pred

                # 得到head的heatmap的结果,用于分析
                # dir_seg_pred, dir_offset_pred, dir_seg_emb_pred, dir_connect_emb_pred, dir_cls_pred, dir_orient_pred
                heatmap = model._forward(batch_inputs=collate_data["inputs"],
                                     batch_data_samples=collate_data["data_samples"])

                heatmap_map_seg = heatmap[0][stages].cpu().detach()

                # 进行heatmap的nms操作
                # heatmap_map_seg = heatmap_nms(heatmap_map_seg)

                heatmap_map_seg = heatmap_map_seg.numpy()[bath_size]

                heatmap_map_offset = heatmap[1][stages].cpu().detach().numpy()[bath_size]
                heatmap_map_seg_emb = heatmap[2][stages].cpu().detach().numpy()[bath_size]
                heatmap_map_connect_emb = heatmap[3][stages].cpu().detach().numpy()[bath_size]

                heatmap_map_cls = heatmap[4][stages].cpu().detach().numpy()[bath_size]
                # heatmap_map_cls = heatmap_map_cls[3]  # 选择不同类别的heatmap

                heatmap_map_orient = heatmap[5][stages].cpu().detach().numpy()[bath_size]
                heatmap_map_visible = heatmap[6][stages].cpu().detach().numpy()[bath_size]
                heatmap_map_hanging = heatmap[7][stages].cpu().detach().numpy()[bath_size]
                heatmap_map_covered = heatmap[8][stages].cpu().detach().numpy()[bath_size]


                curse_lines_with_cls = glass_land_boundary_line_2d_decode_numpy.forward(heatmap_map_seg,
                                                                                        heatmap_map_offset,
                                                                                        heatmap_map_seg_emb,
                                                                                        heatmap_map_connect_emb,
                                                                                        heatmap_map_cls,
                                                                                        orient_pred=heatmap_map_orient,
                                                                                        visible_pred=heatmap_map_visible,
                                                                                        hanging_pred=heatmap_map_hanging,
                                                                                        covered_pred=heatmap_map_covered,)



                img_show2 = draw_pred_result_numpy(img_show2, curse_lines_with_cls[0])

                heatmap_map_seg = heatmap_map_seg[0]
                heatmap_map_offset = heatmap_map_offset[0]
                heatmap_map_seg_emb = heatmap_map_seg_emb[0]
                heatmap_map_connect_emb = heatmap_map_connect_emb[0]
                heatmap_map_visible = heatmap_map_visible[0]
                heatmap_map_hanging = heatmap_map_hanging[0]
                heatmap_map_covered = heatmap_map_covered[0]

                heatmap_map_seg = sigmoid(heatmap_map_seg)
                heatmap_map_cls = sigmoid(heatmap_map_cls)
                heatmap_map_visible = sigmoid(heatmap_map_visible)
                heatmap_map_hanging = sigmoid(heatmap_map_hanging)
                heatmap_map_covered = sigmoid(heatmap_map_covered)

                heatmap_map_cls = np.argmax(heatmap_map_cls, axis=0)

            else:
                heatmap_map_seg = None
                heatmap_map_offset = None
                heatmap_map_seg_emb = None
                heatmap_map_connect_emb = None
                heatmap_map_cls = None
                heatmap_map_orient = None
                heatmap_map_visible = None
                heatmap_map_hanging = None
                heatmap_map_covered = None

            # plt.subplot(2, 1, 1)
            # if img_show is not None:
            #     plt.imshow(img_show[:, :, ::-1])
            #
            # plt.subplot(2, 1, 2)
            # if img_show2 is not None:
            #     plt.imshow(img_show2[:, :, ::-1])

            # show result
            plt.subplot(2, 3, 1)
            if img_show is not None:
                plt.imshow(img_show[:, :, ::-1])

            plt.subplot(2, 3, 2)
            if img_show2 is not None:
                plt.imshow(img_show2[:, :, ::-1])

            plt.subplot(2, 3, 3)
            if gt_line_map is not None:
                plt.imshow(gt_line_map)

            plt.subplot(2, 3, 4)
            # plt.imshow(pred_line_map)
            plt.imshow(heatmap_map_seg)

            plt.subplot(2, 3, 5)
            # plt.imshow(heatmap_map_emb)
            # plt.imshow(heatmap_map_cls)
            # plt.imshow(heatmap_map_covered)
            plt.imshow(heatmap_map_seg_emb)

            if all_loss is not None:
                s_name = "loss_" + str(round(all_loss, 2)) + "_" + img_name
            else:
                s_name = img_name

            s_path = os.path.join(save_root, s_name)
            # plt.savefig(s_path, dpi=300)
            # plt.show()
            print(s_path)
            # cv2.imwrite(s_path, img_show)
            cv2.imwrite(s_path, img_show2)
            plt.subplot(3, 1, 1)
            plt.imshow(img_show[:, :, ::-1])

            plt.subplot(3, 1, 2)
            plt.imshow(img_show2[:, :, ::-1])

            plt.subplot(3, 1, 3)
            plt.imshow(heatmap_map_seg_emb)
            plt.show()
            exit(1)


if __name__ == "__main__":
    print("Start")
    # 调试草地遮挡的情况
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered/gbld_debug_config_no_dcn_datasetv2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered/epoch_250.pth"

    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered5/gbld_debug_config_no_dcn_datasetv2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered5/epoch_250.pth"

    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231023_2/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231023_2/epoch_250.pth"

    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231024_batch_12_no_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231024_batch_12_no_crop_split_line_10_emb_weight/epoch_200.pth"

    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_crop/gbld_v0.3_20231024_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_crop/gbld_v0.3_20231024_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 2031026
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231026_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231026_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 20231031
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231031_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231031_batch_12_with_crop_split_line_10_emb_weight/epoch_100.pth"

    # 20231102
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231102_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231102_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 20231108
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls/epoch_250.pth"

    # 20231113
    config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/gbld_config_v0.3.py"
    checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls_stage3/epoch_250.pth"

    debug_model_infer_decode(config_path, checkpoint_path)
    print("End")