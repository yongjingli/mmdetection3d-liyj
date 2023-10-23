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


def debug_model_infer_decode(config_path, checkpoint_path):
    # test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231013_mmdet3d/train/images"
    test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231017_mmdet3d_debug/train/images"
    # test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231017_mmdet3d_debug/test/images"
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
        # img_name = "1696991348.45783.jpg"
        # img_name = "1696991333.15902.jpg"
        img_name = "1696991118.675607.jpg"
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
                # heatmap_map_seg = torch.sigmoid(heatmap[0][stages].cpu().detach())
                # noise_filter = torch.nn.MaxPool2d([3, 1], stride=1, padding=[1, 0])  # 去锯齿
                # seg_max_pooling = noise_filter(heatmap_map_seg)
                # mask = heatmap_map_seg == seg_max_pooling
                # heatmap_map_seg[~mask] = -1e6
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
            cv2.imwrite(s_path, img_show2)
            plt.subplot(1, 1, 1)
            plt.imshow(img_show2[:, :, ::-1])
            plt.show()
            exit(1)


if __name__ == "__main__":
    print("Start")
    # 调试草地遮挡的情况
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered/gbld_debug_config_no_dcn_datasetv2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered/epoch_250.pth"

    config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered5/gbld_debug_config_no_dcn_datasetv2.py"
    checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered5/epoch_250.pth"

    debug_model_infer_decode(config_path, checkpoint_path)
    print("End")