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
from tools_utils import color_list, decode_gt_lines, cal_points_orient, draw_orient, filter_near_same_points


def main(config_path, checkpoint_path, test_root, b_save_vis=False, s_vis_root=None, device='cuda:0'):
    model = init_model(config_path, checkpoint_path, device=device)
    # 采用官方的方式进行推理
    cfg = model.cfg
    # build the data pipeline
    test_pipeline = deepcopy(cfg.val_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)

    if b_save_vis:
        if s_vis_root is None:
            save_root = test_root + "_vis"
        else:
            save_root = s_vis_root
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.mkdir(save_root)

    img_names = [name for name in os.listdir(test_root) if name[-4:] in ['.jpg', '.png']]
    jump = 1
    count = 0
    for img_name in tqdm(img_names):
        # img_name = "1699432607.004302.jpg"
        # print(img_name)
        count = count + 1
        if count % jump != 0:
            continue

        path_0 = os.path.join(test_root, img_name)
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

            for batch_id, result in enumerate(results):
                img_origin = cv2.imread(img_paths[batch_id])

                stages_result = result.pred_instances.stages_result[0]
                single_stage_result = stages_result[0]
                for curve_line in single_stage_result:
                    curve_line = np.array(curve_line)

                    poitns_cls = curve_line[:, 4]

                    # line_cls = np.argmax(np.bincount(poitns_cls.astype(np.int32)))
                    # points[:, 4] = cls

                    point_num = len(curve_line)
                    pre_point = curve_line[0]

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

                        point_cls = pre_point[4]
                        color = color_list[int(point_cls) % len(color_list)]

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

                if b_save_vis:
                    s_img_path = os.path.join(save_root, img_name)
                    if img_origin is not None:
                        img_h, img_w, _ = img_origin.shape
                        img_origin = cv2.resize(img_origin, (img_w//2, img_h//2))
                        cv2.imwrite(s_img_path, img_origin)

                # plt.imshow(img_origin[:, :, ::-1])
                # plt.show()
                # exit(1)    # show one image

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--test_root', help='train config file path')
    parser.add_argument('--s_vis_root', help='the dir to save logs and models')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()


    # 进行模型预测, 给出图片的路径, 即可进行模型预测
    print("Start")
    # 20231108
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_split_by_cls/epoch_200.pth"

    # 20240104
    # config_path = "/data-ssd2/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/on_labels/gbld_v0.4_20231218_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/gbld_config_v0.4_overfit.py"
    # checkpoint_path = "/data-ssd2/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/on_labels/gbld_v0.4_20231218_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/epoch_250.pth"

    config_path = "/data-ssd2/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/on_server/normal/gbld_v0.4_20240102_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_discrimate_rotate/gbld_config_v0.4_overfit.py"
    checkpoint_path = "/data-ssd2/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/on_server/normal/gbld_v0.4_20240102_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_discrimate_rotate/epoch_250.pth"

    print("config_path:", config_path)
    print("checkpoint_path:", checkpoint_path)
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231109/rosbag2_2023_11_08-16_29_19_gongzidaolu/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231217/rosbag2_2023_12_16-16_28_19_dyg/images"
    # s_vis_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231217/rosbag2_2023_12_16-16_28_19_dyg/1"

    test_root = args.test_root
    s_vis_root = args.s_vis_root
    print("test_root:", test_root)
    print("s_vis_root:", s_vis_root)

    main(config_path, checkpoint_path, test_root, b_save_vis=True, s_vis_root=s_vis_root, device='cuda:0')
    print("End")



