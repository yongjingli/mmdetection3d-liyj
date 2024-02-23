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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():

    # 12-19  全量数据训练 + 行人数据 + 外边界遮挡 + 挑选corner-case数据
    config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231218_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/gbld_config_v0.4_overfit.py"
    checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.4_20231218_batch_24_with_crop_line_10_emb_weight_split_by_cls_stage3_length_filter_no_point_emb/epoch_250.pth"

    model = init_model(config_path, checkpoint_path, device='cuda:0')

    # 采用官方的方式进行推理
    cfg = model.cfg
    # build the data pipeline
    test_pipeline = deepcopy(cfg.val_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)


    test_roots = [1]
    for test_root in test_roots:
        # test_root = "/home/liyongjing/Downloads/front_camera_rect"
        # test_root = "/home/liyongjing/Egolee/hdd-data/debug_data/debug_generate_bev_line_gt/front_camera_rect"
        # test_root = "/home/liyongjing/Egolee/hdd-data/debug_data/debug_generate_bev_line_gt/front_left_camera_rect"
        test_root = "/home/liyongjing/Egolee/hdd-data/debug_data/debug_generate_bev_line_gt/front_right_camera_rect"
        print(test_root)

        save_root = test_root + "_edge_mask"
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.mkdir(save_root)

        img_names = [name for name in os.listdir(test_root) if name[-4:] in ['.jpg', '.png', '.bmp']]
        jump = 1
        count = 0

        for img_name in tqdm(img_names):
            print(test_root, img_name)
            count = count + 1
            if count % jump != 0:
                continue

            path_0 = os.path.join(test_root, img_name)
            # path_0 = "/home/liyongjing/Downloads/1/1700550042.691849.jpg"

            img_paths = [path_0]
            batch_data = []
            for img_path in img_paths:    # 这里可以是列表,然后推理采用batch的形式
                data_info = {}
                data_info['img_path'] = img_path
                data_input = test_pipeline(data_info)
                batch_data.append(data_input)

            # 这里设置为单个数据, 可以是batch的形式
            collate_data = pseudo_collate(batch_data)
            img = cv2.imread(img_paths[0])
            img_h, img_w, _ = img.shape

            # forward the model
            with torch.no_grad():
                # results = model.test_step(collate_data)
                collate_data = model.data_preprocessor(collate_data, False)
                heatmap = model._forward(batch_inputs=collate_data["inputs"],
                                     batch_data_samples=collate_data["data_samples"])

                heatmap_map_seg = heatmap[0][0].cpu().detach()
                heatmap_map_seg = heatmap_map_seg.numpy()[0]
                heatmap_map_seg = heatmap_map_seg[0]
                heatmap_map_seg = sigmoid(heatmap_map_seg)
                heatmap_map_seg = heatmap_map_seg > 0.2
                heatmap_map_seg = heatmap_map_seg.astype(np.uint8)

                heatmap_map_seg = cv2.resize(heatmap_map_seg, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                s_img_mask_path = os.path.join(save_root, img_name)
                cv2.imwrite(s_img_mask_path, heatmap_map_seg)

                # plt.subplot(2, 1, 1)
                # plt.imshow(img[:, :, ::-1])
                #
                # plt.subplot(2, 1, 2)
                # plt.imshow(heatmap_map_seg)
                # plt.show()
                # exit(1)


def check_mask():
    test_root = "/home/liyongjing/Egolee/hdd-data/debug_data/debug_generate_bev_line_gt/front_camera_rect"
    save_root = test_root + "_edge_mask"

    img_names = [name for name in os.listdir(test_root) if name[-4:] in ['.jpg', '.png', '.bmp']]
    for img_name in tqdm(img_names, desc="img_names"):
        img_path = os.path.join(test_root, img_name)
        mask_img_path = os.path.join(save_root, img_name)

        img = cv2.imread(img_path)
        img_mask = cv2.imread(mask_img_path, 0)
        img[img_mask > 0] = (255, 0, 0)

        plt.subplot(2, 1, 1)
        plt.imshow(img[:, :, ::-1])

        plt.subplot(2, 1, 2)
        plt.imshow(img_mask)

        plt.show()
        exit(1)



if __name__ == "__main__":
    # 进行模型预测, 给出图片的路径, 即可进行模型预测
    print("Start")
    main()
    # check_mask()
    print("End")