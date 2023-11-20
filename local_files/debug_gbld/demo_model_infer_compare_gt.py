# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
from mmengine.dataset import Compose, pseudo_collate
import copy
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
from debug_utils import decode_gt_lines, cal_points_orient, draw_orient, filter_near_same_points, parse_ann_infov2
from debug_utils_vis import draw_gbld_pred, draw_gbld_lines_on_image
from debug_utils_metric import GBLDMetricDebug


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]


def main():
    # 将seg-emg weight提升为10,采用split-line的方式
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231024_batch_12_no_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231024_batch_12_no_crop_split_line_10_emb_weight/epoch_200.pth"

    # 20231025
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_crop/gbld_v0.3_20231024_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_crop/gbld_v0.3_20231024_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 20231026
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231026_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231026_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 20231030
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231030_batch_12_with_crop_split_line_15_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231030_batch_12_with_crop_split_line_15_emb_weight/epoch_200.pth"

    # 20231031
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231031_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231031_batch_12_with_crop_split_line_10_emb_weight/epoch_100.pth"

    # 20231102
    # config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231102_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231102_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 20231115
    config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_stage2/gbld_config_v0.3.py"
    checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231108_batch_12_with_crop_line_10_emb_weight_stage2/epoch_200.pth"


    model = init_model(config_path, checkpoint_path, device='cuda:0')
    # onnx output

    # 采用官方的方式进行推理
    cfg = model.cfg
    # build the data pipeline
    test_pipeline = copy.deepcopy(cfg.val_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)

    # 进行kpi的测评
    gbld_metric = GBLDMetricDebug(test_line_thinkness=60, test_t_iou=0.5)
    # gbld_metric = GBLDMetricDebug(test_line_thinkness=60, test_t_iou=0.3)

    # 输入形式为数据集的路径
    # 对测试集进行验证,这样比较容易发现问题（对验证集进行验证也更加容易发现问题）
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231025_mmdet3d_spline/test/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231031_mmdet3d_spline/test/images"
    test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231110_mmdet3d_spline_by_cls/test/images"

    print(test_root)

    save_root = test_root + "_compare_gt"
    # save_root = test_root + "_compare_gt_debug"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)

    img_names = [name for name in os.listdir(test_root) if name[-4:] in ['.jpg', '.png']]
    jump = 1
    count = 0
    for img_name in tqdm(img_names):
        # img_name = "c0bfbf61-2adf-40d3-85b1-ed9332a5fedb_front_camera_5591.jpg"
        # img_name = "1696991193.66982.jpg"
        # img_name = "c0bfbf61-2adf-40d3-85b1-ed9332a5fedb_front_camera_6481.jpg"
        # img_name = "1695031384927132524.jpg"
        print(img_name)
        count = count + 1
        if count % jump != 0:
            continue

        path_0 = os.path.join(test_root, img_name)
        ann_path_0 = path_0.replace("images", "jsons")[:-4] + ".json"
        # path_0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.jpg"

        img = cv2.imread(path_0)
        img_pred_show = copy.deepcopy(img)
        img_pred_ap_show = copy.deepcopy(img)
        img_gt_show = copy.deepcopy(img)
        img_gt_ap_show = copy.deepcopy(img)

        # 模型预测
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
            # data_batch = data_preprocessor(data_batch)
            # results = model(collate_data["inputs"], collate_data["data_samples"], mode="predict")

            for batch_id, result in enumerate(results):
                stages_result = result.pred_instances.stages_result[0]
                single_stage_result = stages_result[0]
                img_pred_show = draw_gbld_pred(img_pred_show, single_stage_result)

        # gt的结果
        info = {"img_path": path_0,
                "ann_path": ann_path_0,
                }
        classes = [
            'road_boundary_line',
            'bushes_boundary_line',
            'fence_boundary_line',
            'stone_boundary_line',
            'wall_boundary_line',
            'water_boundary_line',
            'snow_boundary_line',
            'manhole_boundary_line',
            'others_boundary_line',
        ]

        ann_info = parse_ann_infov2(info, classes=classes)
        gt_lines = ann_info["gt_lines"]

        all_points = []
        all_points_type = []
        all_points_visible = []
        all_points_hanging = []
        all_points_covered = []
        for gt_line in gt_lines:
            all_points.append(gt_line['points'])
            all_points_type.append(gt_line['points_type'])
            all_points_visible.append(gt_line['points_visible'])
            all_points_hanging.append(gt_line['points_hanging'])
            all_points_covered.append(gt_line['points_covered'])

        show_img_line_gt, show_img_cls_gt, show_img_vis_gt, \
        show_img_hang_gt, show_img_covered_gt = draw_gbld_lines_on_image(img_gt_show, all_points,
                                                                         all_points_type,
                                                                         all_points_visible,
                                                                         all_points_hanging,
                                                                         all_points_covered,
                                                                         )

        img_gt_show = show_img_cls_gt

        # 进行kpi的计算
        result_list = [{'stages_pred_result': stages_result, 'eval_gt_lines': gt_lines,
                        'batch_input_shape': img.shape[:2], 'sample_idx':0}]

        metric_f1_line_pixel, metric_f1_line_instance = gbld_metric.gbld_evaluate_line_pixel(result_list=result_list, classes=classes)
        print(metric_f1_line_instance)
        gt_lines_points = [gt_line["points"] for gt_line in gt_lines]
        pred_ap_list = gbld_metric.get_measure_pred_lines_ap(single_stage_result, gt_lines_points, img.shape[:2])
        # 将pred和gt的顺序调转
        gt_ap_list = gbld_metric.get_measure_pred_lines_ap(gt_lines_points, single_stage_result, img.shape[:2])

        pred_colors = []
        for pred_ap in pred_ap_list:
            pred_colors.append((0, 255, 0) if pred_ap else (0, 0, 255))
        img_pred_ap_show = draw_gbld_pred(img_pred_ap_show, single_stage_result, colors=pred_colors)


        gt_colors = []
        for gt_ap in gt_ap_list:
            gt_colors.append((0, 255, 0) if gt_ap else (0, 0, 255))
        img_gt_ap_show = draw_gbld_pred(img_gt_ap_show, gt_lines_points, colors=gt_colors)



        s_img_path = os.path.join(save_root, img_name)
        # img_save = cv2.hconcat([img_pred_ap_show, img_gt_show])
        img_save = cv2.hconcat([img_pred_ap_show, img_gt_ap_show])
        cv2.imwrite(s_img_path, img_save)

        # plt.subplot(3, 1, 1)
        # plt.imshow(img_pred_show[:, :, ::-1])
        #
        # plt.subplot(3, 1, 2)
        # plt.imshow(img_gt_show[:, :, ::-1])
        #
        # plt.subplot(3, 1, 3)
        # plt.imshow(img_pred_ap_show[:, :, ::-1])
        # plt.show()
        #
        #
        # exit(1)

        # print("ff")


if __name__ == "__main__":
    # 进行模型预测, 给出图片的路径, 即可进行模型预测
    # 同时与gt进行对比
    print("Start")
    main()
    print("End")