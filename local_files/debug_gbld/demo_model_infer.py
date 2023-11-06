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
    config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231102_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_v0.3_20231102_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    # 服务器
    # config_path = "/data-ssd2/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/on_server/gbld_v0.3_20231026_batch_12_with_crop_split_line_10_emb_weight/gbld_config_v0.3.py"
    # checkpoint_path = "/data-ssd2/liyj/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/on_server/gbld_v0.3_20231026_batch_12_with_crop_split_line_10_emb_weight/epoch_250.pth"

    model = init_model(config_path, checkpoint_path, device='cuda:0')
    # onnx output

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

    # 对测试集进行验证,这样比较容易发现问题（对验证集进行验证也更加容易发现问题）
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231025_mmdet3d_spline/test/images"

    # 20231031采集车
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231101/rosbag2_2023_10_31-16_09_00/camera_front_images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231101/rosbag2_2023_10_31-16_09_00/camera_front_left_images"

    # 测试近距离
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231104/rosbag2_camera_2023_11_04-17_06_47/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231104/rosbag2_camera_2023_11_04-17_07_30/images"
    # test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231104/rosbag2_camera_2023_11_04-17_09_30/images"
    test_root = "/home/liyongjing/Egolee/hdd-data/test_data/20231104/rosbag2_camera_2023_11_04-17_15_34/images"

    print(test_root)
    save_root = test_root + "_vis"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)

    img_names = [name for name in os.listdir(test_root) if name[-4:] in ['.jpg', '.png']]
    jump = 1
    count = 0
    for img_name in tqdm(img_names):
        print(img_name)
        count = count + 1
        if count % jump != 0:
            continue

        # img_name = "1695031543693612696.jpg"
        # img_name = "1696991348.057879.jpg"
        path_0 = os.path.join(test_root, img_name)
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
            # data_batch = data_preprocessor(data_batch)
            # results = model(collate_data["inputs"], collate_data["data_samples"], mode="predict")

            for batch_id, result in enumerate(results):
                img_origin = cv2.imread(img_paths[batch_id])

                img = collate_data["inputs"]["img"][batch_id]
                img = img.cpu().detach().numpy()
                img = np.transpose(img, (1, 2, 0))
                mean = np.array([103.53, 116.28, 123.675, ])
                std = np.array([1.0, 1.0, 1.0, ])

                # 模型的内部进行归一化的处理data_preprocessor,从dataset得到的数据实际是未处理前的
                # 如果是从result里拿到的img,则需要进行这样的还原
                # img = img * std + mean
                img = img.astype(np.uint8)

                stages_result = result.pred_instances.stages_result[0]
                # meta_info = result.metainfo
                # batch_input_shape = meta_info["batch_input_shape"]
                # pred_line_map = np.zeros(batch_input_shape, dtype=np.uint8)

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
                        # print("point_covered:", point_covered)

                        point_cls = pre_point[4]
                        color = color_list[int(point_cls)]

                        thickness = 3
                        cv2.line(img_origin, (x1, y1), (x2, y2), color, thickness, 8)
                        line_orient = cal_points_orient(pre_point, cur_point)

                        if point_visible < 0.2:
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
                # plt.imshow(img_origin[:, :, ::-1])
                # plt.show()
                # exit(1)

        # print("ff")


if __name__ == "__main__":
    # 进行模型预测, 给出图片的路径, 即可进行模型预测
    print("Start")
    main()
    print("End")