import torch
import os
import copy
import mmdet.models.necks.fpn as FPN
import mmengine
from mmengine.fileio import join_path, list_from_file, load
import numpy as np
import cv2
import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

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
palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 215, 255), (0, 255, 255),
            (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238)]


def debug_write_data_infos():
    metainfo = dict()
    metainfo['dataset'] = 'nuscenes'
    metainfo['info_version'] = '1.1'
    converted_list = []
    for i in range(5):
        temp_data_info = str(i)
        converted_list.append(temp_data_info)
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)
    out_path = "./tmp/22.pkl"
    mmengine.dump(converted_data_info, out_path, 'pkl')

    annotations = load(out_path)

    metainfo = annotations['metainfo']
    raw_data_list = annotations['data_list']

    print("metainfo:", metainfo)
    print("raw_data_list:", raw_data_list)


def test_transform_resize():
    print("fff")


def test_dtype():
    gt_confidence = np.zeros((1, 100, 100), dtype=np.float32)
    t_gt_confidence = torch.from_numpy(gt_confidence)
    print("ffff")

def debug_sum():
    a = torch.tensor(0.2)
    b = torch.tensor(0.1)
    print(sum([a, b]))
    from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop


    from mmengine.runner.loops import ValLoop

def debug_padd():
    import mmcv

    img_path = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230728_mmdet3d/test/images/1689848740939986547.jpg"
    img = cv2.imread(img_path)
    results = {}
    results['img'] = img
    size = (img.shape[0] + 20, img.shape[1] + 20)
    padded_img = mmcv.impad(
        results['img'],
        shape=size,
        pad_val=(0, 0, 0),
        padding_mode='constant')

    plt.imshow(padded_img[:, :, ::-1])
    plt.show()


def debug_parse_label_json():
    root = "/media/dell/Egolee1/liyj/data/label_data/gbld_from_label_system/gbld_20231012_v0.2/all"
    dst_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231013_mmdet3d/debug"
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    img_root = os.path.join(root, "images")
    label_root = os.path.join(root, "json")

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", '.png']]
    for img_name in tqdm(img_names, desc="img_names"):
        # img_name = "dff47067-ede3-4ac0-b903-ef04df89a291_front_camera_821.jpg"  # 17
        # img_name = "dff47067-ede3-4ac0-b903-ef04df89a291_front_camera_461.jpg"  # 17
        # img_name = "1696991212.768321.jpg"
        # img_name = "1689848703511301286.jpg"
        # img_name = "1689848743135381664.jpg"
        # img_name = "1689848751402792956.jpg"
        # img_name = "1689848753489529712.jpg"
        # img_name = "1689848776280394106.jpg"
        # img_name = "1689848807256677791.jpg"
        # img_name = "1689848859777111978.jpg"
        # img_name = "1689848861843888811.jpg"
        # img_name = "1695030137658086822.jpg"
        # img_name = "1695030153329805898.jpg"
        # img_name = "1695030190107402825.jpg"
        # img_name = "1695030191324421119.jpg"
        # img_name = "1695030193582731768.jpg"
        # img_name = "1695030356811090637.jpg"
        # img_name = "1695030898755650041.jpg"
        # img_name = "1695031238613012474.jpg"
        img_name = "1695031389933174128.jpg"
        # print(img_name)

        img_path = os.path.join(img_root, img_name)
        json_path = os.path.join(label_root, img_name[:-4] + ".json")

        img = cv2.imread(img_path)

        with open(json_path, "r") as fp:
            labels = json.load(fp)

        lines = []
        all_points = []
        all_attr_type = []
        all_attr_visible = []
        all_attr_hanging = []
        lines_intersect_indexs = []
        for label in labels:
            intersect_index = np.array(label["intersect_index"])

            points = np.array(label["points"])
            attr_type = np.array(label["type"]).reshape(-1, 1)
            attr_visible = np.array(label["visible"]).reshape(-1, 1)
            attr_hanging = np.array(label["hanging"]).reshape(-1, 1)
            # print(points.shape, attr_type.shape, attr_visible.shape, attr_hanging.shape)

            line = np.concatenate([points, attr_type, attr_visible, attr_hanging], axis=1)

            lines.append(line)
            all_points.append(points)
            all_attr_type.append(attr_type)
            all_attr_visible.append(attr_visible)
            all_attr_hanging.append(attr_hanging)
            lines_intersect_indexs.append(intersect_index)

        # img_line = np.ones_like(img) * 255
        img_line = copy.deepcopy(img)
        img_cls = np.ones_like(img) * 255
        img_vis = np.ones_like(img) * 255
        img_hang = np.ones_like(img) * 255

        for points, attr_type, attr_visible, attr_hanging, lines_intersect_index in \
                zip(all_points, all_attr_type, all_attr_visible, all_attr_hanging,  lines_intersect_indexs):
            # 在lines_intersect_index的首尾添加0, -1的index
            # lines_intersect_index = lines_intersect_index.tolist()
            # lines_intersect_index.insert(0, 0)
            # lines_intersect_index.append(-1)

            pre_point = points[0]
            color = (255, 0, 0)
            for i, cur_point in enumerate(points[1:]):
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                cv2.circle(img, (x1, y1), 1, color, 1)
                cv2.line(img, (x1, y1), (x2, y2), color, 3)
                pre_point = cur_point

            pre_point = points[0]
            pre_point_type = attr_type[0]
            pre_point_vis = attr_visible[0]
            pre_point_hang = attr_hanging[0]

            for cur_point, point_type, point_vis, point_hang in zip(points[1:],
                                                                attr_type[1:],
                                                                attr_visible[1:],
                                                                attr_hanging[1:]):
                if point_type not in classes:
                    print("skip point type:", point_type)
                    continue

                # img_line
                color = (255, 0, 0)
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                cv2.circle(img_line, (x1, y1), 1, color, 1)
                cv2.line(img_line, (x1, y1), (x2, y2), color, 3)
                pre_point = cur_point

                # img_cls
                # A ---- B -----C
                # A点的属性代表AB线的属性
                # B点的属性代表BC线的属性
                cls_index = classes.index(pre_point_type)
                color = palette[cls_index]
                cv2.circle(img_cls, (x1, y1), 1, color, 1)
                cv2.line(img_cls, (x1, y1), (x2, y2), color, 3)
                pre_point_type = point_type

                # img_vis
                # point_vis为true的情况下为可见
                color = (0, 255, 0) if pre_point_vis else (255, 0, 0)
                cv2.circle(img_vis, (x1, y1), 1, color, 1)
                cv2.line(img_vis, (x1, y1), (x2, y2), color, 3)
                pre_point_vis = point_vis

                # img_vis
                # point_hang为true的情况为不悬空
                color = (0, 255, 0) if pre_point_hang else (255, 0, 0)
                cv2.circle(img_hang, (x1, y1), 1, color, 1)
                cv2.line(img_hang, (x1, y1), (x2, y2), color, 3)
                pre_point_hang = point_hang

        img_h, img_w, _ = img_line.shape
        img_line = cv2.resize(img_line, (img_w//4, img_h//4))
        s_img_path = os.path.join(dst_root, img_name)
        cv2.imwrite(s_img_path, img_line)

        # plt.imshow(img[:, :, ::-1])
        # plt.show()

        plt.subplot(2, 2, 1)
        plt.imshow(img_line[:, :, ::-1])

        plt.subplot(2, 2, 2)
        plt.imshow(img_cls[:, :, ::-1])

        plt.subplot(2, 2, 3)
        plt.imshow(img_vis[:, :, ::-1])

        plt.subplot(2, 2, 4)
        plt.imshow(img_hang[:, :, ::-1])
        plt.show()
        exit(1)

def debug_signal_filter():
    def moving_average(x, window_size):
        kernel = np.ones(window_size) / window_size
        # kernel = np.array([1, 1, 1])
        result = np.correlate(x, kernel, mode='same')


        # 将边界效应里的设置为原来的数值
        valid_sie = window_size//2
        result[:valid_sie] = x[:valid_sie]
        result[-valid_sie:] = x[-valid_sie:]
        return result

    # 生成一维信号
    # x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x = np.array([1, 3, 5,  7,  9, 10])

    # 应用均值滤波器
    filtered_signal = moving_average(x, window_size=3)

    print(filtered_signal)

if __name__ == "__main__":
    print("Start")
    # test_mmdet_fpn()
    # debug_write_data_infos()
    # test_dtype()
    # debug_sum()
    # debug_padd()

    # 调试标注系统解析的json文件
    # debug_parse_label_json()

    # 数据集里解析的json文件
    debug_signal_filter()
    print("End")