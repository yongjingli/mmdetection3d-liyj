import os
import shutil
import cv2
import json
import copy
import pickle
import math
from tqdm import tqdm
import numpy as np
import mmengine
import matplotlib.pyplot as plt
from mmengine.fileio import join_path, list_from_file, load
from dense_line_points import dense_line_points, dense_line_points_by_interp, dense_line_points_by_interp_with_attr


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (221, 160, 221),  (128, 0, 128), (203, 192, 255), (238, 130, 238), (0, 69, 255),
              (130, 0, 75), (255, 255, 0), (250, 51, 153), (214, 112, 218), (255, 165, 0),
              (169, 169, 169), (18, 74, 115),
              (240, 32, 160), (192, 192, 192), (112, 128, 105), (105, 128, 128),
              ]


TYPE_DICT = {
    "路面边界线": "road_boundary_line",
    "灌木丛边界线": "bushes_boundary_line",
    "围栏边界线": "fence_boundary_line",
    "石头边界线": "stone_boundary_line",
    "墙体边界线": "wall_boundary_line",
    "水面边界线": "water_boundary_line",
    "雪地边界线": "snow_boundary_line",
    "井盖边界线": "manhole_boundary_line",
    # "悬空物体边界线": "hanging_object_boundary_line",
    "其他线": "others_boundary_line"
}

classes = list(TYPE_DICT.values())



def debug_parse_dataset_json():
    # root = os.path.join(generate_dataset_infos["dst_root"], 'train')
    root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/debug_overfit/train"

    img_root = os.path.join(root, "images")
    label_root = os.path.join(root, "jsons")

    dst_root = img_root + "_debug_vis"
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    # down_scale = 6
    # 原始分辨率到模型输入分辨率(3倍, 2880 - 960), 模型的下采样倍数为4
    down_scale = 6
    thickness = 2
    circle_thickness = 1

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", '.png']]
    for img_name in tqdm(img_names, desc="img_names"):
        # img_name = "1696991193.66982.jpg"
        # img_name = "1689848745238965269.jpg"
        img_name = "1700108695.418044.jpg"
        print("img_name:", img_name)
        img_path = os.path.join(img_root, img_name)
        json_path = os.path.join(label_root, img_name[:-4] + ".json")

        img = cv2.imread(img_path)
        # img = np.ones_like(img) * 255

        with open(json_path, "r") as fp:
            labels = json.load(fp)

        lines = []
        all_points = []
        all_points_type = []
        all_points_visible = []
        all_points_hanging = []
        all_points_covered = []
        lines_intersect_indexs = []
        for label in labels["shapes"]:
            # intersect_index = np.array(label["intersect_index"])

            points = np.array(label["points"])
            points_type = np.array(label["points_type"]).reshape(-1, 1)
            points_visible = np.array(label["points_visible"]).reshape(-1, 1)
            # print(points_visible)
            points_hanging = np.array(label["points_hanging"]).reshape(-1, 1)
            points_covered = np.array(label["points_covered"]).reshape(-1, 1)
            # print(points.shape, attr_type.shape, attr_visible.shape, attr_hanging.shape)

            line = np.concatenate([points, points_type, points_visible, points_hanging, points_covered], axis=1)

            lines.append(line)
            all_points.append(points)
            all_points_type.append(points_type)
            all_points_visible.append(points_visible)
            all_points_hanging.append(points_hanging)
            all_points_covered.append(points_covered)

        # img_line = np.ones_like(img) * 255
        img_h, img_w, img_c = img.shape
        img_h, img_w = img_h//down_scale, img_w//down_scale
        img = cv2.resize(img,(img_w, img_h))

        img_line = np.ones((img_h, img_w, img_c), dtype=np.uint8) * 255
        # img_line = copy.deepcopy(img)

        img_cls = np.ones_like(img_line) * 255
        img_vis = np.ones_like(img_line) * 255
        img_hang = np.ones_like(img_line) * 255
        img_covered = np.ones_like(img_line) * 255

        line_count = 0

        for points, points_type, points_visible, points_hanging, points_covered in \
                zip(all_points, all_points_type, all_points_visible, all_points_hanging, all_points_covered):
            # 在lines_intersect_index的首尾添加0, -1的index
            # lines_intersect_index = lines_intersect_index.tolist()
            # lines_intersect_index.insert(0, 0)
            # lines_intersect_index.append(-1)

            pre_point = points[0]
            color = (255, 0, 0)
            for i, cur_point in enumerate(points[1:]):
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                x1, y1 = x1//down_scale, y1//down_scale
                x2, y2 = x2//down_scale, y2//down_scale

                # cv2.circle(img, (x1, y1), 1, color, thickness=circle_thickness)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness=thickness)
                pre_point = cur_point

            pre_point = points[0]
            pre_point_type = points_type[0]
            pre_point_vis = points_visible[0]
            pre_point_hang = points_hanging[0]
            pre_point_covered = points_covered[0]

            First_Point = True
            for cur_point, point_type, point_vis, point_hang, point_covered in zip(points[1:],
                                                                points_type[1:],
                                                                points_visible[1:],
                                                                points_hanging[1:],
                                                                points_covered[1:]):
                if point_type not in classes:
                    print("skip point type:", point_type)
                    continue

                # img_line
                # color = (255, 0, 0)
                color = color_list[line_count % len(color_list)]
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                x1, y1 = x1 // down_scale, y1 // down_scale
                x2, y2 = x2 // down_scale, y2 // down_scale

                # cv2.circle(img_line, (x1, y1), 1, color, thickness=circle_thickness)
                cv2.line(img_line, (x1, y1), (x2, y2), color,  thickness=1)
                if First_Point:
                    center_point = points[len(points)//2]
                    cv2.putText(img_line, str(line_count), (int(center_point[0]), int(center_point[1])),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 12)
                    First_Point = False

                pre_point = cur_point
                # 点的属性全都是字符类型
                # img_cls
                # A ---- B -----C
                # A点的属性代表AB线的属性
                # B点的属性代表BC线的属性
                cls_index = classes.index(pre_point_type[0])
                color = color_list[cls_index]
                # cv2.circle(img_cls, (x1, y1), 1, color, thickness=circle_thickness)
                cv2.line(img_cls, (x1, y1), (x2, y2), color, thickness=thickness)
                pre_point_type = point_type

                # img_vis
                # 绿色为true, 蓝色为false
                # point_vis为true的情况下为可见
                color = (0, 255, 0) if pre_point_vis[0] == "true" else (255, 0, 0)
                # cv2.circle(img_vis, (x1, y1), 1, color, thickness=circle_thickness)
                cv2.line(img_vis, (x1, y1), (x2, y2), color, thickness=thickness)
                pre_point_vis = point_vis

                # img_hang
                # point_hang为true的情况为悬空
                color = (0, 255, 0) if pre_point_hang[0] == "true" else (255, 0, 0)
                # cv2.circle(img_hang, (x1, y1), 1, color, thickness=circle_thickness)
                cv2.line(img_hang, (x1, y1), (x2, y2), color, thickness=thickness)
                pre_point_hang = point_hang

                # img_covered
                # point_covered为true的情况为被草遮挡
                color = (0, 255, 0) if pre_point_covered[0] == "true" else (255, 0, 0)
                # cv2.circle(img_covered, (x1, y1), 1, color, thickness=circle_thickness)
                cv2.line(img_covered, (x1, y1), (x2, y2), color, thickness=thickness)
                pre_point_covered = point_covered

            line_count = line_count + 1

        img_h, img_w, _ = img_line.shape
        # img_line = cv2.resize(img_line, (img_w//4, img_h//4))
        s_img_path = os.path.join(dst_root, img_name)
        cv2.imwrite(s_img_path, img_line)

        plt.imshow(img[:, :, ::-1])
        plt.show()

        plt.subplot(2, 2, 1)
        plt.imshow(img_line[:, :, ::-1])

        plt.subplot(2, 2, 2)
        plt.imshow(img_cls[:, :, ::-1])

        plt.subplot(2, 2, 3)
        plt.imshow(img_vis[:, :, ::-1])

        plt.subplot(2, 2, 4)
        # plt.imshow(img_hang[:, :, ::-1])
        plt.imshow(img_covered[:, :, ::-1])
        plt.show()
        exit(1)


def _gen_line_map(gt_lines, map_size, thickness=1):
    line_map = np.zeros(map_size, dtype=np.uint8)
    line_map_id = np.zeros(map_size, dtype=np.uint8)
    line_map_cls = np.zeros(map_size, dtype=np.uint8)

    line_map_visible = np.zeros(map_size, dtype=np.uint8)
    line_map_hanging = np.zeros(map_size, dtype=np.uint8)
    line_map_covered = np.zeros(map_size, dtype=np.uint8)

    # thickness = 1
    # if map_size[1] > 240:
    #     thickness = 2
    #
    # if map_size[1] > 480:
    #     thickness = 4

    for gt_line in gt_lines:
        label = gt_line['label']
        line_points = gt_line['points']
        index = gt_line['index'] + 1     # 序号从0开始的
        line_id = gt_line['line_id'] + 1
        category_id = gt_line['category_id'] + 1

        line_points_type = gt_line['points_type']
        line_points_visible = gt_line['points_visible']
        line_points_hanging = gt_line['points_hanging']
        line_points_covered = gt_line['points_covered']

        pre_point = line_points[0]
        pre_point_type = line_points_type[0]
        pre_point_visible = line_points_visible[0]
        pre_point_hanging = line_points_hanging[0]
        pre_point_covered = line_points_covered[0]

        for cur_point, cur_point_type, cur_point_visible, cur_point_hanging, cur_point_covered in \
                zip(line_points[1:], line_points_type[1:], line_points_visible[1:],
                    line_points_hanging[1:], line_points_covered[1:]):

            x1, y1 = round(pre_point[0]), round(pre_point[1])
            x2, y2 = round(cur_point[0]), round(cur_point[1])
            cv2.line(line_map, (x1, y1), (x2, y2), (index,), thickness=thickness)
            cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,), thickness=thickness)

            # cls_value = self.class_dic[label] + 1
            # cv2.line(line_map_cls, (x1, y1), (x2, y2), (category_id,))

            # 以起点的属性为准
            # 修改为分段类别
            cv2.line(line_map_cls, (x1, y1), (x2, y2), (int(pre_point_type[0]) + 1,), thickness=thickness)

            # 新增可见、悬空和被草遮挡的属性预测
            if pre_point_visible[0] > 0:
                cv2.line(line_map_visible, (x1, y1), (x2, y2), (1,), thickness=thickness)

            if pre_point_hanging[0] > 0:
                cv2.line(line_map_hanging, (x1, y1), (x2, y2), (1,), thickness=thickness)

            if pre_point_covered[0] > 0:
                cv2.line(line_map_covered, (x1, y1), (x2, y2), (1,), thickness=thickness)

            pre_point = cur_point
            pre_point_type = cur_point_type
            pre_point_visible = cur_point_visible
            pre_point_hanging = cur_point_hanging
            pre_point_covered = cur_point_covered

    return line_map, line_map_id, line_map_cls, line_map_visible, line_map_hanging, line_map_covered


def _gen_gt_line_maps(line_map, line_map_id, line_map_cls,
                      line_map_visible, line_map_hanging, line_map_covered,
                      grid_size, num_classes=1):
    line_map_h, line_map_w = line_map.shape
    gt_map_h, gt_map_w = math.ceil(line_map_h / grid_size), math.ceil(
        line_map_w / grid_size
    )
    gt_confidence = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
    gt_offset_x = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
    gt_offset_y = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
    gt_line_index = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
    gt_line_id = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)

    gt_line_cls = np.zeros((num_classes, gt_map_h, gt_map_w), dtype=np.float32)

    # 新增可见、悬空、被草遮挡的confidence预测
    gt_confidence_visible = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
    gt_confidence_hanging = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
    gt_confidence_covered = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)

    for y in range(0, gt_map_h):
        for x in range(0, gt_map_w):
            start_x, end_x = x * grid_size, (x + 1) * grid_size
            end_x = end_x if end_x < line_map_w else line_map_w
            start_y, end_y = y * grid_size, (y + 1) * grid_size
            end_y = end_y if end_y < line_map_h else line_map_h
            grid = line_map[start_y:end_y, start_x:end_x]

            grid_id = line_map_id[start_y:end_y, start_x:end_x]
            grid_cls = line_map_cls[start_y:end_y, start_x:end_x]

            confidence = 1 if np.any(grid) else 0
            gt_confidence[0, y, x] = confidence
            if confidence == 1:
                ys, xs = np.nonzero(grid)
                offset_y, offset_x = sorted(
                    zip(ys, xs), key=lambda p: (p[0], -p[1]), reverse=True
                )[0]
                if grid_size != 1:
                    gt_offset_x[0, y, x] = offset_x / (grid_size - 1)
                    gt_offset_y[0, y, x] = offset_y / (grid_size - 1)

                    # 设置成
                    # gt_confidence[0, y, x] = confidence - min(offset_x, offset_y)/grid_size

                gt_line_index[0, y, x] = grid[offset_y, offset_x]
                gt_line_id[0, y, x] = grid_id[offset_y, offset_x]

                cls = grid_cls[offset_y, offset_x]
                if cls > 0:
                    cls_indx = int(cls - 1)
                    gt_line_cls[cls_indx, y, x] = 1

            # gt_confidence_visible
            if np.any(line_map_visible[start_y:end_y, start_x:end_x]):
                gt_confidence_visible[0, y, x] = 1

            # gt_confidence_hanging
            if np.any(line_map_hanging[start_y:end_y, start_x:end_x]):
                gt_confidence_hanging[0, y, x] = 1

            # gt_confidence_covered
            if np.any(line_map_covered[start_y:end_y, start_x:end_x]):
                gt_confidence_covered[0, y, x] = 1

    foreground_mask = gt_confidence.astype(np.uint8)

    # expand foreground mask
    kernel = np.ones((3, 3), np.uint8)
    foreground_expand_mask = cv2.dilate(foreground_mask[0], kernel)
    foreground_expand_mask = np.expand_dims(foreground_expand_mask.astype(np.uint8), axis=0)

    ignore_mask = np.zeros((1, gt_map_h, gt_map_w), dtype=np.uint8)
    # top, bottom = self.line_map_range
    ignore_mask[0, 0:-1, :] = 1  # 手动设置有效范围

    return gt_confidence, gt_offset_x, gt_offset_y, \
        gt_line_index, ignore_mask, foreground_mask, \
        gt_line_id, gt_line_cls, foreground_expand_mask, \
        gt_confidence_visible, gt_confidence_hanging, gt_confidence_covered


def get_gt_lines(labels, down_scale=1):
    gt_lines = []
    for i, label in enumerate(labels["shapes"]):
        gt_line = dict()
        # intersect_index = np.array(label["intersect_index"])
        label_name = label['label']
        index = i + 1
        line_id = i + 1

        category_id = classes.index(label_name) + 1
        points = np.array(label["points"])

        label["points_type"] = [classes.index(_type[0]) for _type in label["points_type"]]
        points_type = np.array(label["points_type"]).reshape(-1, 1)
        #
        # # 将字符类型转为与category_id相同
        # for point_type in points_type:
        #     point_type[0] = self.metainfo["classes"].index(point_type[0])
        #
        # # point_vis为true的情况下为可见
        # for point_visible in points_visible:
        #     point_visible[0] = 1 if point_visible[0] == "true" else 0
        #
        # # point_hang为true的情况为悬空
        # for point_hanging in points_hanging:
        #     point_hanging[0] = 1 if point_hanging[0] == "true" else 0
        #
        # # point_covered为true的情况为被草遮挡
        # for point_covered in points_covered:
        #     point_covered[0] = 1 if point_covered[0] == "true" else 0
        for _visible in label["points_visible"]:
            _visible[0] = 1 if _visible[0] == "true" else 0

        for _hanging in label["points_hanging"]:
            _hanging[0] = 1 if _hanging[0] == "true" else 0

        for _covered in label["points_covered"]:
            _covered[0] = 1 if _covered[0] == "true" else 0

        points_visible = np.array(label["points_visible"]).reshape(-1, 1)
        points_hanging = np.array(label["points_hanging"]).reshape(-1, 1)
        points_covered = np.array(label["points_covered"]).reshape(-1, 1)

        points = points/down_scale

        gt_line["label"] = label_name
        gt_line["points"] = points
        gt_line["index"] = index
        gt_line["line_id"] = line_id
        gt_line["category_id"] = category_id
        gt_line["points_type"] = points_type
        gt_line["points_visible"] = points_visible
        gt_line["points_hanging"] = points_hanging
        gt_line["points_covered"] = points_covered

        gt_lines.append(gt_line)

    return gt_lines


# 调试随机旋转的数据增强
def GgldRotate(img, gt_lines):
    height, width = img.shape[:2]
    angle_range = 15                           # 设置随机旋转角度范围为[-angle_range, angle_range]
    center = (width // 2, height // 2)         # 中心点坐标

    rotate_angle = np.random.randint(-angle_range, angle_range)
    # rotate_angle = -12
    rotation_matrix = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))



    # points = np.array([[50, 50], [100, 50], [100, 100]], dtype=np.float32)  # 示例点集
    rotate_gt_lines = []
    # 得到当前图像线段的数量, 如果线段分段新增线段将赋予新的id
    max_index = 0
    max_line_idx = 0
    for gt_line in gt_lines:
        if gt_line["index"] > max_index:
            max_index = gt_line["index"]

        if gt_line["line_id"] > max_line_idx:
            max_line_idx = gt_line["line_id"]

    for gt_line in gt_lines:
        points = gt_line['points']
        rotated_points = cv2.transform(points.reshape(-1, 1, 2), rotation_matrix)
        rotated_points = rotated_points.squeeze()

        mask_x = np.bitwise_and(rotated_points[:, 0] >= 0, rotated_points[:, 0] < width)
        mask_y = np.bitwise_and(rotated_points[:, 1] >= 0, rotated_points[:, 1] < height)

        mask = np.bitwise_and(mask_x, mask_y)

        if "points_type" in gt_line:
            points_type = gt_line["points_type"]
        else:
            points_type = None

        if "points_visible" in gt_line:
            points_visible = gt_line["points_visible"]
        else:
            points_visible = None

        if "points_hanging" in gt_line:
            points_hanging = gt_line["points_hanging"]
        else:
            points_hanging = None

        if "points_covered" in gt_line:
            points_covered = gt_line["points_covered"]
        else:
            points_covered = None

        # 在这里判断线是否需要分段成为不同的线，具有不同的id, 否则对聚类有影响
        # 将中间mask分段的线段分为不同的线段
        line_indexs = []
        start_indx = None
        for i, _mask in enumerate(mask):
            if _mask and start_indx is None:
                start_indx = i
            elif (not _mask) and start_indx is not None:
                end_indx = i - 1
                line_indexs.append((start_indx, end_indx))
                start_indx = None

            if i == len(mask) - 1:
                end_indx = i
                if start_indx is not None:
                    line_indexs.append((start_indx, end_indx))

        for j, line_index in enumerate(line_indexs):
            crop_gt_line = copy.deepcopy(gt_line)
            # points = points[mask]
            id_0, id_1 = line_index
            id_1 = id_1 + 1
            splint_points = rotated_points[id_0:id_1]

            if len(splint_points) < 2:
                continue

            crop_gt_line["points"] = splint_points

            # 最后一个代表的是下个线段的属性
            if points_type is not None:
                split_points_type = np.concatenate([points_type[id_0:id_1], points_type[id_1 - 1:id_1]], axis=0)
                crop_gt_line["points_type"] = split_points_type

            if points_visible is not None:
                split_points_visible = np.concatenate([points_visible[id_0:id_1], points_visible[id_1 - 1:id_1]], axis=0)
                crop_gt_line["points_visible"] = split_points_visible

            if points_hanging is not None:
                split_points_hanging = np.concatenate([points_hanging[id_0:id_1], points_hanging[id_1 - 1:id_1]], axis=0)
                crop_gt_line["points_hanging"] = split_points_hanging

            if points_covered is not None:
                split_points_covered = np.concatenate([points_covered[id_0:id_1], points_covered[id_1 - 1:id_1]], axis=0)
                crop_gt_line["points_covered"] = split_points_covered

            # 代表新增的线
            if j > 0:
                max_index = max_index + 1
                max_line_idx = max_line_idx + 1
                crop_gt_line["index"] = max_index
                crop_gt_line["line_id"] = max_line_idx

            rotate_gt_lines.append(crop_gt_line)
    return rotated_image, rotate_gt_lines



def debug_generate_heatmap():
    # import sys
    # sys.path.insert(0, "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/gbld2d")
    # from gbld_mono2d_transform import GgldLineMapsGenerateV2

    root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/debug_overfit/train"
    img_root = os.path.join(root, "images")
    label_root = os.path.join(root, "jsons")
    down_scale = 3  # 模型输出的下采样倍数
    gt_down_scales = [16]   # 模型内部的下采样倍数
    thickness = 1
    num_classes = len(classes)

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", '.png']]
    for img_name in tqdm(img_names, desc="img_names"):
        img_name = "1700108695.418044.jpg"
        print("img_name:", img_name)
        img_path = os.path.join(img_root, img_name)
        json_path = os.path.join(label_root, img_name[:-4] + ".json")

        img = cv2.imread(img_path)
        # img = np.ones_like(img) * 255
        with open(json_path, "r") as fp:
            labels = json.load(fp)

        img_h, img_w, img_c = img.shape
        img_h, img_w = img_h//down_scale, img_w//down_scale
        img = cv2.resize(img, (img_w, img_h))

        map_size = (img_h, img_w)
        gt_lines = get_gt_lines(labels, down_scale=down_scale)

        img, gt_lines = GgldRotate(img, gt_lines)


        line_map, line_map_id, line_map_cls, line_map_visible,\
        line_map_hanging, line_map_covered = _gen_line_map(gt_lines, map_size, thickness=thickness)

        for gt_down_scale in gt_down_scales:
            # (line_map, line_map_id, line_map_cls,
            #  line_map_visible, line_map_hanging, line_map_covered,
            #  grid_size, num_classes)

            gt_line_maps = _gen_gt_line_maps(line_map, line_map_id, line_map_cls,
                                             line_map_visible, line_map_hanging,
                                             line_map_covered, gt_down_scale,
                                             num_classes=num_classes)

            gt_confidence, gt_offset_x, gt_offset_y, \
                gt_line_index, ignore_mask, foreground_mask, \
                gt_line_id, gt_line_cls, foreground_expand_mask, \
                gt_confidence_visible, gt_confidence_hanging, gt_confidence_covered = gt_line_maps

        plt.subplot(3, 1, 1)
        plt.imshow(img[:, :, ::-1])

        plt.subplot(3, 1, 2)
        plt.imshow(line_map)

        plt.subplot(3, 1, 3)
        plt.imshow(gt_confidence[0])
        plt.show()
        exit(1)

    print("ff")


if __name__ == "__main__":
    print("Start")
    # 读取数据并绘制曲线
    # debug_parse_dataset_json()

    # 调试产生heatmap
    debug_generate_heatmap()
    print("End")