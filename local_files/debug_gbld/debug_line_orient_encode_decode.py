# 用来绘制表示算法原理的图像
import json
import cv2
import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from debug_utils import decode_gt_lines, cal_points_orient, draw_orient, filter_near_same_points

color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]


def parse_ann_info(info: dict, classes: list, down_scale=1.0) -> dict:
    ann_path = info["ann_path"]
    with open(ann_path, 'r') as f:
        anns = json.load(f)

    ann_info = dict()

    remap_anns = []
    for i, shape in enumerate(anns['shapes']):
        line_id = int(shape['id'])
        label = shape['label']
        #
        if label not in classes:
            continue

        category_id = classes.index(label)

        points = shape['points']
        points_remap = []

        # 读取数据阶段进行缩放, 在图像进行方法的时候才需要
        scale_x = down_scale
        scale_y = down_scale
        pad_left = 0.0
        pad_top = 0
        for point in points:
            x = point[0] * scale_x + pad_left
            y = point[1] * scale_y + pad_top
            points_remap.append([x, y])

        points_remap = np.array(points_remap)
        # remap_annos.append({'label': label, 'points': points_remap, 'index': index, 'same_line_id': index})
        # 新增label对应的类id, category_id
        remap_anns.append(
            {'label': label, 'points': points_remap, 'index': i, 'line_id': line_id, "category_id": category_id})

    ann_info["gt_lines"] = remap_anns
    return ann_info


def draw_gt_maps(ann_info, map_size, thickness=3):
    gt_lines = ann_info["gt_lines"]
    # seg_map = np.ones(map_size, dtype=np.uint8) * 255
    # line_map = np.ones(map_size, dtype=np.uint8) * 255
    # line_map_id = np.ones(map_size, dtype=np.uint8) * 255
    # line_map_cls = np.ones(map_size, dtype=np.uint8) * 255

    seg_map = np.zeros(map_size, dtype=np.uint8)
    line_map = np.zeros(map_size, dtype=np.uint8)
    line_map_id = np.zeros(map_size, dtype=np.uint8)
    line_map_cls = np.zeros(map_size, dtype=np.uint8)
    for gt_line in gt_lines:
        label = gt_line['label']
        line_points = gt_line['points']
        index = gt_line['index'] + 1  # 序号从0开始的
        line_id = gt_line['line_id'] + 1
        category_id = gt_line['category_id'] + 1

        pre_point = line_points[0]
        for cur_point in line_points[1:]:
            x1, y1 = round(pre_point[0]), round(pre_point[1])
            x2, y2 = round(cur_point[0]), round(cur_point[1])
            cv2.line(seg_map, (x1, y1), (x2, y2), color_list[0], thickness)
            cv2.line(line_map, (x1, y1), (x2, y2), color_list[index], thickness)
            cv2.line(line_map_id, (x1, y1), (x2, y2), color_list[line_id], thickness)
            cv2.line(line_map_cls, (x1, y1), (x2, y2), color_list[category_id], thickness)
            pre_point = cur_point

    return seg_map, line_map, line_map_id, line_map_cls


def gen_gt_line_maps(line_map, line_map_id, line_map_cls, grid_size, num_classes):
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
                gt_offset_x[0, y, x] = offset_x / (grid_size - 1)
                gt_offset_y[0, y, x] = offset_y / (grid_size - 1)
                gt_line_index[0, y, x] = grid[offset_y, offset_x]
                gt_line_id[0, y, x] = grid_id[offset_y, offset_x]

                cls = grid_cls[offset_y, offset_x]
                if cls > 0:
                    cls_indx = int(cls - 1)
                    gt_line_cls[cls_indx, y, x] = 1

    foreground_mask = gt_confidence.astype(np.uint8)

    # expand foreground mask
    kernel = np.ones((3, 3), np.uint8)
    foreground_expand_mask = cv2.dilate(foreground_mask[0], kernel)
    foreground_expand_mask = np.expand_dims(foreground_expand_mask.astype(np.uint8), axis=0)

    ignore_mask = np.zeros((1, gt_map_h, gt_map_w), dtype=np.uint8)
    # top, bottom = self.line_map_range
    ignore_mask[0, 0:-1, :] = 1     # 手动设置有效范围

    return gt_confidence, gt_offset_x, gt_offset_y, \
           gt_line_index, ignore_mask, foreground_mask, \
           gt_line_id, gt_line_cls, foreground_expand_mask


def gen_line_map(gt_lines, map_size):
    line_map = np.zeros(map_size, dtype=np.uint8)
    line_map_id = np.zeros(map_size, dtype=np.uint8)
    line_map_cls = np.zeros(map_size, dtype=np.uint8)
    for gt_line in gt_lines:
        label = gt_line['label']
        line_points = gt_line['points']
        index = gt_line['index'] + 1     # 序号从0开始的
        line_id = gt_line['line_id'] + 1
        category_id = gt_line['category_id'] + 1

        pre_point = line_points[0]
        for cur_point in line_points[1:]:
            x1, y1 = round(pre_point[0]), round(pre_point[1])
            x2, y2 = round(cur_point[0]), round(cur_point[1])
            cv2.line(line_map, (x1, y1), (x2, y2), (index,))
            cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,))

            # cls_value = self.class_dic[label] + 1
            cv2.line(line_map_cls, (x1, y1), (x2, y2), (category_id,))
            pre_point = cur_point

    return line_map, line_map_id, line_map_cls


def get_gt_maps(gt_lines, map_size, num_classes, gt_down_scales=[1]):
    # map_size = results["img_shape"]
    # gt_lines = results["gt_lines"]

    gt_line_maps_stages = []
    line_map, line_map_id, line_map_cls = gen_line_map(gt_lines, map_size)

    # gt_down_scales = [1]
    # for gt_down_scale in self.gt_down_scales:
    for gt_down_scale in gt_down_scales:
        gt_line_maps = gen_gt_line_maps(line_map, line_map_id, line_map_cls, gt_down_scale, num_classes)

        gt_confidence = gt_line_maps[0]
        gt_offset_x = gt_line_maps[1]
        gt_offset_y = gt_line_maps[2]
        gt_line_index = gt_line_maps[3]
        ignore_mask = gt_line_maps[4]
        foreground_mask = gt_line_maps[5]
        gt_line_id = gt_line_maps[6]
        gt_line_cls = gt_line_maps[7]
        foreground_expand_mask = gt_line_maps[8]

        gt_line_maps = {
            "gt_confidence": gt_confidence,
            "gt_offset_x": gt_offset_x,
            "gt_offset_y": gt_offset_y,
            "gt_line_index": gt_line_index,
            "ignore_mask": ignore_mask,
            "foreground_mask": foreground_mask,
            "gt_line_id": gt_line_id,
            "gt_line_cls": gt_line_cls,
            "foreground_expand_mask": foreground_expand_mask,
        }

        gt_line_maps_stages.append(gt_line_maps)

    # results['gt_line_maps_stages'] = gt_line_maps_stages
    return gt_line_maps_stages


def debug_line_orient_encode_decode():
    img_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.jpg"
    json_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.json"
    s_root = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data"

    down_scale = 0.8

    ann = {}
    ann["img_path"] = img_path
    ann["ann_path"] = json_path
    classes = ['glass_edge', 'glass_edge_up_plant', 'glass_edge_up_build']
    num_classes = len(classes)
    ann_info = parse_ann_info(ann, classes, down_scale)

    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    img_h = int(img_h * down_scale)
    img_w = int(img_w * down_scale)
    img = cv2.resize(img, (img_w, img_h))
    img_show = copy.deepcopy(img)

    map_size = (img_h, img_w)
    gt_lines = ann_info["gt_lines"]
    grid_size = 1

    if 0:
        gt_line_maps_stages = get_gt_maps(gt_lines, map_size, num_classes, gt_down_scales=[grid_size])
        gt_line_maps_stage = gt_line_maps_stages[0]
        gt_confidence = gt_line_maps_stage["gt_confidence"]
        gt_offset_x = gt_line_maps_stage["gt_offset_x"]
        gt_offset_y = gt_line_maps_stage["gt_offset_y"]
        gt_line_index = gt_line_maps_stage["gt_line_index"]
        gt_line_id = gt_line_maps_stage["gt_line_id"]
        gt_line_cls = gt_line_maps_stage["gt_line_cls"]
        gt_offset = np.concatenate([gt_offset_x, gt_offset_y], axis=0)

        np.save(os.path.join(s_root, "gt_confidence.npy"), gt_confidence)
        np.save(os.path.join(s_root, "gt_offset.npy"), gt_offset)
        np.save(os.path.join(s_root, "gt_line_index.npy"), gt_line_index)
        np.save(os.path.join(s_root, "gt_line_id.npy"), gt_line_id)
        np.save(os.path.join(s_root, "gt_line_cls.npy"), gt_line_cls)
        exit(1)
    else:
        gt_confidence = np.load(os.path.join(s_root, "gt_confidence.npy"))
        gt_offset = np.load(os.path.join(s_root, "gt_offset.npy"))
        gt_line_index = np.load(os.path.join(s_root, "gt_line_index.npy"))
        gt_line_id = np.load(os.path.join(s_root, "gt_line_id.npy"))
        gt_line_cls = np.load(os.path.join(s_root, "gt_line_cls.npy"))


    curse_lines_with_cls = decode_gt_lines(gt_confidence, gt_offset, gt_line_index, gt_line_id, gt_line_cls, grid_size=grid_size)

    # 可视化encode和decode后的gt
    single_stage_result = curse_lines_with_cls[0]
    for curve_line in single_stage_result:
        curve_line = np.array(curve_line)
        curve_line = filter_near_same_points(curve_line)
        pre_point = curve_line[0]

        line_cls = pre_point[4]
        color = color_list[int(line_cls)]
        for i, cur_point in enumerate(curve_line[1:]):
            x1, y1 = int(pre_point[0]), int(pre_point[1])
            x2, y2 = int(cur_point[0]), int(cur_point[1])

            thickness = 3
            # cv2.line(img_show, (x1, y1), (x2, y2), color, thickness, 8)

            # 求每个点的方向
            if i % 50 == 0:
                orient = cal_points_orient(pre_point, cur_point)
                if orient != -1:
                    # 转个90度,指向草地
                    orient = orient + 90
                    if orient > 360:
                        orient = orient - 360

                    img_show = draw_orient(img_show, pre_point, orient, arrow_len=20)

            pre_point = cur_point

    plt.imshow(img_show[:, :, ::-1])
    # plt.imshow(gt_line_index[0])
    plt.show()


if __name__ == "__main__":
    print("Start")
    # 将gt-lines编码为模型训练时的gt,然后再decode出来
    debug_line_orient_encode_decode()
    print("End")

