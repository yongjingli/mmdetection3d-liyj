# 用来绘制表示算法原理的图像
import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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


def draw_gt_maps_opencv(ann_info, map_size, thickness=3):
    gt_lines = ann_info["gt_lines"]
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
            cv2.line(seg_map, (x1, y1), (x2, y2), (1,), thickness)
            cv2.line(line_map, (x1, y1), (x2, y2), (index,), thickness)
            cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,), thickness)
            cv2.line(line_map_cls, (x1, y1), (x2, y2), (category_id,), thickness)
            pre_point = cur_point

    return seg_map, line_map, line_map_id, line_map_cls


def show_gt_map():
    img_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.jpg"
    json_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.json"
    s_root = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data"

    down_scale = 0.8

    ann = {}
    ann["img_path"] = img_path
    ann["ann_path"] = json_path
    classes = ['glass_edge', 'glass_edge_up_plant', 'glass_edge_up_build']
    ann_info = parse_ann_info(ann, classes, down_scale)

    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape
    img_h = int(img_h * down_scale)
    img_w = int(img_w * down_scale)

    # map_size = (img_h, img_w)
    # seg_map, line_map, line_map_id, line_map_cls = draw_gt_maps(ann_info, map_size, thickness=10)

    map_size = (img_h, img_w, 3)
    seg_map, line_map, line_map_id, line_map_cls = draw_gt_maps(ann_info, map_size, thickness=10)

    plt.subplot(2, 2, 1)
    plt.imshow(seg_map)

    plt.subplot(2, 2, 2)
    plt.imshow(line_map)

    plt.subplot(2, 2, 3)
    plt.imshow(line_map_id)

    plt.subplot(2, 2, 4)
    plt.imshow(line_map_cls)

    plt.savefig(os.path.join(s_root, "line_map_cls.jpg"))

    cv2.imwrite(os.path.join(s_root, "seg_map.jpg"), seg_map)
    cv2.imwrite(os.path.join(s_root, "line_map.jpg"), line_map)
    cv2.imwrite(os.path.join(s_root, "line_map_id.jpg"), line_map_id)
    cv2.imwrite(os.path.join(s_root, "line_map_cls.jpg"), line_map_cls)
    print(ann_info)
    plt.show()


if __name__ == "__main__":
    print("Start")
    show_gt_map()
    print("End")

