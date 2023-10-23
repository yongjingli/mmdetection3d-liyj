import numpy as np
import cv2
import os
import json
import math
import copy
import matplotlib.pyplot as plt
from debug_utils import decode_gt_lines, cal_points_orient, draw_orient, filter_near_same_points


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


def remove_short_lines(lines, long_line_thresh=4):
    long_lines = []
    for line in lines:
        if len(line) >= long_line_thresh:
            long_lines.append(line)
    return long_lines


def remove_far_lines(lines, far_line_thresh=10):
    near_lines = []
    for line in lines:
        for point in line:
            if point[1] >= far_line_thresh:
                near_lines.append(line)
                break
    return near_lines


def gt_cluster_line_points(gt_lines, img_h, img_w):
    lines = []
    for i, gt_line in enumerate(gt_lines):
        # 在每个gt-line绘制独立的mask，该mask中所有的点均为该实例的点
        gt_confidence = np.zeros((img_h, img_w), dtype=np.float32)
        pre_point = gt_line['points'][0]
        for cur_point in gt_line['points'][1:]:
            x1, y1 = round(pre_point[0]), round(pre_point[1])
            x2, y2 = round(cur_point[0]), round(cur_point[1])
            cv2.line(gt_confidence, (x1, y1), (x2, y2), (1,))
            pre_point = cur_point

        # gt_confidence[1000, :] = 0
        # plt.imshow(gt_confidence)
        # plt.show()

        ys, xs = np.nonzero(gt_confidence)
        ys, xs = np.flipud(ys), np.flipud(xs)  # 优先将y大的点排在前面
        line = []
        for x, y in zip(xs, ys):
            line.append((x, y))
        lines.append(line)
    return lines


def remove_isolated_points(line_points):
    line_points = np.array(line_points)
    valid_points = []
    for point in line_points:
        distance = abs(point - line_points).max(axis=1)
        if np.any(distance == 1):
            valid_points.append(point.tolist())
    return valid_points


def compute_vertical_distance(point, selected_points):
    vertical_points = [s_pnt for s_pnt in selected_points if s_pnt[0] == point[0]]

    if len(vertical_points) == 0:
        return 0
    else:
        vertical_distance = 10000
        for v_pnt in vertical_points:
            distance = abs(v_pnt[1] - point[1])
            # 得到在y方向的最近距离
            vertical_distance = distance if distance < vertical_distance else vertical_distance
        return vertical_distance


def select_function_points(selected_points, near_points):
    while len(near_points) > 0:
        added_points = []
        for n_pnt in near_points:
            # 与所有的selected_points进行计算
            for s_pnt in selected_points:
                # 同时满足在水平和垂直方向与selected_points相差一个像素的以内的点周围8个点
                distance = max(abs(n_pnt[0] - s_pnt[0]), abs(n_pnt[1] - s_pnt[1]))

                if distance == 1:
                    vertical_distance = compute_vertical_distance(n_pnt, selected_points)

                    # 在y方向的像素距离小于1
                    if vertical_distance <= 1:
                        selected_points = [n_pnt] + selected_points
                        added_points.append(n_pnt)
                        break

        # 如果不存在added_points,此时被断开,为不连续的点
        if len(added_points) == 0:
            break
        else:
            near_points = [n_pnt for n_pnt in near_points if n_pnt not in added_points]
    return selected_points, near_points


def extend_endpoints(selected_points, single_line):
    min_x, max_x = 10000, 0
    left_endpoints, right_endpoints = [], []
    # 得到最左边和最右边的点
    for s_pnt in selected_points:
        if s_pnt[0] == min_x:
            left_endpoints.append(s_pnt)
        elif s_pnt[0] < min_x:
            left_endpoints.clear()
            left_endpoints.append(s_pnt)
            min_x = s_pnt[0]
        if s_pnt[0] == max_x:
            right_endpoints.append(s_pnt)
        elif s_pnt[0] > max_x:
            right_endpoints.clear()
            right_endpoints.append(s_pnt)
            max_x = s_pnt[0]
    for x, y in left_endpoints:
        while (x - 1, y) in single_line:
            selected_points.append((x - 1, y))
            x -= 1
    for x, y in right_endpoints:
        while (x + 1, y) in single_line:
            selected_points.append((x + 1, y))
            x += 1
    return selected_points


def arrange_points_to_line(selected_points):
    selected_points = np.array(selected_points)
    xs, ys = selected_points[:, 0], selected_points[:, 1]
    image_xs = xs
    image_ys = ys

    # image_xs = x_map[ys, xs]
    # image_ys = y_map[ys, xs]
    # confidences = confidence_map[ys, xs]

    # pred_cls_map = np.argmax(pred_cls_map, axis=0)
    # emb_ids = pred_emb_id_map[ys, xs]
    # clses = pred_cls_map[ys, xs]

    indices = image_xs.argsort()
    image_xs = image_xs[indices]
    image_ys = image_ys[indices]
    # confidences = confidences[indices]
    # h, w = map_size
    line = []
    for x, y in zip(image_xs, image_ys):
        conf = 1.0
        emb_id = 0.0
        cls = 0
        # x = min(x, w - 1)
        # y = min(y, h - 1)
        line.append((x, y, conf, emb_id, cls))
    return line


def compute_point_distance(point_0, point_1):
    distance = np.sqrt((point_0[0] - point_1[0]) ** 2 + (point_0[1] - point_1[1]) ** 2)
    return distance


def connect_piecewise_lines(piecewise_lines, endpoint_distance=16):
    long_lines = piecewise_lines
    final_lines = []
    while len(long_lines) > 1:
        # 得到所有点的起始点和终点
        current_line = long_lines[0]
        current_endpoints = [current_line[0], current_line[-1]]
        other_lines = long_lines[1:]
        other_endpoints = []
        for o_line in other_lines:
            other_endpoints.append(o_line[0])
            other_endpoints.append(o_line[-1])
        point_ids = [None, None]
        min_dist = 10000

        # 为第一条曲线的起点和终点找最近距离的曲线
        for i, c_end in enumerate(current_endpoints):
            for j, o_end in enumerate(other_endpoints):
                distance = compute_point_distance(c_end, o_end)
                if distance < min_dist:
                    point_ids[0] = i
                    point_ids[1] = j
                    min_dist = distance

        # 最小距离满足合并的条件
        if min_dist < endpoint_distance:
            # 找到合并的曲线
            adjacent_line = other_lines[point_ids[1] // 2]
            other_lines.remove(adjacent_line)
            # 参考线的起点、合并线的起点
            if point_ids[0] == 0 and point_ids[1] % 2 == 0:
                adjacent_line.reverse()
                left_line = adjacent_line
                right_line = current_line
            # 参考线的起点、合并线的终点
            elif point_ids[0] == 0 and point_ids[1] % 2 == 1:
                left_line = adjacent_line
                right_line = current_line

            # 参考线的终点、合并线的起点
            elif point_ids[0] == 1 and point_ids[1] % 2 == 0:
                left_line = current_line
                right_line = adjacent_line

            # 参考线的终点、合并线的终点
            elif point_ids[0] == 1 and point_ids[1] % 2 == 1:
                left_line = current_line
                adjacent_line.reverse()
                right_line = adjacent_line
            # right_endpoints = left_line[-self.endpoint_precision:]  # find a pair of nearest points
            # left_endpoints = right_line[:self.endpoint_precision]
            # best_ids = [None, None]
            # min_dist = 10000
            # for i, r_end in enumerate(right_endpoints):
            # for j, l_end in enumerate(left_endpoints):
            # distance = self.compute_point_distance(r_end, l_end)
            # if distance < min_dist:
            # best_ids[0] = i
            # best_ids[1] = j
            # min_dist = distance
            # best_id = best_ids[0] - 2 if (best_ids[0] - 2 != 0) else None
            # left_line = left_line[:best_id]
            # best_id = best_ids[1]
            # right_line = right_line[best_id:]

            # 合并完的曲线放回到long_lines中进行循环直到剩下最后一条线
            long_lines = other_lines + [left_line + right_line]
        else:
            final_lines.append(current_line)
            long_lines = other_lines
    final_lines.append(long_lines[0])

    # 根据长度进行判断
    final_lines = sorted(final_lines, key=lambda l: len(l), reverse=True)
    return final_lines


def serialize_single_line(single_line):
    existing_points = single_line.copy()
    piecewise_lines = []

    # 终止的条件是只存在单个孤立的点,在此过程得到多段连续的线(不断开)
    while len(existing_points) > 0:
        # 计算每个点与线上所有点的距离,如果没有相邻的点,那么将该点剔除
        existing_points = remove_isolated_points(existing_points)
        if len(existing_points) == 0:
            break

        # 从existing_points得到距离最近的一个点, 也就是y最大的一个点
        y = np.array(existing_points)[:, 1].max()
        selected_points, alternative_points = [], []

        # 将上面选择的点作为selected_points,其他点作为alternative_points
        for e_pnt in existing_points:
            if e_pnt[1] == y and len(selected_points) == 0:
                selected_points.append(e_pnt)
            else:
                alternative_points.append(e_pnt)

        # 从近到远,得到多段连续的线
        y -= 1

        # 相当于区域生成,形成线段的过程
        while len(alternative_points) > 0:
            near_points, far_points = [], []

            # 找到在y方向是一个像素内的点
            for a_pnt in alternative_points:
                if a_pnt[1] >= y:
                    near_points.append(a_pnt)
                else:
                    far_points.append(a_pnt)
            if len(near_points) == 0:
                break

            # 将与selected_points相邻的点新增到selected_points中,该过程为迭代的过程,直到线被断开(在特定的y范围内)
            selected_points, outliers = select_function_points(
                selected_points, near_points
            )

            # 在得到连续线的过程没有新增加
            if len(outliers) == len(near_points):
                break
            else:
                alternative_points = outliers + far_points
                y -= 1

        # print("end region growth:", y)
        # selected_points点的左边右边, 在single_line是否存在y相同的点
        # 这是为了补充当左右延伸的点断开不相邻时,但是在左右是处于同一y高度上的点(左右水平延伸的点)
        selected_points = extend_endpoints(selected_points, single_line)

        # 对选择的点,按照从左到右的方式进行排序
        piecewise_line = arrange_points_to_line(
            selected_points
        )

        # piecewise_line = self.fit_points_to_line(selected_points, x_map, y_map, confidence_map)  # Curve Fitting
        # 得到分段的曲线
        piecewise_lines.append(piecewise_line)
        existing_points = alternative_points

    if len(piecewise_lines) == 0:
        return []
    elif len(piecewise_lines) == 1:
        exact_lines = piecewise_lines[0]
    else:
        # 计算piecewise_lines之间的距离,选择距离近的进行合并
        exact_lines = connect_piecewise_lines(piecewise_lines)[0]

    # 按照由近到远的方式进行排序
    if exact_lines[0][1] < exact_lines[-1][1]:
        exact_lines.reverse()
    return exact_lines


def simulate_model_decode(gt_lines, img_h, img_w):
    # 得到每条线的实例的点集
    raw_lines = gt_cluster_line_points(gt_lines, img_h, img_w)

    # 将点的数量少的线过滤
    raw_lines = remove_short_lines(raw_lines)

    # 将全部点都在远处的点过滤
    raw_lines = remove_far_lines(raw_lines)

    exact_lines = []
    for each_line in raw_lines:
        single_line = serialize_single_line(each_line)
        if len(single_line) > 0:
            exact_lines.append(single_line)
    return exact_lines


def debug_model_decode():
    img_path = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230904_mmdet3d/train/images/1689848680876640844.jpg"
    json_path = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230904_mmdet3d/train/jsons/1689848680876640844.json"
    s_root = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data"

    down_scale = 0.5

    ann = {}
    ann["img_path"] = img_path
    ann["ann_path"] = json_path
    # classes = ['glass_edge', 'glass_edge_up_plant', 'glass_edge_up_build']
    classes = ['road_boundary_line', 'bushes_boundary_line', 'fence_boundary_line',
               'stone_boundary_line', 'wall_boundary_line', 'water_boundary_line',
               'snow_boundary_line', 'manhole_boundary_line', 'others_boundary_line']
    num_classes = len(classes)
    ann_info = parse_ann_info(ann, classes, down_scale)
    img = cv2.imread(img_path)

    img_h, img_w, _ = img.shape
    img_h = int(img_h * down_scale)
    img_w = int(img_w * down_scale)
    img = cv2.resize(img, (img_w, img_h))
    img_show = copy.deepcopy(img)

    gt_lines = ann_info["gt_lines"]
    gt_lines = simulate_model_decode(gt_lines, img_h, img_w)

    for gt_line in gt_lines:
        # print(gt_line)
        # label = gt_line['label']
        # line_points = gt_line['points']
        line_points = gt_line
        # index = gt_line['index'] + 1  # 序号从0开始的
        # line_id = gt_line['line_id'] + 1
        # category_id = gt_line['category_id'] + 1

        pre_point = line_points[0]
        for cur_point in line_points[1:]:
            x1, y1 = round(pre_point[0]), round(pre_point[1])
            x2, y2 = round(cur_point[0]), round(cur_point[1])
            # cv2.line(line_map, (x1, y1), (x2, y2), (index,))
            # cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,))

            # cls_value = self.class_dic[label] + 1
            cv2.line(img_show, (x1, y1), (x2, y2), (255, 0, 255), 2)
            pre_point = cur_point

    plt.imshow(img_show[:, :, ::-1])
    plt.show()


def show_debug_piecewise_lines():
    # 经过区域生长后,形成多个分段的曲线
    img_path = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230904_mmdet3d/train/images/1689848680876640844.jpg"

    line_path0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/0_line.npy"
    line_path1 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1_line.npy"
    line_path2 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/2_line.npy"

    line0 = np.load(line_path0)
    line1 = np.load(line_path1)
    line2 = np.load(line_path2)

    img = cv2.imread(img_path)
    mask = np.zeros_like(img)

    for i, (line, color) in enumerate(zip([line0, line1, line2],
                           [(255, 0, 0), (0, 255, 0), (0, 0, 255)])):
        if i != 0:
            continue

        pre_point = line[0]
        for cur_point in line[1:]:
            x1, y1 = round(pre_point[0]), round(pre_point[1])
            x2, y2 = round(cur_point[0]), round(cur_point[1])
            # cv2.line(img, (x1, y1), (x2, y2), color, 2)
            # cv2.circle(img, (x1, y1), 5, color, 1)
            # cv2.circle(mask, (), 5, color, 1)

            mask[y1, x1, :] = color

            pre_point = cur_point

    # plt.imshow(img[:, :, ::-1])
    plt.imshow(mask[:, :, ::-1])
    plt.show()


def show_debug_outliers_points():
    # 经过区域生长后,形成多个分段的曲线
    img_path = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230904_mmdet3d/train/images/1689848680876640844.jpg"

    line_path0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/0_line.npy"
    line_path1 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1_line.npy"
    line_path2 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/2_line.npy"

    outliers_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/outliers.npy"

    line0 = np.load(line_path0)
    line1 = np.load(line_path1)
    line2 = np.load(line_path2)
    outliers = np.load(outliers_path)

    img = cv2.imread(img_path)
    mask = np.zeros_like(img)

    # for i, (line, color) in enumerate(zip([line0, line1, line2],
    #                        [(255, 0, 0), (0, 255, 0), (0, 0, 255)])):
    #     for points in line:
    #         x1, y1 = round(points[0]), round(points[1])
    #         img[y1, x1, :] = color
    #         mask[y1, x1, :] = color

    for cur_point in line0:
        x1, y1 = round(cur_point[0]), round(cur_point[1])
        mask[y1, x1, :] = (255, 0, 0)

    for outlier in outliers:
        x1, y1 = round(outlier[0]), round(outlier[1])
        mask[y1, x1, :] = (0, 255, 0)

    # mask[1092:-1, 1577, :] = (0, 0, 255)
    mask[1092:-1, 1577, :] = (255, 0, 255)
    mask[1091:-1, 1576, :] = (0, 255, 0)
    # plt.imshow(img[:, :, ::-1])
    plt.imshow(mask[:, :, ::-1])
    plt.show()



def show_debug_add_points():
    line_path0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/0_line.npy"
    line_path1 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1_line.npy"
    line_path2 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/2_line.npy"
    outliers_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/outliers.npy"

    line0 = np.load(line_path0)
    line1 = np.load(line_path1)
    line2 = np.load(line_path2)
    outliers = np.load(outliers_path)

    # selected_points = line2[:, :2]
    selected_points = line0[:, :2]
    near_points = line2[:, :2]
    # near_points = outliers

    selected_points = selected_points.tolist()
    near_points = near_points.tolist()

    added_points = []
    for n_pnt in near_points:
        # 与所有的selected_points进行计算
        for s_pnt in selected_points:
            # 存在一个满足在水平和垂直方向与selected_points相差一个像素的以内的点,即n_pnt周围8个点
            distance = max(abs(n_pnt[0] - s_pnt[0]), abs(n_pnt[1] - s_pnt[1]))
            if distance <= 1.0:
                print(n_pnt, s_pnt,  distance)

            if distance == 1:
                # 满足上述的条件后, 查看selected_points中是否有与n_pnt的x坐标相同的点,
                # 若没有, vertical_distance=0, 若有, 则计算这些点中最近的垂直方向距离
                # 这样的过滤条件可能会将连续的曲线截断,如下面的方法1□作为near_points, [2, 3, 4, 5]作为selected_points
                # 1□找到满足邻近距离为1的2□,然后在selected_points中找到相同x的5□,得到的最近垂直方向距离为3.那么1□不会加到selected_points中
                # 这样的处理可能是为了在拐弯的时候不会将1和5连接在一起
                #  1□
                # 2□
                # 3□
                # 4□
                #  5□
                vertical_distance = compute_vertical_distance(n_pnt, selected_points)
                print("vertical_distance:",  n_pnt, vertical_distance, s_pnt in selected_points)
                # 在y方向的像素距离小于1
                if vertical_distance <= 1:
                    selected_points = [n_pnt] + selected_points
                    added_points.append(n_pnt)
                    break

    print(added_points)
    # 如果不存在added_points,此时被断开,为不连续的点
    # if len(added_points) == 0:
    #     break


def debug_connect_lines():
    line_path0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/0_line.npy"
    line_path1 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1_line.npy"
    line_path2 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/2_line.npy"
    img_path = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230904_mmdet3d/train/images/1689848680876640844.jpg"

    line0 = np.load(line_path0)
    line1 = np.load(line_path1)
    line2 = np.load(line_path2)
    img = cv2.imread(img_path)

    piecewise_lines = [line0.tolist(), line1.tolist(), line2.tolist()]
    exact_lines = connect_piecewise_lines(piecewise_lines)[0]

    pre_point = exact_lines[0]
    for cur_point in exact_lines[1:]:
        x1, y1 = round(pre_point[0]), round(pre_point[1])
        x2, y2 = round(cur_point[0]), round(cur_point[1])

        color = (0, 255, 0)
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
        # cv2.circle(img, (x1, y1), 5, color, 1)
        # cv2.circle(mask, (), 5, color, 1)
        pre_point = cur_point

    plt.imshow(img[:, :, ::-1])

    plt.show()





def debug_show_gt_lines():
    img_path = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_2023095_mmdet3d/train/images/1689848749353944455.jpg"
    json_path = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_2023095_mmdet3d/train/jsons/1689848749353944455.json"
    s_root = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data"

    down_scale = 0.5

    ann = {}
    ann["img_path"] = img_path
    ann["ann_path"] = json_path
    # classes = ['glass_edge', 'glass_edge_up_plant', 'glass_edge_up_build']
    classes = ['road_boundary_line', 'bushes_boundary_line', 'fence_boundary_line',
               'stone_boundary_line', 'wall_boundary_line', 'water_boundary_line',
               'snow_boundary_line', 'manhole_boundary_line', 'others_boundary_line']
    num_classes = len(classes)
    ann_info = parse_ann_info(ann, classes, down_scale)
    img = cv2.imread(img_path)

    img_h, img_w, _ = img.shape
    img_h = int(img_h * down_scale)
    img_w = int(img_w * down_scale)
    img = cv2.resize(img, (img_w, img_h))
    img_show = copy.deepcopy(img)

    gt_lines = ann_info["gt_lines"]
    # gt_lines = simulate_model_decode(gt_lines, img_h, img_w)

    flip = 1
    if flip:
        img_show = copy.deepcopy(img_show[:, ::-1, :])


    for gt_line in gt_lines:
        # print(gt_line)
        # label = gt_line['label']
        line_points = np.array(gt_line['points'])
        line_points = filter_near_same_points(line_points)
        if flip:
            line_points[:, 0] = img_w - line_points[:, 0]
            line_points = line_points[::-1, :]

        # line_points = gt_line
        # index = gt_line['index'] + 1  # 序号从0开始的
        # line_id = gt_line['line_id'] + 1
        # category_id = gt_line['category_id'] + 1

        pre_point = line_points[0]
        for i, cur_point in enumerate(line_points[1:]):
            x1, y1 = round(pre_point[0]), round(pre_point[1])
            x2, y2 = round(cur_point[0]), round(cur_point[1])
            # if np.all(cur_point == pre_point):
            #     # print(cur_point, pre_point)
            #     print("same")
            #     continue

            # cv2.line(line_map, (x1, y1), (x2, y2), (index,))
            # cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,))

            # cls_value = self.class_dic[label] + 1
            cv2.line(img_show, (x1, y1), (x2, y2), (255, 0, 255), 1)
            # 求每个点的方向
            if i % 50 == 0:
                orient = cal_points_orient(pre_point, cur_point)
                # 当pre_point和cur_point为同一个点时,朝向为-1
                if orient != -1:
                    # if orient == 180 or orient==0:
                    # continue
                    # print(orient)
                    # img_show = draw_orient(img_show, pre_point, orient, arrow_len=20, color=(0, 0, 255))
                    # 转个90度,指向草地
                    # orient = orient + 90
                    # if orient > 360:
                    #     orient = orient - 360

                    img_show = draw_orient(img_show, pre_point, orient, arrow_len=20)
                else:
                    print(pre_point[0], pre_point[1], cur_point[0], cur_point[1])



            pre_point = cur_point
    cv2.imwrite("/home/dell/liyongjing/documents/20230907/2.jpg", img_show)
    plt.imshow(img_show[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    # 对模型decode的部分进行理解
    # debug_model_decode()

    # show debug中间的过程
    # 调试分段曲线,查看分段的分段的曲线分别是怎样的
    # show_debug_piecewise_lines()

    # 调试曲线分段的条件, 可以看到分段的曲线,以及outliers_points(不会合并到曲线中的点)是怎样的
    # show_debug_outliers_points()

    # 判断add points的条件(满足怎样的条件可以加入到piecewise_line中),其实也就是区域生长的条件
    # show_debug_add_points()

    # 调用曲线段合并的条件
    # debug_connect_lines()

    # 将gt-line的orient可视化
    debug_show_gt_lines()

