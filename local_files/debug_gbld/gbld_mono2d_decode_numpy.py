# import torch
# import torch.nn as nn
# from mmengine.model import BaseModule
import math
import matplotlib.pyplot as plt
# from mmdet3d.registry import MODELS
import numpy as np
import cv2

color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cluster_line_points(xs, ys, embeddings, pull_margin=0.8):
    lines = []
    embedding_means = []
    point_numbers = []
    for x, y, eb in zip(xs, ys, embeddings):
        id = None
        min_dist = 10000
        for i, eb_mean in enumerate(embedding_means):
            distance = abs(eb - eb_mean)
            if distance < pull_margin and distance < min_dist:
                id = i
                min_dist = distance
        if id == None:
            lines.append([(x, y)])
            embedding_means.append(eb)
            point_numbers.append(1)
        else:
            lines[id].append((x, y))
            embedding_means[id] = (embedding_means[id] * point_numbers[id] + eb) / (
                point_numbers[id] + 1
            )
            point_numbers[id] += 1
    return lines


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
            vertical_distance = distance if distance < vertical_distance else vertical_distance
        return vertical_distance


def select_function_points(selected_points, near_points):
    while len(near_points) > 0:
        added_points = []
        for n_pnt in near_points:
            for s_pnt in selected_points:
                distance = max(abs(n_pnt[0] - s_pnt[0]), abs(n_pnt[1] - s_pnt[1]))
                if distance == 1:
                    vertical_distance = compute_vertical_distance(n_pnt, selected_points)
                    if vertical_distance <= 1:
                        selected_points = [n_pnt] + selected_points
                        added_points.append(n_pnt)
                        break
        if len(added_points) == 0:
            break
        else:
            near_points = [n_pnt for n_pnt in near_points if n_pnt not in added_points]
    return selected_points, near_points


def extend_endpoints(selected_points, single_line):
    min_x, max_x = 10000, 0
    left_endpoints, right_endpoints = [], []
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


def arrange_points_to_line(selected_points, x_map, y_map, confidence_map,
                           pred_emb_id_map, pred_cls_map, pred_orient_map=None,
                           pred_visible_map=None, pred_hanging_map=None, pred_covered_map=None,
                           map_size=(1920, 1080)):

    selected_points = np.array(selected_points)
    xs, ys = selected_points[:, 0], selected_points[:, 1]
    image_xs = x_map[ys, xs]
    image_ys = y_map[ys, xs]
    confidences = confidence_map[ys, xs]

    pred_cls_map = np.argmax(pred_cls_map, axis=0)
    # import matplotlib.pyplot as plt
    # plt.imshow(pred_cls_map)
    # plt.show()
    # exit(1)


    emb_ids = pred_emb_id_map[ys, xs]
    clses = pred_cls_map[ys, xs]

    indices = image_xs.argsort()
    image_xs = image_xs[indices]
    image_ys = image_ys[indices]
    confidences = confidences[indices]

    emb_ids = emb_ids[indices]
    clses = clses[indices]

    if pred_orient_map is not None:
        orients = pred_orient_map[ys, xs]
        orients = orients[indices]
    else:
        orients = [-1] * len(clses)

    if pred_visible_map is not None:
        visibles = pred_visible_map[ys, xs]
        visibles = visibles[indices]
    else:
        visibles = [-1] * len(clses)

    if pred_hanging_map is not None:
        hangings = pred_hanging_map[ys, xs]
        hangings = hangings[indices]
    else:
        hangings = [-1] * len(clses)

    if pred_covered_map is not None:
        covereds = pred_covered_map[ys, xs]
        covereds = covereds[indices]
    else:
        covereds = [-1] * len(clses)

    h, w = map_size
    line = []
    for x, y, conf, emb_id, cls, orient, visible, hanging, covered in\
            zip(image_xs, image_ys, confidences, emb_ids, clses, orients, visibles, hangings, covereds):
        x = min(x, w - 1)
        y = min(y, h - 1)

        # 转回到MEBOW的表示
        if 180 >= orient >= 0:
            orient = 90 + (180 - orient)
        elif orient >= -90:
            orient = 270 + abs(orient)
        elif orient >= -180:
            orient = abs(orient) - 90
        else:
            orient = -1

        line.append((x, y, conf, emb_id, cls, orient, visible, hanging, covered))
    return line


def compute_point_distance(point_0, point_1):
    distance = np.sqrt((point_0[0] - point_1[0]) ** 2 + (point_0[1] - point_1[1]) ** 2)
    return distance


def connect_piecewise_lines(piecewise_lines, endpoint_distance=16):
    long_lines = piecewise_lines
    final_lines = []
    while len(long_lines) > 1:
        # 选择第一条线
        current_line = long_lines[0]
        current_endpoints = [current_line[0], current_line[-1]]

        # 其他线
        other_lines = long_lines[1:]
        other_endpoints = []
        for o_line in other_lines:
            other_endpoints.append(o_line[0])
            other_endpoints.append(o_line[-1])
        point_ids = [None, None]
        min_dist = 10000

        # 对于选择线的两个端点, 在其他线中的所有端点, 查找与两个选择端点距离最近的距离
        for i, c_end in enumerate(current_endpoints):
            for j, o_end in enumerate(other_endpoints):
                distance = compute_point_distance(c_end, o_end)
                if distance < min_dist:
                    point_ids[0] = i
                    point_ids[1] = j
                    min_dist = distance

        # 如果两个选择端点存在距离小于阈值的端点, 则两条线合并, 合并完毕后继续进行类似的合并操作
        if min_dist < endpoint_distance:
            adjacent_line = other_lines[point_ids[1] // 2]
            other_lines.remove(adjacent_line)

            # 判断两条合并线的左右关系, 进行合并
            if point_ids[0] == 0 and point_ids[1] % 2 == 0:
                adjacent_line.reverse()
                left_line = adjacent_line
                right_line = current_line
            elif point_ids[0] == 0 and point_ids[1] % 2 == 1:
                left_line = adjacent_line
                right_line = current_line
            elif point_ids[0] == 1 and point_ids[1] % 2 == 0:
                left_line = current_line
                right_line = adjacent_line
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
            long_lines = other_lines + [left_line + right_line]

        # 如果如果两个选择端点不存在距离小于阈值的端点, 给线将直接输出
        else:
            final_lines.append(current_line)
            long_lines = other_lines
    final_lines.append(long_lines[0])
    final_lines = sorted(final_lines, key=lambda l: len(l), reverse=True)
    return final_lines


def serialize_single_line(single_line, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map,
                          pred_orient_map=None, pred_visible_map=None, pred_hanging_map=None,
                          pred_covered_map=None):

    existing_points = single_line.copy()
    piecewise_lines = []
    while len(existing_points) > 0:
        existing_points = remove_isolated_points(existing_points)
        if len(existing_points) == 0:
            break
        y = np.array(existing_points)[:, 1].max()
        selected_points, alternative_points = [], []
        for e_pnt in existing_points:
            if e_pnt[1] == y and len(selected_points) == 0:
                selected_points.append(e_pnt)
            else:
                alternative_points.append(e_pnt)

        # 在y方向进行延伸
        y -= 1
        while len(alternative_points) > 0:
            near_points, far_points = [], []
            for a_pnt in alternative_points:
                if a_pnt[1] >= y:
                    near_points.append(a_pnt)
                else:
                    far_points.append(a_pnt)
            if len(near_points) == 0:
                break
            selected_points, outliers = select_function_points(
                selected_points, near_points
            )
            if len(outliers) == len(near_points):
                break
            else:
                alternative_points = outliers + far_points
                y -= 1

        # 在x方向进行延伸
        selected_points = extend_endpoints(selected_points, single_line)
        piecewise_line = arrange_points_to_line(
            selected_points, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map, pred_orient_map,
            pred_visible_map, pred_hanging_map, pred_covered_map
        )
        # piecewise_line = self.fit_points_to_line(selected_points, x_map, y_map, confidence_map)  # Curve Fitting
        piecewise_lines.append(piecewise_line)
        existing_points = alternative_points
    #
    # grid_size = 4
    # img_h, img_w = x_map.shape
    # img_piecewise_lines = np.ones((img_h * grid_size, img_w * grid_size, 3), dtype=np.uint8) * 255
    # for i, raw_line in enumerate(piecewise_lines):
    #     color = color_list[i]
    #     pre_point = raw_line[0]
    #
    #     for cur_point in raw_line[1:]:
    #         # x, y = cur_point[:2]
    #         # img_piecewise_lines[y, x] = color
    #         x1, y1 = int(pre_point[0]), int(pre_point[1])
    #         x2, y2 = int(cur_point[0]), int(cur_point[1])
    #
    #         cv2.line(img_piecewise_lines, (x1, y1), (x2, y2), color, 1, 8)
    #         pre_point = cur_point
    #
    #     start_point = raw_line[0]
    #     end_point = raw_line[-1]
    #     cv2.circle(img_piecewise_lines, (int(start_point[0]), int(start_point[1])), 10, color, -1)
    #     cv2.circle(img_piecewise_lines, (int(end_point[0]), int(end_point[1])), 10, color, -1)
    #
    # plt.subplot(2, 1, 1)
    # plt.imshow(confidence_map)
    # plt.subplot(2, 1, 2)
    # plt.imshow(img_piecewise_lines)
    # plt.show()
    # exit(1)

    if len(piecewise_lines) == 0:
        return []
    elif len(piecewise_lines) == 1:
        exact_lines = piecewise_lines[0]
    else:

        exact_lines = connect_piecewise_lines(piecewise_lines, endpoint_distance=16)[0]
        # all_exact_lines = connect_piecewise_lines(piecewise_lines, endpoint_distance=16)
    #
    # grid_size = 4
    # img_h, img_w = x_map.shape
    # img_exact_lines = np.ones((img_h * grid_size, img_w * grid_size, 3), dtype=np.uint8) * 255
    # # for i, raw_line in enumerate([exact_lines]):
    # for i, raw_line in enumerate(all_exact_lines):
    #     color = color_list[i]
    #     pre_point = raw_line[0]
    #
    #     for cur_point in raw_line[1:]:
    #         # x, y = cur_point[:2]
    #         # img_piecewise_lines[y, x] = color
    #         x1, y1 = int(pre_point[0]), int(pre_point[1])
    #         x2, y2 = int(cur_point[0]), int(cur_point[1])
    #
    #         cv2.line(img_exact_lines, (x1, y1), (x2, y2), color, 1, 8)
    #         pre_point = cur_point
    #     start_point = raw_line[0]
    #     end_point = raw_line[-1]
    #     cv2.circle(img_exact_lines, (int(start_point[0]), int(start_point[1])), 10, color, -1)
    #     cv2.circle(img_exact_lines, (int(end_point[0]), int(end_point[1])), 10, color, -1)
    #
    # plt.subplot(2, 1, 1)
    # plt.imshow(confidence_map)
    # plt.subplot(2, 1, 2)
    # plt.imshow(img_exact_lines)
    # plt.show()
    # exit(1)

    if exact_lines[0][1] < exact_lines[-1][1]:
        exact_lines.reverse()
    return exact_lines


def serialize_all_lines(single_line, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map,
                          pred_orient_map=None, pred_visible_map=None, pred_hanging_map=None,
                          pred_covered_map=None):

    existing_points = single_line.copy()
    piecewise_lines = []
    while len(existing_points) > 0:
        existing_points = remove_isolated_points(existing_points)
        if len(existing_points) == 0:
            break
        y = np.array(existing_points)[:, 1].max()
        selected_points, alternative_points = [], []
        for e_pnt in existing_points:
            if e_pnt[1] == y and len(selected_points) == 0:
                selected_points.append(e_pnt)
            else:
                alternative_points.append(e_pnt)

        # 在y方向进行延伸
        y -= 1
        while len(alternative_points) > 0:
            near_points, far_points = [], []
            for a_pnt in alternative_points:
                if a_pnt[1] >= y:
                    near_points.append(a_pnt)
                else:
                    far_points.append(a_pnt)
            if len(near_points) == 0:
                break
            selected_points, outliers = select_function_points(
                selected_points, near_points
            )
            if len(outliers) == len(near_points):
                break
            else:
                alternative_points = outliers + far_points
                y -= 1

        # 在x方向进行延伸
        selected_points = extend_endpoints(selected_points, single_line)
        piecewise_line = arrange_points_to_line(
            selected_points, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map, pred_orient_map,
            pred_visible_map, pred_hanging_map, pred_covered_map
        )
        # piecewise_line = self.fit_points_to_line(selected_points, x_map, y_map, confidence_map)  # Curve Fitting
        piecewise_lines.append(piecewise_line)
        existing_points = alternative_points
    #
    # grid_size = 4
    # img_h, img_w = x_map.shape
    # img_piecewise_lines = np.ones((img_h * grid_size, img_w * grid_size, 3), dtype=np.uint8) * 255
    # for i, raw_line in enumerate(piecewise_lines):
    #     color = color_list[i]
    #     pre_point = raw_line[0]
    #
    #     for cur_point in raw_line[1:]:
    #         # x, y = cur_point[:2]
    #         # img_piecewise_lines[y, x] = color
    #         x1, y1 = int(pre_point[0]), int(pre_point[1])
    #         x2, y2 = int(cur_point[0]), int(cur_point[1])
    #
    #         cv2.line(img_piecewise_lines, (x1, y1), (x2, y2), color, 1, 8)
    #         pre_point = cur_point
    #
    #     start_point = raw_line[0]
    #     end_point = raw_line[-1]
    #     cv2.circle(img_piecewise_lines, (int(start_point[0]), int(start_point[1])), 10, color, -1)
    #     cv2.circle(img_piecewise_lines, (int(end_point[0]), int(end_point[1])), 10, color, -1)
    #
    # plt.subplot(2, 1, 1)
    # plt.imshow(confidence_map)
    # plt.subplot(2, 1, 2)
    # plt.imshow(img_piecewise_lines)
    # plt.show()
    # exit(1)

    if len(piecewise_lines) == 0:
        return []
    elif len(piecewise_lines) == 1:
        exact_lines = piecewise_lines[0]
        all_exact_lines = piecewise_lines
    else:

        # exact_lines = connect_piecewise_lines(piecewise_lines, endpoint_distance=40)[0]
        all_exact_lines = connect_piecewise_lines(piecewise_lines, endpoint_distance=16)
    #
    # grid_size = 4
    # img_h, img_w = x_map.shape
    # img_exact_lines = np.ones((img_h * grid_size, img_w * grid_size, 3), dtype=np.uint8) * 255
    # # for i, raw_line in enumerate([exact_lines]):
    # for i, raw_line in enumerate(all_exact_lines):
    #     color = color_list[i]
    #     pre_point = raw_line[0]
    #
    #     for cur_point in raw_line[1:]:
    #         # x, y = cur_point[:2]
    #         # img_piecewise_lines[y, x] = color
    #         x1, y1 = int(pre_point[0]), int(pre_point[1])
    #         x2, y2 = int(cur_point[0]), int(cur_point[1])
    #
    #         cv2.line(img_exact_lines, (x1, y1), (x2, y2), color, 1, 8)
    #         pre_point = cur_point
    #     start_point = raw_line[0]
    #     end_point = raw_line[-1]
    #     cv2.circle(img_exact_lines, (int(start_point[0]), int(start_point[1])), 10, color, -1)
    #     cv2.circle(img_exact_lines, (int(end_point[0]), int(end_point[1])), 10, color, -1)
    #
    # plt.subplot(2, 1, 1)
    # plt.imshow(confidence_map)
    # plt.subplot(2, 1, 2)
    # plt.imshow(img_exact_lines)
    # plt.show()
    # exit(1)
    for all_exact_line in all_exact_lines:
        if all_exact_line[0][1] < all_exact_line[-1][1]:
            all_exact_line.reverse()

    # if exact_lines[0][1] < exact_lines[-1][1]:
    #     exact_lines.reverse()
    return all_exact_lines


def connect_line_points(mask_map, embedding_map, x_map, y_map,
                        confidence_map, pred_emb_id, pred_cls_map,
                        pred_orient_map=None, pred_visible_map=None,
                        pred_hanging_map=None, pred_covered_map=None, line_maximum=10):

    ys, xs = np.nonzero(mask_map)
    ys, xs = np.flipud(ys), np.flipud(xs)   # 优先将y大的点排在前面
    ebs = embedding_map[ys, xs]
    raw_lines = cluster_line_points(xs, ys, ebs)

    # 查看聚类的效果
    # img_h, img_w = x_map.shape
    # img_emb = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    # for i, raw_line in enumerate(raw_lines):
    #     color = color_list[i]
    #     for point in raw_line:
    #         x, y = point
    #         # img_emb[y, x] = (i + 1) * 100
    #         img_emb[y, x] = color
    #
    # plt.subplot(2, 1, 1)
    # plt.imshow(mask_map)
    # plt.subplot(2, 1, 2)
    # plt.imshow(img_emb)
    # plt.show()


    raw_lines = remove_short_lines(raw_lines)
    raw_lines = remove_far_lines(raw_lines)

    # img_h, img_w = x_map.shape
    # img_filter = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    # for i, raw_line in enumerate(raw_lines):
    #     color = color_list[i]
    #     for point in raw_line:
    #         x, y = point
    #         # img_emb[y, x] = (i + 1) * 100
    #         img_filter[y, x] = color
    #
    # plt.subplot(2, 1, 1)
    # plt.imshow(mask_map)
    # plt.subplot(2, 1, 2)
    # plt.imshow(img_filter)
    # plt.show()

    exact_lines = []
    for each_line in raw_lines:
        # 只选择最近的single line
        # single_line = serialize_single_line(each_line, x_map, y_map,
        #                                     confidence_map, pred_emb_id,
        #                                     pred_cls_map, pred_orient_map,
        #                                     pred_visible_map, pred_hanging_map, pred_covered_map
        #                                     )
        # if len(single_line) > 0:
        #     exact_lines.append(single_line)

        # 将满足长度大于特定长度的线都提取
        all_lines = serialize_all_lines(each_line, x_map, y_map,
                                            confidence_map, pred_emb_id,
                                            pred_cls_map, pred_orient_map,
                                            pred_visible_map, pred_hanging_map, pred_covered_map
                                            )
        if len(all_lines) > 0:
            single_line = all_lines[0]
            if len(single_line) > 0:
                exact_lines.append(single_line)

            if len(all_lines) > 1:
                for single_line in all_lines[1:]:
                    if len(single_line) > 50:
                        exact_lines.append(single_line)

    # grid_size = 4
    # img_h, img_w = x_map.shape
    # img_serialize = np.ones((img_h * grid_size, img_w * grid_size, 3), dtype=np.uint8) * 255
    # for i, raw_line in enumerate(exact_lines):
    #     color = color_list[i]
    #     for point in raw_line:
    #         x, y = point[:2]
    #         # img_emb[y, x] = (i + 1) * 100
    #         img_serialize[y, x] = color
    #
    # plt.subplot(2, 1, 1)
    # plt.imshow(mask_map)
    # plt.subplot(2, 1, 2)
    # plt.imshow(img_serialize)
    # plt.show()
    # exit(1)

    if len(exact_lines) == 0:
        return []
    exact_lines = sorted(exact_lines, key=lambda l: len(l), reverse=True)
    exact_lines = (
        exact_lines[: line_maximum]
        if len(exact_lines) > line_maximum
        else exact_lines
    )
    exact_lines = sorted(exact_lines, key=lambda l: l[0][0], reverse=False)
    # exact_lines = sorted(exact_lines, key=lambda l: np.array(l)[:, 0].mean(), reverse=False)
    return exact_lines


# 对点进行拟合
class FitPoints2Line():
    def __init__(self):
        self.fit_degree = [1, 3]    # 整数, 表示拟合多项式的度
        self.map_size = [608, 960]  # h, w
        self.fit_margin = 5         # 下面的数据都是大概猜测出来的,需要进行验证
        self.long_line_thresh = 50
        self.fit_iteration = 2
        self.fit_tolerance = 20

    def fit_curve(self, image_points):
        xs, ys = image_points
        assert xs.size >= 2, "Too few points to fit a curve."
        base, max_degree = self.fit_degree
        deg = min(np.ceil(xs.size / base), max_degree)
        coefficients = np.polyfit(xs, ys, deg)
        polynomial = np.poly1d(coefficients)
        fitting_ys = polynomial(xs).round().astype(np.int)
        h, w = self.map_size
        fitting_ys = np.clip(fitting_ys, 0, h - 1)
        fitting_points = [xs.copy(), fitting_ys]
        return fitting_points

    def search_inflexion(self, delta_ys, image_points):
        i_xs, i_ys = image_points
        indices = np.nonzero(delta_ys == delta_ys.max())[0]
        for id in indices:
            inf_x = i_xs[id]
            inf_y = i_ys[id]
            vertical_number = np.logical_and(
                i_xs > inf_x - self.fit_margin, i_xs < inf_x + self.fit_margin
            ).sum()
            horizontal_number = np.logical_and(
                i_ys > inf_y - self.fit_margin, i_ys < inf_y + self.fit_margin
            ).sum()
            if vertical_number <= horizontal_number:
                left_num = (i_xs <= inf_x).sum()
                right_num = (i_xs >= inf_x).sum()
                if left_num >= self.long_line_thresh and right_num >= self.long_line_thresh:
                    return ["x", inf_x]
            else:
                left_num = (i_ys <= inf_y).sum()
                right_num = (i_ys >= inf_y).sum()
                if left_num >= self.long_line_thresh and right_num >= self.long_line_thresh:
                    return ["y", inf_y]
        return None

    def split_fitting_points(self, inflexion, image_points, confidences, fitting_points):
        axis, threshold = inflexion
        i_xs, i_ys = image_points
        f_xs, f_ys = fitting_points
        if axis == "x":
            left_mask = i_xs <= threshold
            right_mask = i_xs >= threshold
        else:
            left_mask = i_ys <= threshold
            right_mask = i_ys >= threshold
        left_i_xs, right_i_xs = i_xs[left_mask], i_xs[right_mask]
        if left_i_xs[0] > right_i_xs[0]:
            left_i_xs, right_i_xs = right_i_xs, left_i_xs
            left_mask, right_mask = right_mask, left_mask
        left_i_ys, right_i_ys = i_ys[left_mask], i_ys[right_mask]
        left_confs, right_confs = confidences[left_mask], confidences[right_mask]
        left_f_xs, right_f_xs = f_xs[left_mask], f_xs[right_mask]
        left_f_ys, right_f_ys = f_ys[left_mask], f_ys[right_mask]
        return (
            [left_i_xs, left_i_ys],
            [right_i_xs, right_i_ys],
            left_confs,
            right_confs,
            [left_f_xs, left_f_ys],
            [right_f_xs, right_f_ys],
        )

    # 调用入口
    def fit_points_to_line(self, selected_points, x_map, y_map, confidence_map):
        selected_points = np.array(selected_points)
        xs, ys = selected_points[:, 0], selected_points[:, 1]
        image_xs = x_map[ys, xs]
        image_ys = y_map[ys, xs]
        confidences = confidence_map[ys, xs]
        indices = image_xs.argsort()
        image_xs = image_xs[indices]
        image_ys = image_ys[indices]
        confidences = confidences[indices]

        image_points = [[image_xs, image_ys]]
        confidences = [confidences]
        need_fitting = [True]
        fitting_points = [[]]

        for iter in range(self.fit_iteration):
            finished_image_points = []
            finished_confidences = []
            finished_need_fitting = []
            finished_fitting_points = []
            for i_pnts, confs, n_fit, f_pnts in zip(
                image_points, confidences, need_fitting, fitting_points
            ):
                if n_fit == True:
                    f_pnts = self.fit_curve(i_pnts)
                    delta_ys = abs(f_pnts[1] - i_pnts[1])
                    if delta_ys.max() >= self.fit_tolerance:
                        inflexion = self.search_inflexion(delta_ys, i_pnts)
                        if inflexion != None:
                            (
                                left_i_pnts,
                                right_i_pnts,
                                left_confs,
                                right_confs,
                                left_f_pnts,
                                right_f_pnts,
                            ) = self.split_fitting_points(inflexion, i_pnts, confs, f_pnts)
                            finished_image_points.append(left_i_pnts)
                            finished_image_points.append(right_i_pnts)
                            finished_confidences.append(left_confs)
                            finished_confidences.append(right_confs)
                            finished_need_fitting.append(True)
                            finished_need_fitting.append(True)
                            finished_fitting_points.append(left_f_pnts)
                            finished_fitting_points.append(right_f_pnts)
                        else:
                            finished_image_points.append(i_pnts)
                            finished_confidences.append(confs)
                            finished_need_fitting.append(False)
                            finished_fitting_points.append(f_pnts)
                    else:
                        finished_image_points.append(i_pnts)
                        finished_confidences.append(confs)
                        finished_need_fitting.append(False)
                        finished_fitting_points.append(f_pnts)
                else:
                    finished_image_points.append(i_pnts)
                    finished_confidences.append(confs)
                    finished_need_fitting.append(n_fit)
                    finished_fitting_points.append(f_pnts)
            image_points = finished_image_points
            confidences = finished_confidences
            need_fitting = finished_need_fitting
            fitting_points = finished_fitting_points
        final_line = []
        for i, (f_pnts, confs) in enumerate(zip(fitting_points, confidences)):
            f_xs, f_ys = f_pnts
            for j, pnt in enumerate(zip(f_xs, f_ys, confs)):
                if i >= 1 and j == 0:
                    last_pnt = final_line.pop()
                    pnt = (pnt[0], (0.5 * (pnt[1] + last_pnt[1])).round().astype(np.int), pnt[2])
                final_line.append(pnt)
        return final_line


# @MODELS.register_module()
# class GlasslandBoundaryLine2DDecode(BaseModule):
class GlasslandBoundaryLine2DDecodeNumpy():
    def __init__(self,
                 confident_t,
                 grid_size=4,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        # super().__init__(init_cfg)
        self.confident_t = confident_t   # 预测seg-heatmap的阈值
        self.grid_size = grid_size
        self.line_map_range = [0, -1]    # 不进行过滤

        # self.noise_filter = nn.MaxPool2d([3, 1], stride=1, padding=[1, 0])  # 去锯齿
    def get_line_cls(self, exact_curse_lines_multi_cls):
        output_curse_lines_multi_cls = []
        for exact_curse_lines in exact_curse_lines_multi_cls:
            lines = []
            for exact_curse_line in exact_curse_lines:
                points = np.array(exact_curse_line)
                poitns_cls = points[:, 4]
                # cls = np.argmax(np.bincount(poitns_cls.astype(np.int32)))

                # points[:, 4] = cls   # 统计线的类别

                lines.append(np.array(list(points)))

            output_curse_lines_multi_cls.append(lines)
        return output_curse_lines_multi_cls

    def cal_points_orient(self, pre_point, cur_point):
        # 得到pre_point指向cur_point的方向
        # 转为以pre_point为原点的坐标系
        x1, y1 = pre_point[0], pre_point[1]
        x2, y2 = cur_point[0], cur_point[1]
        x = x2 - x1
        y = y2 - y1

        # 转为与人体朝向的坐标定义类似，以正前方的指向为0，然后逆时针得到360的朝向
        # 记住图像的坐标系为y向下,x向右
        orient = -1
        if x != 0:
            angle = abs(math.atan(y / x)) / math.pi * 180
            # 判断指向所在的象限
            # 在3、4象限
            if y >= 0:
                # 在3象限
                if x < 0:
                    orient = 90 + angle
                # 在4象限
                else:
                    orient = 180 + (90 - angle)
            # 在1、2象限
            else:
                # 在1象限
                if x > 0:
                    orient = 270 + angle
                # 在2象限
                else:
                    orient = 90 - angle
        else:
            # 当x为0的时候
            if y >= 0:
                if y == 0:
                    orient = -1
                else:
                    orient = 180
            else:
                orient = 0
        return orient

    def get_line_orient(self, exact_curse_lines_multi_cls):
        output_curse_lines_with_orient = []
        for exact_curse_lines in exact_curse_lines_multi_cls:
            lines = []
            for exact_curse_line in exact_curse_lines:
                revese_num = [0, 0]

                pre_point = exact_curse_line[0]
                for cur_point in exact_curse_line[1:]:
                    x1, y1 = int(pre_point[0]), int(pre_point[1])
                    x2, y2 = int(cur_point[0]), int(cur_point[1])

                    line_orient = self.cal_points_orient(pre_point, cur_point)

                    orient = pre_point[5]
                    if orient != -1:
                        reverse = False  # 代表反向是否反了
                        orient_diff = abs(line_orient - orient)
                        if orient_diff > 180:
                            orient_diff = 360 - orient_diff

                        if orient_diff > 90:
                            reverse = True

                    pre_point = cur_point

                    if reverse:
                        revese_num[0] = revese_num[0] + 1
                    else:
                        revese_num[1] = revese_num[1] + 1

                # 判断是否需要调转顺序
                if revese_num[0] > revese_num[1]:
                    exact_curse_line = exact_curse_line[::-1, :]
                lines.append(exact_curse_line)
            output_curse_lines_with_orient.append(lines)
        return output_curse_lines_with_orient

    def forward(self, seg_pred, offset_pred, seg_emb_pred, connect_emb_pred, cls_pred, orient_pred=None,
                visible_pred=None, hanging_pred=None, covered_pred=None,):
        # 对pred_confidene 进行max-pooling
        # 应该是decode加上offset后,在进行h方向的max-pooling
        # seg_max_pooling = self.noise_filter(seg_pred)
        # mask = seg_pred == seg_max_pooling
        # seg_pred[~mask] = -1e6

        # seg_pred = seg_pred.cpu().detach().numpy()
        # offset_pred = offset_pred.cpu().detach().numpy()
        # seg_emb_pred = seg_emb_pred.cpu().detach().numpy()
        # connect_emb_pred = connect_emb_pred.cpu().detach().numpy()
        # cls_pred = cls_pred.cpu().detach().numpy()

        # if orient_pred is not None:
        #     orient_pred = orient_pred.cpu().detach().numpy()
        #
        # if visible_pred is not None:
        #     visible_pred = visible_pred.cpu().detach().numpy()
        #
        # if hanging_pred is not None:
        #     hanging_pred = hanging_pred.cpu().detach().numpy()
        #
        # if covered_pred is not None:
        #     covered_pred = covered_pred.cpu().detach().numpy()
        curse_lines_with_cls = self.decode_curse_line(seg_pred, offset_pred[0:1],
                                                  offset_pred[1:2], seg_emb_pred, connect_emb_pred,
                                                      cls_pred, orient_pred, visible_pred,
                                                      hanging_pred, covered_pred)

        curse_lines_with_cls = self.get_line_cls(curse_lines_with_cls)
        curse_lines_with_cls = self.get_line_orient(curse_lines_with_cls)
        curse_lines_with_cls = self.filter_lines(curse_lines_with_cls)

        return curse_lines_with_cls

    # 进行均值滤波
    def moving_average(self, x, window_size):
        kernel = np.ones(window_size) / window_size
        # kernel = np.array([1, 1, 1])
        result = np.correlate(x, kernel, mode='same')

        # 将边界效应里的设置为原来的数值
        valid_sie = window_size//2
        result[:valid_sie] = x[:valid_sie]
        result[-valid_sie:] = x[-valid_sie:]
        return result

    # 对预测曲线进行平滑处理
    def filter_lines(self, exact_curse_lines_multi_cls):
        output_curse_lines_multi_cls = []
        for exact_curse_lines in exact_curse_lines_multi_cls:
            lines = []
            for exact_curse_line in exact_curse_lines:
                points = np.array(exact_curse_line)
                poitns_cls = points[:, 4]

                if len(poitns_cls) > 5:
                    # 进行平滑处理
                    line_x = exact_curse_line[:, 0]
                    line_y = exact_curse_line[:, 1]

                    line_x = self.moving_average(line_x, window_size=5)
                    line_y = self.moving_average(line_y, window_size=5)

                    # 对误差大的点进行过滤
                    pixel_err = abs(line_x - exact_curse_line[:, 0]) + abs(line_y - exact_curse_line[:, 1])
                    mask = pixel_err < 20  # 608 * 960

                    exact_curse_line[:, 0][mask] = line_x[mask]
                    exact_curse_line[:, 1][mask] = line_y[mask]

                    if len(exact_curse_line) > 0:
                        lines.append(exact_curse_line)

            output_curse_lines_multi_cls.append(lines)
        return output_curse_lines_multi_cls

    def decode_curse_line(self, pred_confidence, pred_offset_x,
                          pred_offset_y, pred_emb, pred_emb_id,
                          pred_cls, orient_pred=None,
                          visible_pred=None, hanging_pred=None, covered_pred=None):

        pred_confidence = pred_confidence.clip(-20, 20)
        pred_confidence = sigmoid(pred_confidence)
        pred_cls = sigmoid(pred_cls)

        # import matplotlib.pyplot as plt
        # plt.subplot(2, 1, 1)
        # plt.imshow(pred_cls[1])
        # plt.subplot(2, 1, 2)
        # plt.imshow(pred_cls[3])
        # plt.show()
        # exit(1)

        self.pred_confidence = pred_confidence
        self.pred_emb = pred_emb
        self.pred_emb_id = pred_emb_id
        self.pred_cls = pred_cls

        pred_offset_x = pred_offset_x.clip(-20, 20)
        pred_offset_y = pred_offset_y.clip(-20, 20)

        pred_offset_x = sigmoid(pred_offset_x) * (self.grid_size - 1)
        pred_offset_y = sigmoid(pred_offset_y) * (self.grid_size - 1)

        # pred_offset_x = pred_offset_x * (self.grid_size - 1)
        # pred_offset_y = pred_offset_y * (self.grid_size - 1)

        pred_offset_x = pred_offset_x.round().astype(np.int).clip(0, self.grid_size - 1)
        pred_offset_y = pred_offset_y.round().astype(np.int).clip(0, self.grid_size - 1)

        _, h, w = pred_offset_x.shape
        pred_grid_x = np.arange(w).reshape(1, 1, w).repeat(h, axis=1) * self.grid_size
        pred_grid_y = np.arange(h).reshape(1, h, 1).repeat(w, axis=2) * self.grid_size
        pred_x = pred_grid_x + pred_offset_x
        pred_y = pred_grid_y + pred_offset_y

        if orient_pred is not None:
            orient_pred = sigmoid(orient_pred) * 2 - 1
            # 转成kitti的角度表示
            orient_pred = np.arctan2(orient_pred[0], orient_pred[1]) / np.pi * 180

        if visible_pred is not None:
            visible_pred = sigmoid(visible_pred)
            visible_pred = visible_pred[0]

        if hanging_pred is not None:
            hanging_pred = sigmoid(hanging_pred)
            hanging_pred = hanging_pred[0]

        if covered_pred is not None:
            covered_pred = sigmoid(covered_pred)
            covered_pred = covered_pred[0]

        min_y, max_y = self.line_map_range

        mask = np.zeros_like(pred_confidence, dtype=np.bool)
        mask[:, min_y:max_y, :] = pred_confidence[:, min_y:max_y, :] > self.confident_t

        exact_lines = []
        count = 0
        for _mask, _pred_emb, _pred_x, _pred_y, _pred_confidence, _pred_emb_id in zip(mask, pred_emb,
                                                                                      pred_x, pred_y,
                                                                                      pred_confidence, pred_emb_id,
                                                                                      ):
            _exact_lines = connect_line_points(_mask, _pred_emb, _pred_x,
                                               _pred_y, _pred_confidence, _pred_emb_id,
                                               pred_cls, orient_pred, visible_pred, hanging_pred, covered_pred)

            exact_lines.append(_exact_lines)

        return exact_lines


if __name__ == "__main__":
    print("Start")
    glass_land_boundary_line_2d_decode_numpy = GlasslandBoundaryLine2DDecodeNumpy(confident_t=0.2)
    print("End")