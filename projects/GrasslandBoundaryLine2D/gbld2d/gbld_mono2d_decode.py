import copy

import torch
import torch.nn as nn
from mmengine.model import BaseModule
import math
from mmdet3d.registry import MODELS
import numpy as np
import cv2
from scipy.spatial import distance
from skimage import morphology

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


def sort_points_by_min_dist(image_xs, image_ys):
    points = np.array([[xs, ys] for xs, ys in zip(image_xs, image_ys)])
    point_n = len(points)

    points_mask = np.array([True] * point_n)
    points_idx = np.arange(point_n)

    select_idx = len(points) - 1
    points_mask[select_idx] = False

    indices = []
    indices.append(select_idx)

    while np.sum(points_mask) > 0:
        other_points = points[points_mask]

        other_points_idx = points_idx[points_mask]

        select_point = points[select_idx]
        point_dists = np.sqrt((other_points[:, 0] - select_point[0]) ** 2 + \
                      (other_points[:, 1] - select_point[1]) ** 2)

        select_idx = np.argmin(point_dists)
        min_dist = point_dists[select_idx]
        if min_dist > 50:
            break

        select_idx = other_points_idx[select_idx]

        points_mask[select_idx] = False
        indices.append(select_idx)

    return indices

def sort_point_by_x_and_y_direction(image_xs, image_ys):
    # 首先按照x从小到大排序，判断线的主体方向，在x相同的位置，按照线主体方向进行排序
    indexs = image_xs.argsort()
    # image_xs_sort = image_xs[indexs]
    image_ys_sort = image_ys[indexs]

    # 判断线的方向主体方向
    y_direct = image_ys_sort[0] - image_ys_sort[-1]
    if y_direct > 0:
        # 按照 x 坐标进行从小到大排序，在 x 相同时按照 y 坐标从大到小进行排序
        sorted_indices = np.lexsort((-image_ys, image_xs))
    else:
        # 按照 x 坐标进行从小到大排序，在 x 相同时按照 y 坐标从小到大进行排序
        sorted_indices = np.lexsort((image_ys, image_xs))
    return sorted_indices

def sort_point_by_x_y(image_xs, image_ys):
    # 首先按照x从小到大排序，在x相同的位置，按照与上一个x对应的y进行排序
    indexs = image_xs.argsort()
    image_xs_sort = image_xs[indexs]
    image_ys_sort = image_ys[indexs]

    # 将第一个点加入
    x_same = image_xs_sort[0]  # 判断是否为同一个x
    y_pre = image_ys_sort[0]  # 上一个不是同一个相同x的点的y

    indexs_with_same_x = [indexs[0]]  # 记录相同的x的index
    ys_with_same_x = [image_ys_sort[0]]  # 记录具有相同x的ys

    new_indexs = []
    for i, (idx, x_s, y_s) in enumerate(zip(indexs[1:], image_xs_sort[1:], image_ys_sort[1:])):
        if x_s == x_same:
            indexs_with_same_x.append(idx)
            ys_with_same_x.append(y_s)
        else:
            # 如果当前xs与前面的不一样，将前面的进行截断统计分析
            # 对y进行排序， 需要判断y是从大到小还是从小到大排序, 需要跟上一个x对应的y来判断，距离近的排在前面
            # 首先按照从小到大排序
            index_y_with_same_x = np.array(ys_with_same_x).argsort()

            # 判断是否需要倒转过来排序
            if len(index_y_with_same_x) > 1:
                if abs(index_y_with_same_x[-1] - y_pre) < abs(index_y_with_same_x[0] - y_pre):
                    index_y_with_same_x = index_y_with_same_x[::-1]

            new_indexs = new_indexs + np.array(indexs_with_same_x)[index_y_with_same_x].tolist()

            # 为下次的判断作准备
            y_pre = ys_with_same_x[index_y_with_same_x[-1]]
            x_same = x_s
            indexs_with_same_x = [idx]
            ys_with_same_x = [y_s]

        if i == len(image_xs) - 2:  # 判断是否为最后一个点
            index_y_with_same_x = np.array(ys_with_same_x).argsort()
            if len(index_y_with_same_x) > 1:
                if abs(index_y_with_same_x[-1] - y_pre) < abs(index_y_with_same_x[0] - y_pre):
                    index_y_with_same_x = index_y_with_same_x[::-1]
            new_indexs = new_indexs + np.array(indexs_with_same_x)[index_y_with_same_x].tolist()
    return new_indexs

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
    # 首先按照x从小到大排序，判断线的主体方向，在x相同的位置，按照线主体方向进行排序
    # indices = sort_point_by_x_and_y_direction(image_xs, image_ys)
    # indices = sort_point_by_x_y(image_xs, image_ys)

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
        current_line = long_lines[0]
        current_endpoints = [current_line[0], current_line[-1]]
        other_lines = long_lines[1:]
        other_endpoints = []
        for o_line in other_lines:
            # num_len = len(other_lines)
            other_endpoints.append(o_line[0])
            other_endpoints.append(o_line[-1])
        point_ids = [None, None]
        min_dist = 10000
        for i, c_end in enumerate(current_endpoints):
            for j, o_end in enumerate(other_endpoints):
                distance = compute_point_distance(c_end, o_end)
                if distance < min_dist:
                    point_ids[0] = i
                    point_ids[1] = j
                    min_dist = distance
        if min_dist < endpoint_distance:
            adjacent_line = other_lines[point_ids[1] // 2]
            other_lines.remove(adjacent_line)
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
        else:
            final_lines.append(current_line)
            long_lines = other_lines
    final_lines.append(long_lines[0])
    final_lines = sorted(final_lines, key=lambda l: len(l), reverse=True)
    return final_lines


def serialize_single_line(single_line, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map,
                          pred_orient_map=None, pred_visible_map=None, pred_hanging_map=None, pred_covered_map=None):
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

        selected_points = extend_endpoints(selected_points, single_line)

        piecewise_line = arrange_points_to_line(
            selected_points, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map, pred_orient_map,
            pred_visible_map, pred_hanging_map, pred_covered_map
        )
        # piecewise_line = self.fit_points_to_line(selected_points, x_map, y_map, confidence_map)  # Curve Fitting
        piecewise_lines.append(piecewise_line)
        existing_points = alternative_points


    if len(piecewise_lines) == 0:
        return []
    elif len(piecewise_lines) == 1:
        exact_lines = piecewise_lines[0]
    else:
        exact_lines = connect_piecewise_lines(piecewise_lines)[0]
        # exact_lines = connect_piecewise_lines(piecewise_lines, endpoint_distance=40)[0]
    if exact_lines[0][1] < exact_lines[-1][1]:
        exact_lines.reverse()
    return exact_lines


def split_piecewise_lines(piecewise_lines, split_dist=12):
    split_piecewise_lines = []
    for i, raw_line in enumerate(piecewise_lines):
        pre_point = raw_line[0]
        start_idx = 0
        end_idx = 0

        for j, cur_point in enumerate(raw_line[1:]):
            point_dist = compute_point_distance(pre_point, cur_point)
            if point_dist > split_dist:
                split_piecewise_line = raw_line[start_idx:end_idx + 1]
                if len(split_piecewise_line) > 1:
                    split_piecewise_lines.append(split_piecewise_line)

                start_idx = j + 1  # 需要加上0-index的长度

            pre_point = cur_point
            end_idx = j + 1

            if j == len(raw_line) - 2:
                split_piecewise_line = raw_line[start_idx:end_idx + 1]
                if len(split_piecewise_line) > 1:
                    split_piecewise_lines.append(split_piecewise_line)
    return split_piecewise_lines

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
        # piecewise_lines.append(piecewise_line)
        if len(piecewise_line) > 1:
            piecewise_lines.append(piecewise_line)

        existing_points = alternative_points

    # 在这里判断是否继续对line进行分段，如果两个点的距离太大就会断开，防止两个相邻点之间的距离过大
    piecewise_lines = split_piecewise_lines(piecewise_lines, split_dist=12)

    if len(piecewise_lines) == 0:
        return []
    elif len(piecewise_lines) == 1:
        # exact_lines = piecewise_lines[0]
        all_exact_lines = piecewise_lines
    else:

        # exact_lines = connect_piecewise_lines(piecewise_lines, endpoint_distance=40)[0]
        # all_exact_lines = connect_piecewise_lines(piecewise_lines, endpoint_distance=30)
        all_exact_lines = connect_piecewise_lines(piecewise_lines, endpoint_distance=16)
        # all_exact_lines = connect_piecewise_lines(piecewise_lines, endpoint_distance=64)

    for all_exact_line in all_exact_lines:
        if all_exact_line[0][1] < all_exact_line[-1][1]:
            all_exact_line.reverse()

    # if exact_lines[0][1] < exact_lines[-1][1]:
    #     exact_lines.reverse()
    return all_exact_lines

def split_line_points_by_cls(raw_lines, pred_cls_map):
    pred_cls_map = np.argmax(pred_cls_map, axis=0)
    raw_lines_cls = []
    for raw_line in raw_lines:
        raw_line = np.array(raw_line)
        xs, ys = raw_line[:, 0], raw_line[:, 1]
        clses = pred_cls_map[ys, xs]
        cls_ids = np.unique(clses)
        for cls_id in cls_ids:
            cls_mask = clses == cls_id
            raw_line_cls = raw_line[cls_mask]
            raw_lines_cls.append(raw_line_cls.tolist())
    return raw_lines_cls


def get_line_key_point(line, order, fixed):
    index = []
    if order == 0:
        for point in line:
            if int(point[0]) == int(fixed):
                index.append(int(point[1]))
    else:
        for point in line:
            if int(point[1]) == int(fixed):
                index.append(int(point[0]))

    index = np.sort(index)

    start = False
    last_ind = -1
    start_ind = -1
    keypoint = []
    if len(index) == 1:
        if order == 0:
            keypoint.append([fixed, index[0]])
        else:
            keypoint.append([index[0], fixed])

    elif len(index) > 1:
        for i in range(len(index)):
            if start == False:
                start = True
                start_ind = index[i]
            if i == 0:
                start = True
                last_ind = index[i]
                start_ind = index[i]
                continue
            # if abs(index[i] - last_ind) > 1 or i == len(index) - 1:
            #     end_ind = last_ind
            #     start = False
            #     if order == 0:
            #         keypoint.append([fixed, int((start_ind + end_ind) / 2)])
            #     else:
            #         keypoint.append([int((start_ind + end_ind) / 2), fixed])
            #     start_ind = index[i]

            if abs(index[i] - last_ind) > 1:
                end_ind = last_ind
                start = False
                if order == 0:
                    keypoint.append([fixed, int((start_ind + end_ind) / 2)])
                else:
                    keypoint.append([int((start_ind + end_ind) / 2), fixed])
                start_ind = index[i]

                if i == len(index) - 1:
                    if order == 0:
                        keypoint.append([fixed, int(index[i])])
                    else:
                        keypoint.append([int(index[i]), fixed])

            elif i == len(index) - 1:
                end_ind = index[i]
                start = False
                if order == 0:
                    keypoint.append([fixed, int((start_ind + end_ind) / 2)])
                else:
                    keypoint.append([int((start_ind + end_ind) / 2), fixed])
                start_ind = index[i]

            last_ind = index[i]

    return keypoint


def get_slim_points(line, start_x, end_x, start_y, end_y, step, order):
    slim_points = []
    for x_index in range(start_x, end_x+1, step):
        keypoint = get_line_key_point(line, order, x_index)
        slim_points.extend(keypoint)
    return slim_points


def zhang_suen_thining_condiction2(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    f1 = 0
    if (x3 - x2) == 1:
        f1 += 1
    if (x4 - x3) == 1:
        f1 += 1
    if (x5 - x4) == 1:
        f1 += 1
    if (x6 - x5) == 1:
        f1 += 1
    if (x7 - x6) == 1:
        f1 += 1
    if (x8 - x7) == 1:
        f1 += 1
    if (x9 - x8) == 1:
        f1 += 1
    if (x2 - x9) == 1:
        f1 += 1
    return f1


def get_point_neighbor(line, point, b_inv=True):
    # b_inv为true的时候,1代表neighbor不存在
    # 这里的计算对应的图像坐标系 y-1 为x2, 图像的实现方式
    # x9 x2 x3
    # x8 x1 x4
    # x7 x6 x5

    # 但是计算出来的线实际为y+1为x2, 修改y的offset来实现
    # x1, x2, x3, x4, x5, x6, x7, x8, x9
    x_offset = [0,  0,  1,  1,  1,  0,  -1, -1, -1]
    # y_offset = [0, -1, -1,  0,  1,  1,   1,  0, -1]    # y-1 为x2
    y_offset = [0,  1,  1, 0, -1, -1,  -1,  0, 1]    # y+1 为x2
    point_neighbors = []
    x_p, y_p = point[0], point[1]

    line_points = line
    for x_o, y_o in zip(x_offset, y_offset):
        x_n, y_n = x_p + x_o, y_p + y_o
        if b_inv:
            not_has_neighbot = 1 if [x_n, y_n] not in line_points else 0
            # if not_has_neighbot != 0:
            #     print("fffff")
            #     exit(1)
            point_neighbors.append(not_has_neighbot)
        else:
            has_neighbot = 1 if [x_n, y_n] in line_points else 0
            point_neighbors.append(has_neighbot)
    return point_neighbors

def zhang_suen_thining_points(line):
    # line = line.tolist()
    out = line.tolist()
    while True:
        s1 = []
        s2 = []
        # x9 x2 x3
        # x8 x1 x4
        # x7 x6 x5
        for point in out:
            # condition 2
            x1, x2, x3, x4, x5, x6, x7, x8, x9 = get_point_neighbor(out, point, b_inv=True)
            f1 = zhang_suen_thining_condiction2(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if f1 != 1:
                continue

            # condition 3
            f2 = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9)
            if f2 < 2 or f2 > 6:
                continue

            # condition 4
            # x2 x4 x6
            if (x2 + x4 + x6) < 1:
                continue

            # x4 x6 x8
            if (x4 + x6 + x8) < 1:
                continue
            s1.append(point)

        # 将s1中的点去除
        out = [point for point in out if point not in s1]
        for point in out:
            x1, x2, x3, x4, x5, x6, x7, x8, x9 = get_point_neighbor(out, point, b_inv=True)
            f1 = zhang_suen_thining_condiction2(x1, x2, x3, x4, x5, x6, x7, x8, x9)

            if f1 != 1:
                continue

            f2 = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9)
            if f2 < 2 or f2 > 6:
                continue

            if (x2 + x4 + x6) < 1:
                continue

            if (x4 + x6 + x8) < 1:
                continue
            s2.append(point)

        # 将s2中的点去除
        out = [point for point in out if point not in s2]

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break
    return out


def get_slim_lines(lines):
    slim_lines = []
    for line in lines:
        line = np.array(line)
        xs, ys = line[:, 0], line[:, 1],
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # 采用zhang_suen骨架算法
        slim_points = zhang_suen_thining_points(line)

        # pixel_step = 1
        # 判断条件采用ch单边slim处理
        # x_len = xmax - xmin
        # y_len = ymax - ymin
        # ratio_len = max(x_len, y_len) / (min(x_len, y_len) + 1e-8)
        # if ratio_len > 3:
        #     if x_len > y_len:
        #         order = 0
        #         slim_points = get_slim_points(slim_points, xmin, xmax, ymin, ymax, pixel_step, order)
        #     else:
        #         order = 1
        #         slim_points = get_slim_points(slim_points, ymin, ymax, xmin, xmax, pixel_step, order)
        # else:
        #
        #     # 直接采用ch双边slim处理
        #     order = 0
        #     slim_points_x = get_slim_points(slim_points, xmin, xmax, ymin, ymax, pixel_step, order)
        #     order = 1
        #     slim_points_y = get_slim_points(slim_points, ymin, ymax, xmin, xmax, pixel_step, order)
        #     slim_points = slim_points_x + slim_points_y

        # if len(slim_points) > 1:
        # 去除孤立的点
        # slim_points = remove_isolated_points(slim_points)

        slim_lines.append(slim_points)
    return slim_lines


def cluster_line_points_high_dim(xs, ys, emb_map, pull_margin=1.5):
    emb_map = np.transpose(emb_map, (1, 2, 0))
    ebs = emb_map[ys, xs]

    lines = []
    embedding_means = []
    point_numbers = []
    for x, y, eb in zip(xs, ys, ebs):
        id = None
        min_dist = 10000
        for i, eb_mean in enumerate(embedding_means):
            # distance = sum(abs(eb - eb_mean))
            distance = np.linalg.norm(eb - eb_mean, ord=2)    # 求两个向量的l2范数
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

def connect_line_points(mask_map, embedding_map, x_map, y_map,
                        confidence_map, pred_emb_id, pred_cls_map,
                        pred_orient_map=None, pred_visible_map=None,
                        pred_hanging_map=None, pred_covered_map=None,
                        discriminative_map=None,
                        line_maximum=10):
    # 采用骨架细化的方式
    # mask_map = morphology.skeletonize(mask_map)
    # mask_map = mask_map.astype(np.uint8)

    # 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素
    mask_map = mask_map.astype(np.uint8)
    mask_map = cv2.morphologyEx(mask_map, cv2.MORPH_CLOSE, kernel, iterations=1)

    ys, xs = np.nonzero(mask_map)
    ys, xs = np.flipud(ys), np.flipud(xs)   # 优先将y大的点排在前面
    ebs = embedding_map[ys, xs]

    raw_lines = cluster_line_points(xs, ys, ebs)
    # raw_lines = cluster_line_points_high_dim(xs, ys, discriminative_map)

    # 聚类出来的lines是没有顺序的点击, 根据类别将点集划分为更细的点击
    raw_lines = split_line_points_by_cls(raw_lines, pred_cls_map)

    # 将线进行细化
    raw_lines = get_slim_lines(raw_lines)

    raw_lines = remove_short_lines(raw_lines)
    raw_lines = remove_far_lines(raw_lines)

    exact_lines = []
    for each_line in raw_lines:
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
                    # if len(single_line) > 45:
                    if len(single_line) > 20:   # 2023-11-15
                        exact_lines.append(single_line)

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


@MODELS.register_module()
class GlasslandBoundaryLine2DDecode(BaseModule):

    def __init__(self,
                 confident_t,
                 grid_size=4,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super().__init__(init_cfg)
        self.confident_t = confident_t   # 预测seg-heatmap的阈值
        self.grid_size = grid_size
        self.line_map_range = [0, -1]    # 不进行过滤

        self.max_pooling_col = nn.MaxPool2d((3, 1), stride=(1, 1), padding=[1, 0])
        self.max_pooling_row = nn.MaxPool2d((1, 3), stride=(1, 1), padding=[0, 1])
        self.max_pooling_dilate = nn.MaxPool2d([3, 3], stride=1, padding=[1, 1])  # 去锯齿

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

                        if reverse:
                            revese_num[0] = revese_num[0] + 1
                        else:
                            revese_num[1] = revese_num[1] + 1

                    pre_point = cur_point
                # 判断是否需要调转顺序
                if revese_num[0] > revese_num[1]:
                    exact_curse_line = exact_curse_line[::-1, :]
                lines.append(exact_curse_line)
            output_curse_lines_with_orient.append(lines)
        return output_curse_lines_with_orient

    def heatmap_nms(self, seg_pred):
        seg_max_pooling_col = self.max_pooling_col(seg_pred)
        seg_max_pooling_row = self.max_pooling_row(seg_pred)

        mask_col = seg_pred == seg_max_pooling_col
        mask_row = seg_pred == seg_max_pooling_row
        mask = torch.bitwise_or(mask_col, mask_row)
        seg_pred[~mask] = -1e6
        # seg_pred[~mask_col] = -1e6
        # seg_pred[~mask_row] = -1e6
        # seg_pred = self.max_pooling_dilate(seg_pred)

        return seg_pred

    def forward(self, seg_pred, offset_pred, seg_emb_pred, connect_emb_pred, cls_pred, orient_pred=None,
                visible_pred=None, hanging_pred=None, covered_pred=None, discriminative_pred=None):
        # 对pred_confidene 进行max-pooling
        # seg_pred = self.heatmap_nms(seg_pred)

        seg_pred = seg_pred.cpu().detach().numpy()
        offset_pred = offset_pred.cpu().detach().numpy()
        seg_emb_pred = seg_emb_pred.cpu().detach().numpy()
        connect_emb_pred = connect_emb_pred.cpu().detach().numpy()
        cls_pred = cls_pred.cpu().detach().numpy()


        if orient_pred is not None:
            orient_pred = orient_pred.cpu().detach().numpy()

        if visible_pred is not None:
            visible_pred = visible_pred.cpu().detach().numpy()

        if hanging_pred is not None:
            hanging_pred = hanging_pred.cpu().detach().numpy()

        if covered_pred is not None:
            covered_pred = covered_pred.cpu().detach().numpy()

        if discriminative_pred is not None:
            discriminative_pred = discriminative_pred.cpu().detach().numpy()

        curse_lines_with_cls = self.decode_curse_line(seg_pred, offset_pred[0:1],
                                                  offset_pred[1:2], seg_emb_pred, connect_emb_pred,
                                                      cls_pred, orient_pred, visible_pred,
                                                      hanging_pred, covered_pred,
                                                      discriminative_pred)

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

                # 大于10个点的才会输出
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

                    if len(exact_curse_line) > 1:
                        lines.append(exact_curse_line)

                # else:
                #     lines.append(exact_curse_line)

            output_curse_lines_multi_cls.append(lines)
        return output_curse_lines_multi_cls

    def decode_curse_line(self, pred_confidence, pred_offset_x,
                          pred_offset_y, pred_emb, pred_emb_id,
                          pred_cls, orient_pred=None,
                          visible_pred=None, hanging_pred=None, covered_pred=None,
                          discriminative_pred=None):

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

        pred_offset_x = pred_offset_x.round().astype(np.int32).clip(0, self.grid_size - 1)
        pred_offset_y = pred_offset_y.round().astype(np.int32).clip(0, self.grid_size - 1)

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

        mask = np.zeros_like(pred_confidence, dtype=np.bool8)
        mask[:, min_y:max_y, :] = pred_confidence[:, min_y:max_y, :] > self.confident_t

        # import matplotlib.pyplot as plt
        # plt.imshow(mask[0])
        # plt.show()
        # exit(1)

        exact_lines = []
        count = 0
        for _mask, _pred_emb, _pred_x, _pred_y, _pred_confidence, _pred_emb_id in zip(mask, pred_emb,
                                                                                      pred_x, pred_y,
                                                                                      pred_confidence, pred_emb_id,
                                                                                      ):
            _exact_lines = connect_line_points(_mask, _pred_emb, _pred_x,
                                               _pred_y, _pred_confidence, _pred_emb_id,
                                               pred_cls, orient_pred, visible_pred, hanging_pred, covered_pred,
                                               discriminative_map=discriminative_pred,
                                               line_maximum=15)

            exact_lines.append(_exact_lines)

        return exact_lines