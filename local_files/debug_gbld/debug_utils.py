import numpy as np
import math
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_inverse(p):
    return np.log(p / (1 - p))

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


def arrange_points_to_line(selected_points, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map, map_size=(1920, 1080)):
    selected_points = np.array(selected_points)
    xs, ys = selected_points[:, 0], selected_points[:, 1]
    image_xs = x_map[ys, xs]
    image_ys = y_map[ys, xs]
    confidences = confidence_map[ys, xs]

    pred_cls_map = np.argmax(pred_cls_map, axis=0)
    emb_ids = pred_emb_id_map[ys, xs]
    clses = pred_cls_map[ys, xs]

    indices = image_xs.argsort()
    image_xs = image_xs[indices]
    image_ys = image_ys[indices]
    confidences = confidences[indices]
    h, w = map_size
    line = []
    for x, y, conf, emb_id, cls in zip(image_xs, image_ys, confidences, emb_ids, clses):
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
        current_line = long_lines[0]
        current_endpoints = [current_line[0], current_line[-1]]
        other_lines = long_lines[1:]
        other_endpoints = []
        for o_line in other_lines:
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


def serialize_single_line(single_line, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map):
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
            selected_points, x_map, y_map, confidence_map, pred_emb_id_map, pred_cls_map
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
    if exact_lines[0][1] < exact_lines[-1][1]:
        exact_lines.reverse()
    return exact_lines


def connect_line_points(mask_map, embedding_map, x_map, y_map,
                        confidence_map, pred_emb_id, pred_cls_map, line_maximum=10):
    ys, xs = np.nonzero(mask_map)
    ys, xs = np.flipud(ys), np.flipud(xs)   # 优先将y大的点排在前面
    ebs = embedding_map[ys, xs]
    raw_lines = cluster_line_points(xs, ys, ebs)

    raw_lines = remove_short_lines(raw_lines)
    raw_lines = remove_far_lines(raw_lines)

    exact_lines = []
    for each_line in raw_lines:
        single_line = serialize_single_line(each_line, x_map, y_map,
                                            confidence_map, pred_emb_id, pred_cls_map)
        if len(single_line) > 0:
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


def decode_curse_line(pred_confidence, pred_offset_x,
                      pred_offset_y, pred_emb, pred_emb_id,
                      pred_cls, grid_size=1):
    # pred_confidence = pred_confidence.clip(-20, 20)
    # pred_confidence = sigmoid(pred_confidence)
    # pred_cls = sigmoid(pred_cls)

    # self.pred_confidence = pred_confidence
    # self.pred_emb = pred_emb
    # self.pred_emb_id = pred_emb_id
    # self.pred_cls = pred_cls

    # pred_offset_x = pred_offset_x.clip(-20, 20)
    # pred_offset_y = pred_offset_y.clip(-20, 20)

    # pred_offset_x = sigmoid(pred_offset_x) * (self.grid_size - 1)
    # pred_offset_y = sigmoid(pred_offset_y) * (self.grid_size - 1)

    pred_offset_x = pred_offset_x * (grid_size - 1)
    pred_offset_y = pred_offset_y * (grid_size - 1)

    pred_offset_x = pred_offset_x.round().astype(np.int).clip(0, grid_size - 1)
    pred_offset_y = pred_offset_y.round().astype(np.int).clip(0, grid_size - 1)

    _, h, w = pred_offset_x.shape
    pred_grid_x = np.arange(w).reshape(1, 1, w).repeat(h, axis=1) * grid_size
    pred_grid_y = np.arange(h).reshape(1, h, 1).repeat(w, axis=2) * grid_size
    pred_x = pred_grid_x + pred_offset_x
    pred_y = pred_grid_y + pred_offset_y

    # min_y, max_y = self.line_map_range
    min_y, max_y = 0, -1
    confident_t = 0.2

    mask = np.zeros_like(pred_confidence, dtype=np.bool)
    # mask[:, min_y:max_y, :] = pred_confidence[:, min_y:max_y, :] > self.confident_t
    # mask[:, min_y:max_y, :] = pred_confidence[:, min_y:max_y, :] > confident_t
    mask = pred_confidence > confident_t

    exact_lines = []
    count = 0
    for _mask, _pred_emb, _pred_x, _pred_y, _pred_confidence, _pred_emb_id in zip(mask, pred_emb,
                                                                                  pred_x, pred_y,
                                                                                  pred_confidence, pred_emb_id):
        _exact_lines = connect_line_points(_mask, _pred_emb, _pred_x,
                                           _pred_y, _pred_confidence,
                                           _pred_emb_id, pred_cls)

        exact_lines.append(_exact_lines)

    return exact_lines


def get_line_cls(exact_curse_lines_multi_cls):
    output_curse_lines_multi_cls = []
    for exact_curse_lines in exact_curse_lines_multi_cls:
        lines = []
        for exact_curse_line in exact_curse_lines:
            points = np.array(exact_curse_line)
            poitns_cls = points[:, 4]
            cls = np.argmax(np.bincount(poitns_cls.astype(np.int32)))
            points[:, 4] = cls

            lines.append(np.array(list(points)))

        output_curse_lines_multi_cls.append(lines)
    return output_curse_lines_multi_cls


def decode_gt_lines(seg_pred, offset_pred, seg_emb_pred, connect_emb_pred, cls_pred, grid_size=1):
    curse_lines_with_cls = decode_curse_line(seg_pred, offset_pred[0:1],
                                                offset_pred[1:2], seg_emb_pred,
                                                connect_emb_pred, cls_pred, grid_size=grid_size)

    curse_lines_with_cls = get_line_cls(curse_lines_with_cls)

    return curse_lines_with_cls


def cal_points_orient(pre_point, cur_point):
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
        angle = abs(math.atan(y/x)) /math.pi * 180
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


def draw_orient(img, point, orient, arrow_len=10, color=(0, 255, 0)):
    # orient = 270

    c_x, c_y = point[0], point[1]
    # if orient_score > 0.3:
    orient = 90 + orient
    if orient > 360:
        orient = orient - 360

    # w = x2 - x1
    # h = y2 - y1
    # c_x = int(x1 + w * 1 / 2)
    # c_y = int(y1 + h * 1 / 2)
    # c_r = int(min(w, h) * 0.5)

    x1 = c_x + arrow_len * math.cos(orient * np.pi / 180.0)
    y1 = c_y - arrow_len * math.sin(orient * np.pi / 180.0)
    # cv2.line(img, (int(x1), int(y1)), (int(c_x), int(c_y)), (0, 0, 255), thickness=1)
    cv2.arrowedLine(img, (int(c_x), int(c_y)), (int(x1), int(y1)), color, 2, 8, 0, 1.0)
    return img


def filter_near_same_points(line_points):
    mask = [True] * len(line_points)
    pre_point = line_points[0]
    for i, cur_point in enumerate(line_points[1:]):
        if np.all(cur_point == pre_point):
            mask[i + 1] = False
        else:
            pre_point = cur_point

    line_points = line_points[mask]
    return line_points


def parse_ann_info(info: dict) -> dict:
    metainfo = {}
    metainfo["classes"] = [
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
    ann_path = info["ann_path"]
    with open(ann_path, 'r') as f:
        anns = json.load(f)

    ann_info = dict()

    remap_anns = []
    for i, shape in enumerate(anns['shapes']):
        line_id = int(shape['id'])
        label = shape['label']

        if label not in metainfo["classes"]:
            continue

        category_id = metainfo["classes"].index(label)

        points = shape['points']
        points_remap = []

        # 读取数据阶段进行缩放, 在图像进行方法的时候才需要
        scale_x = 1.0
        scale_y = 1.0
        pad_left = 0.0
        pad_top = 0
        for point in points:
            x = point[0] * scale_x + pad_left
            y = point[1] * scale_y + pad_top
            points_remap.append([x, y])

        points_remap = np.array(points_remap)
        # remap_annos.append({'label': label, 'points': points_remap, 'index': index, 'same_line_id': index})
        # 新增label对应的类id, category_id
        remap_anns.append({'label': label, 'points': points_remap, 'index': i, 'line_id': line_id, "category_id": category_id})

    ann_info["gt_lines"] = remap_anns
    return ann_info


def parse_ann_infov2(info: dict, classes=None) -> dict:
    metainfo = {}
    metainfo["classes"] = [
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

    if classes is not None:
        metainfo["classes"] = classes

    ann_path = info["ann_path"]
    with open(ann_path, 'r') as f:
        anns = json.load(f)

    ann_info = dict()

    remap_anns = []
    for i, shape in enumerate(anns['shapes']):
        line_id = int(shape['id'])
        label = shape['label']

        if label not in metainfo["classes"]:
            continue

        category_id = metainfo["classes"].index(label)

        points = shape['points']
        points_remap = []

        # 读取数据阶段进行缩放, 在图像进行方法的时候才需要
        scale_x = 1.0
        scale_y = 1.0
        pad_left = 0.0
        pad_top = 0
        for point in points:
            x = point[0] * scale_x + pad_left
            y = point[1] * scale_y + pad_top
            points_remap.append([x, y])

        points_remap = np.array(points_remap)
        # remap_annos.append({'label': label, 'points': points_remap, 'index': index, 'same_line_id': index})
        # 新增label对应的类id, category_id

        # 新增点的类别、可见属性、悬空属性和被草遮挡属性等
        points_type = shape["points_type"]
        points_visible = shape["points_visible"]
        points_hanging = shape["points_hanging"]
        points_covered = shape["points_covered"]

        # 将字符类型转为与category_id相同
        for point_type in points_type:
            point_type[0] = metainfo["classes"].index(point_type[0])

        # point_vis为true的情况下为可见
        for point_visible in points_visible:
            point_visible[0] = 1 if point_visible[0] == "true" else 0

        # point_hang为true的情况为悬空
        for point_hanging in points_hanging:
            point_hanging[0] = 1 if point_hanging[0] == "true" else 0

        # point_covered为true的情况为被草遮挡
        for point_covered in points_covered:
            point_covered[0] = 1 if point_covered[0] == "true" else 0

        points_type = np.array(shape["points_type"])
        points_visible = np.array(shape["points_visible"])
        points_hanging = np.array(shape["points_hanging"])
        points_covered = np.array(shape["points_covered"])

        remap_anns.append({'label': label, 'points': points_remap, 'index': i,
                           'line_id': line_id, "category_id": category_id,
                           'points_type': points_type, 'points_visible': points_visible,
                           'points_hanging': points_hanging, 'points_covered': points_covered,
                           })

    ann_info["gt_lines"] = remap_anns
    return ann_info

def cluster_line_points_high_dim(xs, ys, embeddings, pull_margin=1.5):
    lines = []
    embedding_means = []
    point_numbers = []
    for x, y, eb in zip(xs, ys, embeddings):
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


def discriminative_cluster_postprocess(seg_map, cluster_map):
    plt.imshow(seg_map)
    plt.show()
    embed_dim, img_h, img_w = cluster_map.shape
    cluster_map = np.transpose(cluster_map, (1, 2, 0))

    # mean-shift
    # show_cluster_map_mean_shift = np.zeros((img_h, img_w), dtype=np.uint8)
    # cluster_list = cluster_map[seg_map]
    # # mean_shift = MeanShift(bandwidth=1.5, bin_seeding=True, n_jobs=-1)
    # mean_shift = MeanShift(bandwidth=2.0, bin_seeding=True, n_jobs=-1)
    # mean_shift.fit(cluster_list)
    #
    # labels = mean_shift.labels_
    # print("mean_shift:", np.unique(labels))
    # show_cluster_map_mean_shift[seg_map] = labels + 1
    #
    # show_cluster_map_mean_shift2 = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    #
    # for i, label in enumerate(np.unique(labels) + 1):
    #     label_mask = show_cluster_map_mean_shift==(label)
    #     show_cluster_map_mean_shift2[label_mask, :] = color_list[i]
    #
    # plt.imshow(show_cluster_map_mean_shift2)
    # plt.title("show_cluster_map_mean_shift2")
    # plt.show()

    # local shift
    show_cluster_map_local = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    ys, xs = np.nonzero(seg_map)
    ys, xs = np.flipud(ys), np.flipud(xs)   # 优先将y大的点排在前面
    ebs = cluster_map[ys, xs]
    lines = cluster_line_points_high_dim(xs, ys, ebs)

    for i, line in enumerate(lines):
        line = np.array(line)
        x, y = line[:, 0], line[:, 1]
        show_cluster_map_local[y, x, :] = color_list[i % len(color_list)]

    plt.imshow(show_cluster_map_local)
    plt.title("show_cluster_map_local")
    plt.show()
    print(len(lines))
    exit(1)



def draw_pred_result_numpy(img_show, single_stage_result):
    # stages_result = results.pred_instances.stages_result[0]
    # meta_info = results.metainfo
    # batch_input_shape = meta_info["batch_input_shape"]
    #
    # pred_line_map = np.zeros(batch_input_shape, dtype=np.uint8)
    # single_stage_result = stages_result[0]
    thickness = 1
    color = (0, 0, 255)

    for curve_line in single_stage_result:
        curve_line = np.array(curve_line)

        pre_point = curve_line[0]
        line_cls = pre_point[4]
        # color = color_list[int(line_cls)]
        # x1, y1 = int(pre_point[0]), int(pre_point[1])

        point_num = len(curve_line)
        for i, cur_point in enumerate(curve_line[1:]):
            x1, y1 = int(pre_point[0]), int(pre_point[1])
            x2, y2 = int(cur_point[0]), int(cur_point[1])

            # cv2.line(pred_line_map, (x1, y1), (x2, y2), (1), thickness, 8)
            cv2.line(img_show, (x1, y1), (x2, y2), color, thickness, 8)

            line_orient = cal_points_orient(pre_point, cur_point)

            if i % 40 == 0:
                orient = pre_point[5]
                if orient != -1:
                    reverse = False  # 代表反向是否反了
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

                    # img_show = draw_orient(img_show, pre_point, orient, arrow_len=30, color=color)

            if i == point_num // 2:
                line_orient = line_orient + 90
                if line_orient > 360:
                    line_orient = line_orient - 360
                img_show = draw_orient(img_show, pre_point, line_orient, arrow_len=50, color=color)

            pre_point = cur_point
    # plt.imshow(img_show[:, :, ::-1])
    # # plt.imshow(pred_line_map)
    # plt.show()
    # exit(1)
    return img_show