import cv2
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d


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
        # 对曲线进行拟合
        piecewise_line = fit_piece_curve_line(piecewise_line)

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


def fit_piece_curve_line(piece_curve_line):
    # y3 = poly1d(polyfit(x, y, 3))  # 后面的数据代表次数项
    # p = plt.plot(x, y3(x))
    piece_curve_line = np.array(piece_curve_line)
    x = piece_curve_line[:, 0]
    y = piece_curve_line[:, 1]
    fit_func = poly1d(polyfit(x, y, 5))
    y_fit = fit_func(x)
    piece_curve_line[:, 1] = y_fit

    piece_curve_line = piece_curve_line.tolist()

    return piece_curve_line



def get_dense_points(points, img_h, img_w):
    gt_confidence = np.zeros((img_h, img_w), dtype=np.float32)
    pre_point = points[0]
    for cur_point in points[1:]:
        x1, y1 = round(pre_point[0]), round(pre_point[1])
        x2, y2 = round(cur_point[0]), round(cur_point[1])
        cv2.line(gt_confidence, (x1, y1), (x2, y2), (1,))
        pre_point = cur_point

    ys, xs = np.nonzero(gt_confidence)
    ys, xs = np.flipud(ys), np.flipud(xs)  # 优先将y大的点排在前面
    line = []
    for x, y in zip(xs, ys):
        line.append((x, y))
    return line


def order_lines(dense_points, start_point):
    # 对比dense_points的起点和终点距离起始位置的距离, 将距离近的作为起点
    dense_start_point = dense_points[0]
    dense_end_point = dense_points[-1]
    dense_start_dist = compute_point_distance(dense_start_point, start_point)
    dense_end_dist = compute_point_distance(dense_end_point, start_point)

    if dense_end_dist < dense_start_dist:
        dense_points.reverse()

    return dense_points

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


def draw_orient(img, point, orient, arrow_len=10):
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
    cv2.arrowedLine(img, (int(c_x), int(c_y)), (int(x1), int(y1)), (0, 255, 0), 2, 8, 0, 1.0)
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


def dense_line_points(img, points):
    img_h, img_w, _ = img.shape
    dense_points = get_dense_points(points, img_h, img_w)
    single_line = serialize_single_line(dense_points)
    single_line = order_lines(single_line, points[0])

    single_line = np.array(single_line)[:, :2]
    single_line = filter_near_same_points(single_line)

    # debug show
    if 0:
        img_show = copy.deepcopy(img)
        pre_point = single_line[0]
        for i, cur_point in enumerate(single_line[1:]):
            x1, y1 = round(pre_point[0]), round(pre_point[1])
            x2, y2 = round(cur_point[0]), round(cur_point[1])
            # cv2.line(line_map, (x1, y1), (x2, y2), (index,))
            # cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,))

            # cls_value = self.class_dic[label] + 1
            # cv2.line(img_show, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # 求每个点的方向
            if i % 50 == 0:
                orient = cal_points_orient(pre_point, cur_point)
                # 转个90度,指向草地
                # orient = orient + 90
                # if orient > 360:
                #     orient = orient - 360

                img_show = draw_orient(img_show, pre_point, orient, arrow_len=15)

            pre_point = cur_point

        plt.imshow(img_show[:, :, ::-1])
        plt.show()
        exit(1)

    return single_line


def interp_line_points_with_same_dist(start_point, end_point, segment_length=0.1):
    # 计算两个点之间的距离
    distance = np.linalg.norm(end_point - start_point)
    if distance < 2 * segment_length:
        points = [start_point, end_point]

    else:
        # 计算每个点之间的等距离
        # segment_length = distance / (num_points - 1)
        num_points = int(distance/segment_length)

        # 计算每个点的坐标
        points = []
        for i in range(num_points):
            ratio = i / (num_points - 1)  # 计算当前点在整个直线上的比例
            point = start_point + ratio * (end_point - start_point)  # 根据比例计算当前点的坐标
            points.append(point)

    points = np.array(points)
    return points


# 采用插补的方式补充点
def dense_line_points_by_interp(img, points):
    interp_points = []

    for i in range(0, len(points)-1):
        # 插补完再进行滑窗平滑
        # print(i, len(points)-1, points)
        start_point = points[i]  # 起点坐标
        end_point = points[i+1]  # 终点坐标
        segment_length = 1        # 间隔为1个像素
        dense_points = interp_line_points_with_same_dist(start_point, end_point, segment_length=segment_length)
        interp_points.append(dense_points)
    interp_points = np.concatenate(interp_points, axis=0)
    return interp_points


# 采用插补的方式补充点,同时补充点相关属性
def dense_line_points_by_interp_with_attr(img, points,  points_type,
                                          points_visible, points_hanging, points_covered):
    interp_points = []
    interp_points_type = []
    interp_points_visible = []
    interp_points_hanging = []
    interp_points_covered = []

    for i in range(0, len(points)-1):
        # 插补完再进行滑窗平滑
        # print(i, len(points)-1, points)
        start_point = points[i]  # 起点坐标
        end_point = points[i+1]  # 终点坐标
        segment_length = 1        # 间隔为1个像素
        dense_points = interp_line_points_with_same_dist(start_point, end_point, segment_length=segment_length)
        interp_points.append(dense_points)

        num_dense = int(dense_points.shape[0])

        # 对属性进行插值, 插入为起点的属性
        point_type = points_type[i]
        point_visible = points_visible[i]
        point_hanging = points_hanging[i]
        point_covered = points_covered[i]

        interp_point_type = np.array(list(point_type) * num_dense).reshape(-1, 1)
        interp_point_visible = np.array(list(point_visible) * num_dense).reshape(-1, 1)
        interp_point_hanging = np.array(list(point_hanging) * num_dense).reshape(-1, 1)
        interp_point_covered = np.array(list(point_covered) * num_dense).reshape(-1, 1)

        interp_points_type.append(interp_point_type)
        interp_points_visible.append(interp_point_visible)
        interp_points_hanging.append(interp_point_hanging)
        interp_points_covered.append(interp_point_covered)

    interp_points = np.concatenate(interp_points, axis=0)
    interp_points_type = np.concatenate(interp_points_type, axis=0)
    interp_points_visible = np.concatenate(interp_points_visible, axis=0)
    interp_points_hanging = np.concatenate(interp_points_hanging, axis=0)
    interp_points_covered = np.concatenate(interp_points_covered, axis=0)
    return interp_points, interp_points_type, interp_points_visible, interp_points_hanging, interp_points_covered
