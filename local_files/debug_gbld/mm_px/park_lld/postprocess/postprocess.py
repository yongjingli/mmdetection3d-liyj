import json

import numpy as np

from xpilot_vision.utils.json_helper.dump_helper import dump_to_json, JsonSerilizable
from xpilot_vision.tasks.utils import sigmoid, softmax
from xpilot_vision.tasks.base.postprocess.postprocess import BasePostProcess
from xpilot_lightning.machine_learning.tasks.builder import POSTPROCESSES


@POSTPROCESSES.register_module
class AP_LLDPostProcessing(BasePostProcess):
    def __init__(
        self, global_config, task_config, output_json_path="./ap_lld_jsons/", scale=8, **kwargs
    ):
        super().__init__(global_config, task_config, **kwargs)
        self.pred_channels = task_config.pred_channels
        self.gt_channels = task_config.gt_channels
        self.scale = task_config.grid_size if hasattr(task_config, "grid_size") else scale
        self.crop_size = task_config.crop_size
        self.map_pad = task_config.map_pad
        self.near_conf_thresh = task_config.near_confidence_threshold
        self.far_range = task_config.far_range
        self.far_conf_thresh = task_config.far_confidence_threshold
        self.pull_margin = task_config.pull_margin
        self.long_line_thresh = task_config.long_line_threshold
        self.far_line_thresh = task_config.far_line_threshold
        self.map_size = task_config.map_size
        self.fit_iteration = task_config.fitting_iteration
        self.fit_degree = task_config.fitting_degree
        self.fit_tolerance = task_config.fitting_tolerance
        self.fit_margin = task_config.fitting_margin
        self.endpoint_distance = task_config.endpoint_distance
        self.endpoint_precision = task_config.endpoint_precision
        self.line_maximum = task_config.line_maximum
        self.output_json_path = output_json_path

        self.mark_type_thresh = task_config.mark_type_threshold
        self.mark_conf_thresh = task_config.mark_confidence_threshold
        self.mark_nms_dist = task_config.mark_nms_distance ** 2
        self.mark_embed_dist = task_config.mark_embed_distance
        self.mark_min_group = task_config.mark_group_minimum
        self.mark_max_group = task_config.mark_group_maximum
        self.mark_vert_thresh = task_config.mark_vertex_threshold
        self.mark_height_thresh = task_config.mark_height_threshold
        self.mark_group_distance = [d ** 2 for d in task_config.mark_group_distance]

        self.ips_resize = global_config.ips_resize if hasattr(global_config,'ips_resize') else False
        self.ips_decode = global_config.ips_decode if hasattr(global_config,'ips_decode') else False

        # Reset
        self.reset()

    def reset(self):
        self.start_id = 0
        self.lanes = []
        self.marks = []

    def process(self, y_hat, is_gt=None, filename=None, output_json_path=None):
        self.reset()
        self.is_gt = is_gt

        if isinstance(y_hat, list):
            y_hat = np.array(y_hat).astype(np.float16)

        self.decode_lines(y_hat)
        self.decode_arrows(y_hat)

        if self.ips_decode:
            self.lanes = self.convert_points_lane(self.lanes)
            self.marks = self.convert_points_mark(self.marks)

        ap_lld_dict = {
            "task": "ap_lld",
            "file_name": filename if filename else "",
            "lanes": self.lanes,
            "marks": self.marks,
        }
        # Json Encode
        ap_lld_dict = json.loads(JsonSerilizable().encode(ap_lld_dict))

        if output_json_path:
            dump_to_json(ap_lld_dict, output_json_path)

        return self.lanes, ap_lld_dict

    def convert_to_visulize(self, x, y, ips_resize):
        rs1, up, buttom, left, right, rs2 = ips_resize
        x1 = (x * rs1 + left) * rs2
        y1 = (y * rs1 + up) * rs2
        return x1, y1

    def convert_points_lane(self, lanes):  # todo: round method and cam1 cam3,cam4 and debug
        points_lanes = []
        for lane in lanes:
            points_lane = {}
            if lane.__contains__('left_line'):
                points_lane['left_line'] = lane['left_line']
                convert_points = []
                for point in points_lane['left_line']['points']:
                    x = point['x']  # cam1 in 914,474; cam3,cam4 in 640,480
                    y = point['y']
                    score = point['score']
                    x, y = self.convert_to_visulize(x, y, self.ips_resize)
                    convert_points.append({'x': x, 'y': y, 'score': score})
                points_lane['left_line']['points'] = convert_points
            if lane.__contains__('right_line'):
                points_lane['right_line'] = lane['right_line']
                convert_points = []
                for point in points_lane['right_line']['points']:
                    x = point['x']  # cam1 in 914,474; cam3,cam4 in 640,480
                    y = point['y']
                    score = point['score']
                    x, y = self.convert_to_visulize(x, y, self.ips_resize)
                    convert_points.append({'x': x, 'y': y, 'score': score})
                points_lane['right_line']['points'] = convert_points

            points_lanes.append(points_lane)
        return points_lanes

    def convert_points_mark(self, marks):  # todo: round method and cam1 cam3,cam4 and debug
        points_marks = []
        for mark in marks:
            points_mark = mark
            if mark.__contains__('head'):
                x, y, score = mark['head']
                x, y = self.convert_to_visulize(x, y, self.ips_resize)
                points_mark['head'] = [x, y, score]
            if mark.__contains__('tail'):
                x, y, score = mark['tail']
                x, y = self.convert_to_visulize(x, y, self.ips_resize)
                points_mark['tail'] = [x, y, score]
            points_marks.append(points_mark)
        return points_marks

    def decode_lines(self, y_hat):
        pred_channels = self.gt_channels if self.is_gt else self.pred_channels
        id_p = self.start_id
        start_p, end_p = pred_channels[id_p], pred_channels[id_p + 1]
        pred_confidence = y_hat[start_p:end_p, ...]
        id_p += 1
        start_p, end_p = pred_channels[id_p], pred_channels[id_p + 1]
        pred_offset_x = y_hat[start_p : start_p + 1, ...]
        pred_offset_y = y_hat[start_p + 1 : end_p, ...]
        id_p += 1
        start_p, end_p = pred_channels[id_p], pred_channels[id_p + 1]
        pred_embedding = y_hat[start_p:end_p, ...]
        self.start_id = id_p + 1

        if self.is_gt:
            pred_offset_x = pred_offset_x * (self.scale - 1)
            pred_offset_y = pred_offset_y * (self.scale - 1)
            pred_embedding = pred_embedding.astype(np.int)
        else:
            pred_confidence = pred_confidence.clip(-20, 20)
            pred_confidence = sigmoid(pred_confidence)
            pred_offset_x = pred_offset_x.clip(-20, 20)
            pred_offset_y = pred_offset_y.clip(-20, 20)
            pred_offset_x = sigmoid(pred_offset_x) * (self.scale - 1)
            pred_offset_y = sigmoid(pred_offset_y) * (self.scale - 1)
        pred_offset_x = pred_offset_x.round().astype(np.int).clip(0, self.scale - 1)
        pred_offset_y = pred_offset_y.round().astype(np.int).clip(0, self.scale - 1)
        _, h, w = pred_offset_x.shape
        pred_grid_x = np.arange(w).reshape(1, 1, w).repeat(h, axis=1) * self.scale
        pred_grid_y = np.arange(h).reshape(1, h, 1).repeat(w, axis=2) * self.scale
        pred_x = pred_grid_x + pred_offset_x
        pred_y = pred_grid_y + pred_offset_y
        if len(self.crop_size) == 4:
            top, bottom, left, right = self.crop_size
            pred_x = pred_x + left * self.scale
            pred_y = pred_y + top * self.scale
        if len(self.map_pad) == 4:
            top, bottom, left, right = self.map_pad
            pred_x = pred_x - left
            pred_y = pred_y - top

        if self.is_gt:
            line_id = 0
            line_total = pred_embedding.max() + 1
            for i in range(1, line_total):
                mask = pred_embedding == i
                if np.any(mask):
                    xs = pred_x[mask]
                    ys = pred_y[mask]
                    if line_id % 2 == 0:
                        self.lanes.append({"left_line": {"points": []}})
                        line_name = "left_line"
                    else:
                        self.lanes[-1]["right_line"] = {"points": []}
                        line_name = "right_line"
                    for x, y in zip(xs, ys):
                        self.lanes[-1][line_name]["points"].append(
                            {"x": float(x), "y": float(y), "score": 1.0}
                        )
                    line_id += 1
        else:
            mask = pred_confidence > self.near_conf_thresh
            count = 0
            for i in range(len(mask[0])):
                for j in range(len(mask[0][0])):
                    if mask[0][i][j] == True:
                        count += 1
            # print('point_count:',count)
            if len(self.far_range) == 2:
                min_y, max_y = self.far_range
                mask[:, min_y:max_y, :] = pred_confidence[:, min_y:max_y, :] > self.far_conf_thresh
            if np.any(mask):
                lane_lines = self.connect_line_points(
                    mask[0], pred_embedding[0], pred_x[0], pred_y[0], pred_confidence[0]
                )
                for i, line in enumerate(lane_lines):
                    if i % 2 == 0:
                        self.lanes.append({"left_line": {"points": []}})
                        line_name = "left_line"
                        if "right_line" not in self.lanes[-1]:
                            self.lanes[-1]["right_line"] = {"points": []}
                    else:
                        self.lanes[-1]["right_line"] = {"points": []}
                        line_name = "right_line"
                    for x, y, conf in line:
                        self.lanes[-1][line_name]["points"].append(
                            {"x": float(x), "y": float(y), "score": conf}
                        )

    def connect_line_points(self, mask_map, embedding_map, x_map, y_map, confidence_map):
        ys, xs = np.nonzero(mask_map)
        ys, xs = np.flipud(ys), np.flipud(xs)
        ebs = embedding_map[ys, xs]
        raw_lines = self.cluster_line_points(xs, ys, ebs)
        raw_lines = self.remove_short_lines(raw_lines)
        raw_lines = self.remove_far_lines(raw_lines)
        exact_lines = []
        for each_line in raw_lines:
            single_line = self.serialize_single_line(each_line, x_map, y_map, confidence_map)
            if len(single_line) > 0:
                exact_lines.append(single_line)
        if len(exact_lines) == 0:
            return []
        exact_lines = sorted(exact_lines, key=lambda l: len(l), reverse=True)
        exact_lines = (
            exact_lines[: self.line_maximum]
            if len(exact_lines) > self.line_maximum
            else exact_lines
        )
        exact_lines = sorted(exact_lines, key=lambda l: l[0][0], reverse=False)
        # exact_lines = sorted(exact_lines, key=lambda l: np.array(l)[:, 0].mean(), reverse=False)
        return exact_lines

    def cluster_line_points(self, xs, ys, embeddings):
        lines = []
        embedding_means = []
        point_numbers = []
        for x, y, eb in zip(xs, ys, embeddings):
            id = None
            min_dist = 10000
            for i, eb_mean in enumerate(embedding_means):
                distance = abs(eb - eb_mean)
                if distance < self.pull_margin and distance < min_dist:
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

    def remove_short_lines(self, lines):
        long_lines = []
        for line in lines:
            if len(line) >= self.long_line_thresh:
                long_lines.append(line)
        return long_lines

    def remove_far_lines(self, lines):
        near_lines = []
        for line in lines:
            for point in line:
                if point[1] >= self.far_line_thresh:
                    near_lines.append(line)
                    break
        return near_lines

    def serialize_single_line(self, single_line, x_map, y_map, confidence_map):
        existing_points = single_line.copy()
        piecewise_lines = []
        while len(existing_points) > 0:
            existing_points = self.remove_isolated_points(existing_points)
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
                selected_points, outliers = self.select_function_points(
                    selected_points, near_points
                )
                if len(outliers) == len(near_points):
                    break
                else:
                    alternative_points = outliers + far_points
                    y -= 1
            selected_points = self.extend_endpoints(selected_points, single_line)
            piecewise_line = self.arrange_points_to_line(
                selected_points, x_map, y_map, confidence_map
            )
            # piecewise_line = self.fit_points_to_line(selected_points, x_map, y_map, confidence_map)  # Curve Fitting
            piecewise_lines.append(piecewise_line)
            existing_points = alternative_points
        if len(piecewise_lines) == 0:
            return []
        elif len(piecewise_lines) == 1:
            exact_lines = piecewise_lines[0]
        else:
            exact_lines = self.connect_piecewise_lines(piecewise_lines)[0]
        if exact_lines[0][1] < exact_lines[-1][1]:
            exact_lines.reverse()
        return exact_lines

    def remove_isolated_points(self, line_points):
        line_points = np.array(line_points)
        valid_points = []
        for point in line_points:
            distance = abs(point - line_points).max(axis=1)
            if np.any(distance == 1):
                valid_points.append(point.tolist())
        return valid_points

    def select_function_points(self, selected_points, near_points):
        while len(near_points) > 0:
            added_points = []
            for n_pnt in near_points:
                for s_pnt in selected_points:
                    distance = max(abs(n_pnt[0] - s_pnt[0]), abs(n_pnt[1] - s_pnt[1]))
                    if distance == 1:
                        vertical_distance = self.compute_vertical_distance(n_pnt, selected_points)
                        if vertical_distance <= 1:
                            selected_points = [n_pnt] + selected_points
                            added_points.append(n_pnt)
                            break
            if len(added_points) == 0:
                break
            else:
                near_points = [n_pnt for n_pnt in near_points if n_pnt not in added_points]
        return selected_points, near_points

    def compute_vertical_distance(self, point, selected_points):
        vertical_points = [s_pnt for s_pnt in selected_points if s_pnt[0] == point[0]]
        if len(vertical_points) == 0:
            return 0
        else:
            vertical_distance = 10000
            for v_pnt in vertical_points:
                distance = abs(v_pnt[1] - point[1])
                vertical_distance = distance if distance < vertical_distance else vertical_distance
            return vertical_distance

    def extend_endpoints(self, selected_points, single_line):
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

    def arrange_points_to_line(self, selected_points, x_map, y_map, confidence_map):
        selected_points = np.array(selected_points)
        xs, ys = selected_points[:, 0], selected_points[:, 1]
        image_xs = x_map[ys, xs]
        image_ys = y_map[ys, xs]
        confidences = confidence_map[ys, xs]
        indices = image_xs.argsort()
        image_xs = image_xs[indices]
        image_ys = image_ys[indices]
        confidences = confidences[indices]
        h, w = self.map_size
        line = []
        for x, y, conf in zip(image_xs, image_ys, confidences):
            x = min(x, w - 1)
            y = min(y, h - 1)
            line.append((x, y, conf))
        return line

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

    def connect_piecewise_lines(self, piecewise_lines):
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
                    distance = self.compute_point_distance(c_end, o_end)
                    if distance < min_dist:
                        point_ids[0] = i
                        point_ids[1] = j
                        min_dist = distance
            if min_dist < self.endpoint_distance:
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

    def compute_point_distance(self, point_0, point_1):
        distance = np.sqrt((point_0[0] - point_1[0]) ** 2 + (point_0[1] - point_1[1]) ** 2)
        return distance

    def decode_arrows(self, y_hat):
        pred_channels = self.gt_channels if self.is_gt else self.pred_channels
        id_p = self.start_id
        start_p, end_p = pred_channels[id_p], pred_channels[id_p + 1]
        pred_confidence = y_hat[start_p:end_p, ...]
        id_p += 1
        start_p, end_p = pred_channels[id_p], pred_channels[id_p + 1]
        pred_offset_x = y_hat[start_p : start_p + 1, ...]
        pred_offset_y = y_hat[start_p + 1 : end_p, ...]
        id_p += 1
        start_p, end_p = pred_channels[id_p], pred_channels[id_p + 1]
        pred_embedding = y_hat[start_p:end_p, ...]
        id_p += 1
        start_p, end_p = pred_channels[id_p], pred_channels[id_p + 1]
        pred_type = y_hat[start_p:end_p, ...]
        id_p += 1
        start_p, end_p = pred_channels[id_p], pred_channels[id_p + 1]
        pred_vertex_type = y_hat[start_p:end_p, ...]

        if self.is_gt:
            pred_offset_x = pred_offset_x * (self.scale - 1)
            pred_offset_y = pred_offset_y * (self.scale - 1)
            pred_type = pred_type.astype(np.int)
            pred_vertex_type = pred_vertex_type.astype(np.int)
        else:
            pred_confidence = pred_confidence.clip(-20, 20)
            pred_confidence = sigmoid(pred_confidence)
            pred_offset_x = pred_offset_x.clip(-20, 20)
            pred_offset_y = pred_offset_y.clip(-20, 20)
            pred_offset_x = sigmoid(pred_offset_x) * (self.scale - 1)
            pred_offset_y = sigmoid(pred_offset_y) * (self.scale - 1)
            pred_type = softmax(pred_type, axis=0)
            pred_type = np.argmax(pred_type, axis=0)[None, ...]
            pred_vertex_type = pred_vertex_type.clip(-20, 20)
            pred_vertex_type = sigmoid(pred_vertex_type)
            pred_vertex_type = np.where(pred_vertex_type > self.mark_type_thresh, 1, 0)
        pred_offset_x = pred_offset_x.round().astype(np.int).clip(0, self.scale - 1)
        pred_offset_y = pred_offset_y.round().astype(np.int).clip(0, self.scale - 1)
        _, h, w = pred_offset_x.shape
        pred_grid_x = np.arange(w).reshape(1, 1, w).repeat(h, axis=1) * self.scale
        pred_grid_y = np.arange(h).reshape(1, h, 1).repeat(w, axis=2) * self.scale
        pred_x = pred_grid_x + pred_offset_x
        pred_y = pred_grid_y + pred_offset_y
        if len(self.crop_size) == 4:
            top, bottom, left, right = self.crop_size
            pred_x = pred_x + left * self.scale
            pred_y = pred_y + top * self.scale
        if len(self.map_pad) == 4:
            top, bottom, left, right = self.map_pad
            pred_x = pred_x - left
            pred_y = pred_y - top

        if self.is_gt:
            mark_total = pred_embedding.max().astype(np.int) + 1
            for i in range(1, mark_total):
                mask = (pred_embedding == i)
                if np.any(mask):
                    xs = pred_x[mask]
                    ys = pred_y[mask]
                    embeds = pred_embedding[mask]
                    arrow_types = pred_type[mask]
                    vertex_types = pred_vertex_type[mask]
                    if mask.sum() == 2:
                        head = [xs[0], ys[0], 1.0] if vertex_types[0] == 0 else [xs[1], ys[1], 1.0]
                        tail = [xs[1], ys[1], 1.0] if vertex_types[0] == 0 else [xs[0], ys[0], 1.0]
                        arrow_type = arrow_types[0]
                        self.marks.append({"head": head, "tail": tail, "type": arrow_type})
                    else:
                        if vertex_types[0] == 0:
                            head = [xs[0], ys[0], 1.0]
                            arrow_type = arrow_types[0]
                            self.marks.append({"head": head, "type": arrow_type})
                        else:
                            tail = [xs[0], ys[0], 1.0]
                            arrow_type = arrow_types[0]
                            self.marks.append({"tail": tail, "type": arrow_type})
        else:
            mask = pred_confidence > self.mark_conf_thresh
            xs = pred_x[mask]
            ys = pred_y[mask]
            scores = pred_confidence[mask]
            embeds = pred_embedding[mask]
            arrow_types = pred_type[mask]
            vertex_types = pred_vertex_type[mask]
            valid_points = self.custom_nms(xs, ys, scores, embeds, arrow_types, vertex_types)
            valid_points = sorted(valid_points, key=lambda p: p[1], reverse=True)
            while len(valid_points) > 0:
                i_point = valid_points.pop(0)
                i_x, i_y, i_s, i_e, i_t, i_v = i_point
                if i_y < self.mark_height_thresh:
                    break

                relevant_points, irrelevant_points = [], []
                for j_point in valid_points:
                    j_x, j_y, j_s, j_e, j_t, j_v = j_point
                    embedding = abs(i_e - j_e)
                    distance = self.compute_nms_distance((i_x, i_y), (j_x, j_y))
                    if embedding > self.mark_embed_dist:
                        irrelevant_points.append(j_point)
                    elif embedding < self.mark_embed_dist and distance > self.mark_group_distance[0]:
                        irrelevant_points.append(j_point)
                    elif embedding < self.mark_embed_dist and distance < self.mark_group_distance[1]:
                        relevant_points.append(j_point)
                    elif embedding < self.mark_embed_dist and i_t == j_t:
                        relevant_points.append(j_point)
                    else:
                        irrelevant_points.append(j_point)
                relevant_points.append(i_point)
                valid_points = irrelevant_points

                head, tail = None, None
                for point in relevant_points:
                    x, y, s, e, t, v = point
                    if v == 0:
                        if not head or s > head[2]:
                            head = point
                    else:
                        if not tail or s > tail[2]:
                            tail = point

                type_number = {}
                y_max, alternative = 0, None
                for point in relevant_points:
                    x, y, s, e, t, v = point
                    if t in type_number:
                        type_number[t] += 1
                    else:
                        type_number[t] = 1
                    if y > y_max:
                        y_max = y
                        alternative = t
                type_number = [(n, t) for t, n in type_number.items()]
                type_number = sorted(type_number, key=lambda x: x[0], reverse=True)
                if len(type_number) > 1 and type_number[0][0] == type_number[1][0]:
                    arrow_type = alternative
                else:
                    arrow_type = type_number[0][1]

                if head and tail:
                    self.marks.append({"head": head[0:3], "tail": tail[0:3], "type": arrow_type})
                elif head:
                    self.marks.append({"head": head[0:3], "type": arrow_type})
                elif tail:
                    self.marks.append({"tail": tail[0:3], "type": arrow_type})

    def custom_nms(self, xs, ys, scores, embeds, arrow_types, vertex_types):
        candidate_points = []
        for x, y, s, e, t, v in zip(xs, ys, scores, embeds, arrow_types, vertex_types):
            point = {'x': x, 'y': y, 's': s, 'e': e, 't': t, 'v': v}
            candidate_points.append(point)

        point_groups = []
        while len(candidate_points) > 0:
            group = []
            queue = [candidate_points.pop()]
            while len(queue) > 0 and len(candidate_points) > 0:
                i = queue.pop()
                i_x, i_y, i_e = i['x'], i['y'], i['e']
                irrelevant_points = []
                for j in candidate_points:
                    j_x, j_y, j_e = j['x'], j['y'], j['e']
                    distance = self.compute_nms_distance((i_x, i_y), (j_x, j_y))
                    embedding = abs(i_e - j_e)
                    if distance > self.mark_nms_dist or embedding > self.mark_embed_dist:
                        irrelevant_points.append(j)
                    else:
                        queue.append(j)
                group.append(i)
                candidate_points = irrelevant_points
            group += queue
            point_groups.append(group)

        valid_points = []
        for group in point_groups:
            if len(group) >= self.mark_min_group:
                xs, ys, ss, es, ts, vs = [], [], [], [], [], []
                for point in group:
                    xs.append(point['x'])
                    ys.append(point['y'])
                    ss.append(point['s'])
                    es.append(point['e'])
                    ts.append(point['t'])
                    vs.append(point['v'])
                indices = np.argsort(ss)[::-1]
                indices = indices[:min(len(indices), self.mark_max_group)]
                best_x = np.array(xs)[indices].mean()
                best_y = np.array(ys)[indices].mean()
                best_s = np.array(ss)[indices].mean()
                best_e = np.array(es)[indices].mean()
                best_t = np.array(ts)[indices][0]
                best_v = 0 if np.array(vs)[indices].mean() <= self.mark_vert_thresh else 1
                valid_points.append([best_x, best_y, best_s, best_e, best_t, best_v])
        return valid_points

    def compute_nms_distance(self, point_1, point_2):
        x1, y1 = point_1
        x2, y2 = point_2
        distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        return distance
