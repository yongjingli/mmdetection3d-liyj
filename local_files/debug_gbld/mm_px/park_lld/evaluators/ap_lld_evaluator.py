import numpy as np
from scipy.spatial.distance import cdist

from xpilot_lightning.machine_learning.tasks.builder import EVALUATORS
from xpilot_vision.tasks.base.evaluators.utils import robust_div


@EVALUATORS.register_module
class AP_LLDEvaluator(object):
    def __init__(self, global_config, task_config, print_to_terminal=False):
        """All evaluation related vars are dict[filename] = values
        only few of vars are still in list, will consider whether convert them or not
        use dict to store value make it easier to track the detail for each image and future develop
        Args:
            config: ap_lld config
            print_to_terminal: if print evaluation results to terminal
        """
        self.global_config = global_config
        self.config = task_config
        self.grid_size = task_config.grid_size
        self.line_conf_thresh = task_config.validation_xboard["line_confidence_threshold"]
        self.line_point_ratio = task_config.validation_xboard["line_point_ratio"]
        self.line_height_range = task_config.validation_xboard["line_height_range"]
        self.mark_loc_distance = task_config.validation_xboard["mark_location_distance"]
        self.mark_type_total = len(task_config.arrow2id)
        self.print_to_terminal = print_to_terminal
        self._reset()

    def _reset(self):
        # List to store bad cases, for AutoQA check
        self.AutoQA = {"line_exist": set([]), "line_point": set([])}
        # For recall/precision table
        self.line_exist_pred = {}
        self.line_exist_gt = {}
        self.line_point_pred = {}
        self.line_point_gt = {}
        self.mark_location = {}
        self.mark_type = {}
        self.mark_vertex_type = {}
        # For distill purpose
        self.distill = {k: [] for k, v in self.AutoQA.items()}
        # Dict to store all evaluation results
        self.ap_lld_evaluation_results = {}
        # list to store the filename of empty prediction
        self.empty_pred = []  # empty filtered_pred

    def process_once(self, pred_dict, gt_dict, filename):
        """Func to call while evaluator ap_lld task

        This function processes one single image.

        Args:
            pred_dict: dict of the ap_lld pred
            gt_dict: dict of the ap_lld gt
            filename: filename of paired dict

        Returns: None

        """
        score = {k: [] for k, v in self.AutoQA.items()}
        pred_lane, gt_lane = pred_dict["lanes"], gt_dict["lanes"]
        # Line level
        pred_, gt_ = self.compute_line_exist(pred_lane, gt_lane, filename, score)
        self.line_exist_pred[filename] = pred_
        self.line_exist_gt[filename] = gt_
        # Point level
        pred_, gt_ = self.compute_line_points(pred_lane, gt_lane, filename, score)
        self.line_point_pred[filename] = pred_
        self.line_point_gt[filename] = gt_

        pred_mark, gt_mark = pred_dict["marks"], gt_dict["marks"]
        # Vertex level
        self.compute_mark_vertices(pred_mark, gt_mark, filename)

        # Distill
        for k, v in score.items():
            average_score = sum(v) / len(v) if len(v) != 0 else np.nan
            self.distill[k].append(average_score)

    def compute_line_exist(self, pred_lanes, gt_lanes, filename, score):
        """compute line_exist feature accuracy."""
        pred_lines = self.get_line_list(pred_lanes)
        gt_lines = self.get_line_list(gt_lanes)
        tp_num, pred_num, gt_num = self.get_line_number(
            pred_lines, gt_lines, self.line_conf_thresh, self.line_point_ratio
        )
        if tp_num != gt_num:
            self.AutoQA["line_exist"].add(filename)
        valid_points = {}
        for each_line in pred_lines:
            for grid, point in each_line.items():
                if point[2] >= self.line_conf_thresh:
                    valid_points[grid] = point
        for each_line in gt_lines:
            point_dict = {}
            for grid, point in each_line.items():
                if grid in valid_points:
                    point_dict[grid] = point
            if len(point_dict) / len(each_line) >= self.line_point_ratio:
                for grid, point in point_dict.items():
                    score["line_exist"].append(point[2])
        return pred_lines, gt_lines

    def get_line_list(self, lanes):
        line_list = []
        for each_lane in lanes:
            for _, line in each_lane.items():
                point_dict = {}
                for point in line["points"]:
                    x = int(point["x"])
                    y = int(point["y"])
                    s = point["score"]
                    grid_x = x // self.grid_size
                    grid_y = y // self.grid_size
                    point_dict[(grid_x, grid_y)] = [x, y, s]
                line_list.append(point_dict)
        return line_list

    def get_line_number(self, pred_lines, gt_lines, point_thresh, point_ratio):
        valid_lines = []
        valid_points = {}
        for each_line in pred_lines:
            point_dict = {}
            for grid, point in each_line.items():
                if point[2] >= point_thresh:
                    point_dict[grid] = point
                    valid_points[grid] = point
            if len(point_dict) > 0:
                valid_lines.append(point_dict)
        pred_num = len(valid_lines)
        gt_num = len(gt_lines)
        tp_num = 0
        for each_line in gt_lines:
            point_dict = {}
            for grid, point in each_line.items():
                if grid in valid_points:
                    point_dict[grid] = point
            if len(point_dict) / len(each_line) >= point_ratio:
                tp_num += 1
        return tp_num, pred_num, gt_num

    def compute_line_points(self, pred_lanes, gt_lanes, filename, score):
        """compute line_point feature accuracy."""
        pred_points = self.get_point_dict(pred_lanes)
        gt_points = self.get_point_dict(gt_lanes)
        tp_num, pred_num, gt_num = self.get_point_number(
            pred_points, gt_points, self.line_conf_thresh
        )
        if gt_num > 0 and tp_num / gt_num < self.line_point_ratio:
            self.AutoQA["line_point"].add(filename)
        for grid, point in pred_points.items():
            if grid in gt_points and point[2] >= self.line_conf_thresh:
                score["line_point"].append(point[2])
        return pred_points, gt_points

    def get_point_dict(self, lanes):
        point_dict = {}
        for each_lane in lanes:
            for _, line in each_lane.items():
                for point in line["points"]:
                    x = int(point["x"])
                    y = int(point["y"])
                    s = point["score"]
                    grid_x = x // self.grid_size
                    grid_y = y // self.grid_size
                    point_dict[(grid_x, grid_y)] = [x, y, s]
        return point_dict

    def get_point_number(self, pred_points, gt_points, threshold):
        tp_num, pred_num, gt_num = 0, 0, 0
        for grid, point in pred_points.items():
            if point[2] >= threshold:
                if grid in gt_points:
                    tp_num += 1
                pred_num += 1
        gt_num = len(gt_points)
        return tp_num, pred_num, gt_num

    def compute_mark_vertices(self, pred_marks, gt_marks, filename):
        pred_location, pred_type, pred_vertex = self.get_vertex_list(pred_marks)
        gt_location, gt_type, gt_vertex = self.get_vertex_list(gt_marks)
        # Location
        location_statistics = self.count_location(pred_location, gt_location, self.mark_loc_distance)
        self.mark_location[filename] = location_statistics
        # Type
        type_statistics = self.count_type(pred_type, gt_type, location_statistics['correlation'])
        self.mark_type[filename] = type_statistics['confusion']
        # Vertex type
        vertex_statistics = self.count_vertex_type(pred_vertex, gt_vertex, location_statistics['correlation'])
        self.mark_vertex_type[filename] = vertex_statistics['confusion']

    def get_vertex_list(self, marks):
        locations, types, vertex_types = [], [], []
        for mark in marks:
            if 'head' in mark:
                locations.append(mark['head'])
                types.append(int(mark['type']))
                vertex_types.append(0)
            if 'tail' in mark:
                locations.append(mark['tail'])
                types.append(int(mark['type']))
                vertex_types.append(1)
        return locations, types, vertex_types

    def count_location(self, pred_location, gt_location, location_distance):
        if len(pred_location) == 0:
            TP = 0
            FP = 0
            FN = len(gt_location)
            correlation_matrix = np.array([], dtype=np.bool)
        elif len(gt_location) == 0:
            TP = 0
            FP = len(pred_location)
            FN = 0
            correlation_matrix = np.array([], dtype=np.bool)
        else:
            pred_location = np.array([p[0:2] for p in pred_location])
            gt_location   = np.array([g[0:2] for g in gt_location])
            distance = cdist(pred_location, gt_location)
            best_gt_for_pred = distance.min(axis=1).reshape(-1, 1)
            best_pred_for_gt = distance.min(axis=0).reshape(1, -1)
            best_gt_for_pred = (distance == best_gt_for_pred)
            best_pred_for_gt = (distance == best_pred_for_gt)
            valid_pred_gt = (distance < location_distance)
            correlation_matrix = best_gt_for_pred & best_pred_for_gt & valid_pred_gt
            if correlation_matrix.sum(axis=1).max() > 1:
                row_indices = np.argmax(correlation_matrix, axis=1)
                row_indices = np.expand_dims(row_indices, axis=1)
                correlation_matrix.fill(False)
                np.put_along_axis(correlation_matrix, row_indices, True, axis=1)
            TP = correlation_matrix.sum()
            FP = len(pred_location) - TP
            FN = len(gt_location) - TP
        statistics = {'TP': TP, 'FP': FP, 'FN': FN, 'correlation': correlation_matrix}
        return statistics

    def count_type(self, pred_type, gt_type, correlation_matrix):
        if correlation_matrix.sum() == 0:
            confusion_matrix = np.zeros((self.mark_type_total, self.mark_type_total), dtype=np.int)
        else:
            pred_type = np.array(pred_type)
            gt_type   = np.array(gt_type)
            m = pred_type.size
            n = gt_type.size
            pred_type = pred_type.reshape(-1, 1).repeat(n, axis=1)
            gt_type   = gt_type.reshape(1, -1).repeat(m, axis=0)
            pred_type = pred_type[correlation_matrix]
            gt_type   = gt_type[correlation_matrix]
            confusion_matrix = np.zeros((self.mark_type_total, self.mark_type_total), dtype=np.int)
            for p, g in zip(pred_type, gt_type):
                confusion_matrix[p, g] += 1
        statistics = {'confusion': confusion_matrix}
        return statistics

    def count_vertex_type(self, pred_vertex, gt_vertex, correlation_matrix):
        if correlation_matrix.sum() == 0:
            confusion_matrix = np.zeros((2, 2), dtype=np.int)
        else:
            pred_vertex = np.array(pred_vertex)
            gt_vertex   = np.array(gt_vertex)
            m = pred_vertex.size
            n = gt_vertex.size
            pred_vertex = pred_vertex.reshape(-1, 1).repeat(n, axis=1)
            gt_vertex   = gt_vertex.reshape(1, -1).repeat(m, axis=0)
            pred_vertex = pred_vertex[correlation_matrix]
            gt_vertex   = gt_vertex[correlation_matrix]
            confusion_matrix = np.zeros((2, 2), dtype=np.int)
            for p, g in zip(pred_vertex, gt_vertex):
                confusion_matrix[p, g] += 1
        statistics = {'confusion': confusion_matrix}
        return statistics

    def eval_metrics(self, is_ips=False):
        """Display metrics

        Returns: Overall evaluation results

        """
        def calc_recall(num_TP, num_FN):
            return robust_div(num_TP, num_TP + num_FN)

        def calc_precision(num_TP, num_FP):
            return robust_div(num_TP, num_TP + num_FP)

        def calc_f1score(recall, precision):
            return 2 * robust_div(recall * precision, recall + precision)

        # Line
        self.ap_lld_evaluation_results[
            "line_exist_pr_threshold"
        ] = self.get_line_exist_pr_threshold()
        self.ap_lld_evaluation_results[
            "line_point_pr_threshold"
        ] = self.get_line_point_pr_threshold()
        self.ap_lld_evaluation_results["line_point_pr_distance"] = self.get_line_point_pr_distance()
        self.ap_lld_evaluation_results[
            "line_point_position_error"
        ] = self.get_line_point_position_error()

        # Arrow
        total_TP = sum([num['TP'] for _, num in self.mark_location.items()])
        total_FP = sum([num['FP'] for _, num in self.mark_location.items()])
        total_FN = sum([num['FN'] for _, num in self.mark_location.items()])
        recall = calc_recall(total_TP, total_FN)
        precision = calc_precision(total_TP, total_FP)
        self.ap_lld_evaluation_results["mark_location"] = [
            recall, precision, calc_f1score(recall, precision), total_TP + total_FP, total_TP + total_FN
        ]

        type_TP, type_FP = 0, 0
        for _, confusion in self.mark_type.items():
            type_TP += confusion.trace()
            type_FP += confusion.sum() - confusion.trace()
        self.ap_lld_evaluation_results["mark_type"] = [
            calc_precision(type_TP, type_FP), type_TP, type_TP + type_FP
        ]

        vertex_confusion = np.zeros((2, 2), dtype=np.int)
        for _, confusion in self.mark_vertex_type.items():
            vertex_confusion += confusion
        vertex_TP = vertex_confusion[0, 0] + vertex_confusion[1, 1]
        vertex_FP = vertex_confusion[0, 1] + vertex_confusion[1, 0]
        self.ap_lld_evaluation_results["mark_vertex_type"] = [
            calc_precision(vertex_TP, vertex_FP), vertex_TP, vertex_TP + vertex_FP
        ]

        if self.print_to_terminal:
            self.check_corner_case()
            for kpi in self.ap_lld_evaluation_results:
                print(kpi, ": ", self.ap_lld_evaluation_results[kpi])
        return self.ap_lld_evaluation_results

    def get_line_exist_pr_threshold(self):
        pr_table = {}
        threshold = self.line_conf_thresh
        while threshold < 1.0:
            tp_total, pred_total, gt_total = 0, 0, 0
            for uuid in self.line_exist_pred.keys():
                pred_lines = self.line_exist_pred[uuid]
                gt_lines = self.line_exist_gt[uuid]
                tp_num, pred_num, gt_num = self.get_line_number(
                    pred_lines, gt_lines, threshold, self.line_point_ratio
                )
                tp_total += tp_num
                pred_total += pred_num
                gt_total += gt_num
            recall = "NaN" if gt_total == 0 else tp_total / gt_total
            precision = "NaN" if pred_total == 0 else tp_total / pred_total
            pr_table[threshold] = {
                "recall": recall,
                "precision": precision,
                "gt_num": gt_total,
                "pred_num": pred_total,
            }
            threshold += 0.05
        return pr_table

    def get_line_point_pr_threshold(self):
        pr_table = {}
        threshold = self.line_conf_thresh
        while threshold < 1.0:
            tp_total, pred_total, gt_total = 0, 0, 0
            for uuid in self.line_point_pred.keys():
                pred_points = self.line_point_pred[uuid]
                gt_points = self.line_point_gt[uuid]
                tp_num, pred_num, gt_num = self.get_point_number(pred_points, gt_points, threshold)
                tp_total += tp_num
                pred_total += pred_num
                gt_total += gt_num
            recall = "NaN" if gt_total == 0 else tp_total / gt_total
            precision = "NaN" if pred_total == 0 else tp_total / pred_total
            pr_table[threshold] = {
                "recall": recall,
                "precision": precision,
                "gt_num": gt_total,
                "pred_num": pred_total,
            }
            threshold += 0.05
        return pr_table

    def get_line_point_pr_distance(self):
        pr_table = {}
        height_ranges = self.line_height_range
        for min_y, max_y in height_ranges:
            tp_total, pred_total, gt_total = 0, 0, 0
            for uuid in self.line_point_pred.keys():
                pred_points = self.line_point_pred[uuid]
                gt_points = self.line_point_gt[uuid]
                pred_, gt_ = {}, {}
                for grid, point in pred_points.items():
                    if point[1] >= min_y and point[1] < max_y:
                        pred_[grid] = point
                for grid, point in gt_points.items():
                    if point[1] >= min_y and point[1] < max_y:
                        gt_[grid] = point
                tp_num, pred_num, gt_num = self.get_point_number(pred_, gt_, self.line_conf_thresh)
                tp_total += tp_num
                pred_total += pred_num
                gt_total += gt_num
            recall = "NaN" if gt_total == 0 else tp_total / gt_total
            precision = "NaN" if pred_total == 0 else tp_total / pred_total
            key = str(min_y) + ":" + str(max_y)
            pr_table[key] = {
                "recall": recall,
                "precision": precision,
                "gt_num": gt_total,
                "pred_num": pred_total,
            }
        return pr_table

    def get_line_point_position_error(self):
        position_error = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
        for uuid in self.line_point_pred.keys():
            pred_points = self.line_point_pred[uuid]
            gt_points = self.line_point_gt[uuid]
            for grid, pd_pnt in pred_points.items():
                if grid in gt_points and pd_pnt[2] >= self.line_conf_thresh:
                    gt_pnt = gt_points[grid]
                    distance = np.sqrt((pd_pnt[0] - gt_pnt[0]) ** 2 + (pd_pnt[1] - gt_pnt[1]) ** 2)
                    k = min(round(distance.item()), 10)
                    position_error[k] += 1
        return position_error

    def check_corner_case(self):
        """Display the filename of corner case for visualization

        Returns: None

        """
        print("There are {} empty prediction".format(len(self.empty_pred)))
