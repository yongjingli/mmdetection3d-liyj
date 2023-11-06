import cv2
import numpy as np


class GBLDMetricDebug():
    def __init__(self, test_line_thinkness=20, test_t_iou=0.3):
        self.line_thinkness = test_line_thinkness
        self.t_iou = test_t_iou
        self.test_stage = [0]

    def gbld_evaluate_line_pixel(self,
                                 result_list,
                                 metric='line_pixel',
                                 classes=None):

        # 初始化测评指标
        measure_names = ["all"] + classes
        metric_f1_line_pixel = {}
        metric_f1_line_instance = {}

        for measure_name in measure_names:
            metric_f1_line_pixel[measure_name] = []
            metric_f1_line_instance[measure_name] = []

        for result in result_list:
            stages_pred_result = result['stages_pred_result']

            eval_gt_lines = result['eval_gt_lines']
            batch_input_shape = result['batch_input_shape']
            sample_idx = result['sample_idx']

            # 统计单个样本在不同类别中的指标
            for class_id, class_name in enumerate(measure_names):
                class_id = class_id - 1
                measure_gt_lines = []

                # gt-line-map
                for eval_gt_line in eval_gt_lines:
                    gt_label = eval_gt_line["label"]
                    gt_line = eval_gt_line["points"]
                    gt_line_cls = eval_gt_line["category_id"]

                    if class_name == "all":
                        measure_gt_lines.append(gt_line)
                    elif gt_line_cls == class_id:
                        measure_gt_lines.append(gt_line)

                measure_pred_lines = []
                for stage_idx in self.test_stage:
                    pred_result = stages_pred_result[stage_idx]

                    for pred_line in pred_result:
                        # image_xs, image_ys, confidences, emb_ids, clses
                        pred_line = np.array(pred_line)
                        pred_line_cls = pred_line[0][4]

                        if class_name == "all":
                            measure_pred_lines.append(pred_line[:, :2])
                        elif pred_line_cls == class_id:
                            measure_pred_lines.append(pred_line[:, :2])

                # 像素级的统计信息
                precision_pixel, recall_pixel, fscore_pixel = self.get_line_map_F1(measure_gt_lines,
                                                                                   measure_pred_lines,
                                                                                   batch_input_shape,
                                                                                   thickness=self.line_thinkness)
                metric_f1_line_pixel[class_name].append([precision_pixel, recall_pixel, fscore_pixel])

                # 实例级别的统计信息
                acc_list, recall_list = self.get_line_instance_F1(measure_gt_lines, measure_pred_lines,
                                                                  batch_input_shape, thickness=self.line_thinkness,
                                                                  t_iou=self.t_iou)
                metric_f1_line_instance[class_name].append([acc_list, recall_list])

        # 求所有类别测评的平均
        metric_f1_line_pixel = self.get_static_infos(metric_f1_line_pixel, measure_names)
        metric_f1_line_instance = self.get_static_infos_instance(metric_f1_line_instance, measure_names)

        return metric_f1_line_pixel, metric_f1_line_instance

    def get_static_infos_instance(self, measure_data, measure_names):
        for measure_name in measure_names:
            measure_datas = measure_data[measure_name]
            accs = []
            recalls = []
            for data in measure_datas:
                accs = accs + data[0]
                recalls = recalls + data[1]
            if len(accs) == 0:
                precision = -1
            else:
                precision = sum(accs) / len(accs)
                precision = round(precision, 2)
            if len(recalls) == 0:
                recall = -1
            else:
                recall = sum(recalls) / len(recalls)
                recall = round(recall, 2)

            eps = 0.001
            if len(accs) == 0 or len(recalls) == 0:
                fscore = -1
            else:
                fscore = (2 * precision * recall) / (precision + recall + eps)

            measure_data[measure_name] = [precision, recall, fscore]
        return measure_data

    def get_static_infos(self, measure_data, measure_names):
        for measure_name in measure_names:
            measure_datas = measure_data[measure_name]
            measure_datas = np.array(measure_datas)

            precision = measure_datas[:, 0]
            recall = measure_datas[:, 1]
            fscore = measure_datas[:, 2]

            precision = precision[precision != -1]
            recall = recall[recall != -1]
            fscore = fscore[fscore != -1]

            mean_precision = round(np.mean(precision), 2) if len(precision) != 0 else -1
            mean_recall = round(np.mean(recall), 2) if len(recall) != 0 else -1
            mean_fscore = round(np.mean(fscore), 2) if len(fscore) != 0 else -1

            measure_data[measure_name] = [mean_precision, mean_recall, mean_fscore]
        return measure_data

    def get_line_instance_F1(self, measure_gt_lines, measure_pred_lines, heatmap_size, thickness=3, t_iou=0.3):
        # 得到某个类别的F1
        acc_list = [False] * len(measure_pred_lines)
        recall_list = [False] * len(measure_gt_lines)

        gt_lines_heatmap = self.get_line_heatmap(measure_gt_lines, heatmap_size, thickness=thickness)
        pred_lines_heatmap = self.get_line_heatmap(measure_pred_lines, heatmap_size, thickness=thickness)

        gt_lines_heatmap = np.array(gt_lines_heatmap, np.float32)
        pred_lines_heatmap = np.array(pred_lines_heatmap, np.float32)

        # acc
        for i, measure_pred_line in enumerate(measure_pred_lines):
            pred_line_heatmap = self.get_line_heatmap([measure_pred_line], heatmap_size, thickness=thickness)
            pred_line_heatmap = np.array(pred_line_heatmap, np.float32)

            ap_pixel = np.sum(gt_lines_heatmap * pred_line_heatmap)
            all_pixel = np.sum(pred_line_heatmap)
            iou = ap_pixel / all_pixel
            if iou > t_iou:
                acc_list[i] = True

        # recall
        for i, measure_gt_line in enumerate(measure_gt_lines):
            gt_line_heatmap = self.get_line_heatmap([measure_gt_line], heatmap_size, thickness=thickness)
            gt_line_heatmap = np.array(gt_line_heatmap, np.float32)

            ap_pixel = np.sum(pred_lines_heatmap * gt_line_heatmap)
            all_pixel = np.sum(gt_line_heatmap)
            iou = ap_pixel / all_pixel
            if iou > t_iou:
                recall_list[i] = True
        return acc_list, recall_list

    def get_line_map_F1(self, measure_gt_lines, measure_pred_lines, heatmap_size, thickness=3):
        gt_lines_heatmap = self.get_line_heatmap(measure_gt_lines, heatmap_size, thickness=thickness)
        pred_lines_heatmap = self.get_line_heatmap(measure_pred_lines, heatmap_size, thickness=thickness)

        # debug
        # plt.subplot(2, 1, 1)
        # plt.imshow(gt_lines_heatmap)
        # plt.subplot(2, 1, 2)
        # plt.imshow(pred_lines_heatmap)
        # plt.show()
        # print("fff")
        # exit(1)

        gt_heatmap = np.array(gt_lines_heatmap, np.float32)
        pred_heatmap = np.array(pred_lines_heatmap, np.float32)

        intersection = np.sum(gt_heatmap * pred_heatmap)
        # union = np.sum(gt_heatmap) + np.sum(gt_heatmap)
        eps = 0.001
        # dice = (2. * intersection + eps) / (union + eps)

        recall = intersection / (np.sum(gt_heatmap) + eps)
        precision = intersection / (np.sum(pred_heatmap) + eps)

        fscore = (2 * precision * recall) / (precision + recall + eps)

        if len(measure_gt_lines) == 0:
            precision = -1
            fscore = -1

        if len(measure_pred_lines) == 0:
            recall = -1
            fscore = -1

        return precision, recall, fscore

    def get_line_heatmap(self, lines, heatmap_size, thickness=3):
        lines_heatmap = np.zeros(heatmap_size, dtype=np.uint8)
        img_h, img_w = heatmap_size
        for line in lines:
            pre_point = line[0]
            for cur_point in line[1:]:
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                draw_thickness = thickness
                if y1 > img_h//2 and y2 > img_h//2:
                    # draw_thickness = draw_thickness + 20
                    draw_thickness = int(draw_thickness * 1.4)

                # cv2.line(lines_heatmap, (x1, y1), (x2, y2), (1), thickness, 8)
                cv2.line(lines_heatmap, (x1, y1), (x2, y2), (1), draw_thickness, 8)
                pre_point = cur_point

        return lines_heatmap

    def get_measure_pred_lines_ap(self, measure_pred_lines, measure_gt_lines, heatmap_size):
        ap_list = [False] * len(measure_pred_lines)
        gt_lines_heatmap = self.get_line_heatmap(measure_gt_lines, heatmap_size, thickness=self.line_thinkness)

        for i, measure_pred_line in enumerate(measure_pred_lines):
            pred_line_heatmap = self.get_line_heatmap([measure_pred_line], heatmap_size, thickness=self.line_thinkness)
            pred_line_heatmap = np.array(pred_line_heatmap, np.float32)

            ap_pixel = np.sum(gt_lines_heatmap * pred_line_heatmap)
            all_pixel = np.sum(pred_line_heatmap)
            iou = ap_pixel / all_pixel
            if iou > self.t_iou:
                ap_list[i] = True
        return ap_list