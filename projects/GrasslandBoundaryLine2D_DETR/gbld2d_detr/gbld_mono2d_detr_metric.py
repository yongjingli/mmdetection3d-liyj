# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import json
import os
import mmengine
import numpy as np
import pyquaternion
import torch
from mmengine import Config, load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)

from mmengine.utils import ProgressBar, mkdir_or_exist

import matplotlib.pyplot as plt

@METRICS.register_module()
class GbldDetrMetric(BaseMetric):
    """Nuscenes evaluation metric.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        jsonfile_prefix (str, optional): The prefix of json files including the
            file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        eval_version (str): Configuration version of evaluation.
            Defaults to 'detection_cvpr_2019'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    # NameMapping = {
    #     'movable_object.barrier': 'barrier',
    #     'vehicle.bicycle': 'bicycle',
    #     'vehicle.bus.bendy': 'bus',
    #     'vehicle.bus.rigid': 'bus',
    #     'vehicle.car': 'car',
    #     'vehicle.construction': 'construction_vehicle',
    #     'vehicle.motorcycle': 'motorcycle',
    #     'human.pedestrian.adult': 'pedestrian',
    #     'human.pedestrian.child': 'pedestrian',
    #     'human.pedestrian.construction_worker': 'pedestrian',
    #     'human.pedestrian.police_officer': 'pedestrian',
    #     'movable_object.trafficcone': 'traffic_cone',
    #     'vehicle.trailer': 'trailer',
    #     'vehicle.truck': 'truck'
    # }
    # DefaultAttribute = {
    #     'car': 'vehicle.parked',
    #     'pedestrian': 'pedestrian.moving',
    #     'trailer': 'vehicle.parked',
    #     'truck': 'vehicle.parked',
    #     'bus': 'vehicle.moving',
    #     'motorcycle': 'cycle.without_rider',
    #     'construction_vehicle': 'vehicle.parked',
    #     'bicycle': 'cycle.without_rider',
    #     'barrier': '',
    #     'traffic_cone': '',
    # }
    # # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    # ErrNameMapping = {
    #     'trans_err': 'mATE',
    #     'scale_err': 'mASE',
    #     'orient_err': 'mAOE',
    #     'vel_err': 'mAVE',
    #     'attr_err': 'mAAE'
    # }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 dataset_meta: Optional[dict] = None,
                 test_stage: list = [],
                 metric: Union[str, List[str]] = 'line_pixel',
                 line_thinkness=3,
                 t_iou=0.3,
                 rescale=True,
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 eval_version: str = 'detection_cvpr_2019',
                 collect_device: str = 'cpu',
                 output_dir: bool = None,
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Gbld metric'
        super(GbldDetrMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
            )
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.format_only = format_only
        if self.format_only:
            assert jsonfile_prefix is not None, 'jsonfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleanup at the end.'

        self.jsonfile_prefix = jsonfile_prefix
        self.backend_args = backend_args

        self.metrics = metric if isinstance(metric, list) else [metric]

        self.eval_version = eval_version
        self.eval_detection_configs = config_factory(self.eval_version)

        self.dataset_meta = dataset_meta
        self.test_stage = test_stage
        self.line_thinkness = line_thinkness
        self.t_iou = t_iou
        self.rescale = rescale
        self.output_dir = output_dir
        if output_dir is not None:
            mkdir_or_exist(output_dir)

    # def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
    def process(self, data_batch: dict, data_samples) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.metrics) > 0:
            for data_sample in data_samples:
                result = dict()

                # 在不同分辨率stage预测的结果
                stages_pred_result = data_sample["pred_instances"]["stages_result"][0]
                # stages_pred_result = data_sample.pred_instances.stages_result[0]

                if self.rescale:
                    batch_input_shape = data_sample["ori_shape"]
                    eval_gt_lines = data_sample["ori_eval_gt_lines"]
                else:
                    eval_gt_lines = data_sample["eval_gt_lines"]
                    batch_input_shape = data_sample["batch_input_shape"]

                result['stages_pred_result'] = stages_pred_result
                result['batch_input_shape'] = batch_input_shape
                result['eval_gt_lines'] = eval_gt_lines

                sample_idx = data_sample['sample_idx']
                result['sample_idx'] = sample_idx
                self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.basename(self.jsonfile_prefix)}')
            return metric_dict

        result_list = results

        for metric in self.metrics:
            if metric == "line_pixel" or metric == "line_instance":
                f1_line_pixel, f1_line_instance = self.gbld_evaluate_line_pixel(
                    result_list, classes=classes, metric=metric, logger=logger)

                metric_dict["line_instance"] = f1_line_instance
                metric_dict["line_pixel"] = f1_line_pixel

        #


        # 保存测评log
        if self.output_dir is not None:
            with open(os.path.join(self.output_dir, 'log.txt'), 'w') as file:
                file.write(json.dumps(metric_dict))

        return metric_dict

    def gbld_evaluate_line_pixel(self,
                     result_list: list,
                     metric: str = 'line_pixel',
                     classes: Optional[List[str]] = None,
                     logger: Optional[MMLogger] = None) -> Dict[str, float]:

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
                precision_pixel, recall_pixel, fscore_pixel = self.get_line_map_F1(measure_gt_lines, measure_pred_lines,
                                                                 batch_input_shape, thickness=self.line_thinkness)
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
                precision = sum(accs)/len(accs)
                precision = round(precision, 2)
            if len(recalls) == 0:
                recall = -1
            else:
                recall = sum(recalls)/len(recalls)
                recall = round(recall, 2)

            eps = 0.001
            if len(accs) == 0 or len(recalls) ==0:
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
            iou = ap_pixel/all_pixel
            if iou > t_iou:
                acc_list[i] = True

        # recall
        for i, measure_gt_line in enumerate(measure_gt_lines):
            gt_line_heatmap = self.get_line_heatmap([measure_gt_line], heatmap_size, thickness=thickness)
            gt_line_heatmap = np.array(gt_line_heatmap, np.float32)

            ap_pixel = np.sum(pred_lines_heatmap * gt_line_heatmap)
            all_pixel = np.sum(gt_line_heatmap)
            iou = ap_pixel/all_pixel
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