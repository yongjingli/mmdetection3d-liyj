# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
import mmcv
import math
from mmengine.fileio import get

from mmcv.transforms import LoadImageFromFile
from mmcv.transforms import Resize, Pad, RandomFlip, RandomChoiceResize
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from mmcv.image.geometric import _scale_size

import mmengine
import numpy as np
import torch
from mmcv.transforms.base import BaseTransform

from mmengine.structures import InstanceData
from numpy import dtype

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.points import BasePoints

from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from .gbld2d_detr_utils import GBLDDetrInstanceLines


from albumentations import (
    RandomBrightnessContrast,
    OneOf,
    HueSaturationValue,
    Compose,
    Normalize,
    Blur,
    GaussianBlur,
    MedianBlur,
    MotionBlur
)

@TRANSFORMS.register_module()
class GgldDetrLoadLines(BaseTransform):
    def __init__(self, name="load_lines") -> None:
        self.name = name

    def transform(self, results: dict) -> dict:
        has_labels = False
        if results.get('eval_ann_info', None) is not None:
            if results['eval_ann_info'].get('gt_lines', None) is not None:
                eval_gt_lines = results['eval_ann_info']['gt_lines']
                # eval_gt_lines = []   # DEBUG when no gts
                results['eval_gt_lines'] = eval_gt_lines
                results['ori_eval_gt_lines'] = copy.deepcopy(eval_gt_lines)

                # results['gt_lines'] = eval_gt_lines
                # results['ori_gt_lines'] = copy.deepcopy(eval_gt_lines)
                has_labels = True

        if results.get('ann_info', None) is not None:
            if results['ann_info'].get('gt_lines', None) is not None:
                gt_lines = results['ann_info']['gt_lines']
                # gt_lines = []  # DEBUG when no gts

                results['gt_lines'] = gt_lines
                results['ori_gt_lines'] = copy.deepcopy(gt_lines)
                has_labels = True
        if not has_labels:
            print("no gt lines")
        # assert has_labels, "no lines label"
        return results

@TRANSFORMS.register_module()
# class SegLabelMapping(BaseTransform):
# 目前该方法实现还有些问题,主要是因为当line的超过图像范围进行截断时,会直接舍弃点,应该修改为插值补点
class GgldDetrRandomRotate(BaseTransform):
    def __init__(self, prob=0.5, angle_range=12) -> None:
        self.prob = prob
        self.angle_range = angle_range  # 设置随机旋转角度范围为[-angle_range, angle_range]

    def rotate_gt_lines(self, img, gt_lines):
        height, width = img.shape[:2]
        center = (width // 2, height // 2)  # 中心点坐标

        rotate_angle = np.random.randint(-self.angle_range, self.angle_range)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

        rotate_gt_lines = []
        # 得到当前图像线段的数量, 如果线段分段新增线段将赋予新的id
        max_index = 0
        max_line_idx = 0
        for gt_line in gt_lines:
            if gt_line["index"] > max_index:
                max_index = gt_line["index"]

            if gt_line["line_id"] > max_line_idx:
                max_line_idx = gt_line["line_id"]

        for gt_line in gt_lines:
            points = gt_line['points']
            rotated_points = cv2.transform(points.reshape(-1, 1, 2), rotation_matrix)
            rotated_points = rotated_points.squeeze()

            mask_x = np.bitwise_and(rotated_points[:, 0] >= 0, rotated_points[:, 0] < width)
            mask_y = np.bitwise_and(rotated_points[:, 1] >= 0, rotated_points[:, 1] < height)

            mask = np.bitwise_and(mask_x, mask_y)

            if "points_type" in gt_line:
                points_type = gt_line["points_type"]
            else:
                points_type = None

            if "points_visible" in gt_line:
                points_visible = gt_line["points_visible"]
            else:
                points_visible = None

            if "points_hanging" in gt_line:
                points_hanging = gt_line["points_hanging"]
            else:
                points_hanging = None

            if "points_covered" in gt_line:
                points_covered = gt_line["points_covered"]
            else:
                points_covered = None

            # 在这里判断线是否需要分段成为不同的线，具有不同的id, 否则对聚类有影响
            # 将中间mask分段的线段分为不同的线段
            line_indexs = []
            start_indx = None
            for i, _mask in enumerate(mask):
                if _mask and start_indx is None:
                    start_indx = i
                elif (not _mask) and start_indx is not None:
                    end_indx = i - 1
                    line_indexs.append((start_indx, end_indx))
                    start_indx = None

                if i == len(mask) - 1:
                    end_indx = i
                    if start_indx is not None:
                        line_indexs.append((start_indx, end_indx))

            for j, line_index in enumerate(line_indexs):
                crop_gt_line = copy.deepcopy(gt_line)
                # points = points[mask]
                id_0, id_1 = line_index
                id_1 = id_1 + 1
                splint_points = rotated_points[id_0:id_1]

                if len(splint_points) < 2:
                    continue

                crop_gt_line["points"] = splint_points

                # 最后一个代表的是下个线段的属性
                if points_type is not None:
                    split_points_type = np.concatenate([points_type[id_0:id_1], points_type[id_1 - 1:id_1]], axis=0)
                    crop_gt_line["points_type"] = split_points_type

                if points_visible is not None:
                    split_points_visible = np.concatenate([points_visible[id_0:id_1], points_visible[id_1 - 1:id_1]],
                                                          axis=0)
                    crop_gt_line["points_visible"] = split_points_visible

                if points_hanging is not None:
                    split_points_hanging = np.concatenate([points_hanging[id_0:id_1], points_hanging[id_1 - 1:id_1]],
                                                          axis=0)
                    crop_gt_line["points_hanging"] = split_points_hanging

                if points_covered is not None:
                    split_points_covered = np.concatenate([points_covered[id_0:id_1], points_covered[id_1 - 1:id_1]],
                                                          axis=0)
                    crop_gt_line["points_covered"] = split_points_covered

                # 代表新增的线
                if j > 0:
                    max_index = max_index + 1
                    max_line_idx = max_line_idx + 1
                    crop_gt_line["index"] = max_index
                    crop_gt_line["line_id"] = max_line_idx

                rotate_gt_lines.append(crop_gt_line)
        return rotated_image, rotate_gt_lines

    def transform(self, results: dict) -> dict:
        # results['img'] = results['img'][crop_y1:crop_y2, crop_x1:crop_x2, ...]
        # results["img_shape"] = results['img'].shape[:2]

        img = results['img']

        if np.random.rand() < self.prob:
            # flip lane
            if results.get('eval_gt_lines', None) is not None:
                eval_gt_lines = results['eval_gt_lines']
                img, eval_gt_lines = self.rotate_gt_lines(img, eval_gt_lines)

                results['img'] = img
                results["img_shape"] = img.shape[:2]
                results['eval_gt_lines'] = eval_gt_lines

            if results.get('gt_lines', None) is not None:
                gt_lines = results['gt_lines']
                img, gt_lines = self.rotate_gt_lines(img, gt_lines)

                results['img'] = img
                results["img_shape"] = img.shape[:2]
                results['gt_lines'] = gt_lines

        return results

@TRANSFORMS.register_module()
# class SegLabelMapping(BaseTransform):
# 目前该方法实现还有些问题,主要是因为当line的超过图像范围进行截断时,会直接舍弃点,应该修改为插值补点
class GgldDetrRandomCrop(BaseTransform):
    def __init__(self, prob=0.5, max_margin_scale=0.2, keep_ratio=True) -> None:
        self.prob = prob
        self.max_margin_scale = max_margin_scale
        self.keep_ratio = keep_ratio

    def get_img_crop_size(self, img_shape, max_margin_scale):
        # max_margin_scale 为两个方向上最大的margin,不能超过0.5
        assert max_margin_scale < 0.5, "max_margin_scale < 0.5"

        img_h, img_w = img_shape
        left_margin = int(np.random.uniform(low=0.0, high=max_margin_scale) * img_w)
        right_margin = int(np.random.uniform(low=0.0, high=max_margin_scale) * img_w)

        top_margin = int(np.random.uniform(low=0.0, high=max_margin_scale) * img_h)
        bottom_margin = int(np.random.uniform(low=0.0, high=max_margin_scale) * img_h)

        crop_img_w = img_w - left_margin - right_margin
        crop_img_h = img_h - top_margin - bottom_margin

        crop_x1, crop_x2 = left_margin, left_margin + crop_img_w
        crop_y1, crop_y2 = top_margin, top_margin + crop_img_h

        if self.keep_ratio:
            # 保持原来的长宽比
            h_scale = crop_img_h / img_h
            w_scale = crop_img_w / img_w
            if h_scale > w_scale:
                # 那么说明h偏大了, 对crop_img_h进行缩小
                crop_img_h = int(img_h * w_scale)

            else:
                crop_img_w = int(img_w * h_scale)

            crop_x1, crop_x2 = left_margin, left_margin + crop_img_w
            crop_y1, crop_y2 = top_margin, top_margin + crop_img_h
        return crop_x1, crop_x2, crop_y1, crop_y2

    def crop_gt_lines(self, gt_lines, crop_x1, crop_x2, crop_y1, crop_y2):
        crop_gt_lines = []
        # 得到当前图像线段的数量, 如果线段分段新增线段将赋予新的id
        max_index = 0
        max_line_idx = 0
        for gt_line in gt_lines:
            if gt_line["index"] > max_index:
                max_index = gt_line["index"]

            if gt_line["line_id"] > max_line_idx:
                max_line_idx = gt_line["line_id"]

        for gt_line in gt_lines:
            points = gt_line["points"]

            mask_x = np.bitwise_and(points[:, 0] > (crop_x1 - 1), points[:, 0] < crop_x2)
            mask_y = np.bitwise_and(points[:, 1] > (crop_y1 - 1), points[:, 1] < crop_y2)

            mask = np.bitwise_and(mask_x, mask_y)

            if "points_type" in gt_line:
                points_type = gt_line["points_type"]
            else:
                points_type = None

            if "points_visible" in gt_line:
                points_visible = gt_line["points_visible"]
            else:
                points_visible = None

            if "points_hanging" in gt_line:
                points_hanging = gt_line["points_hanging"]
            else:
                points_hanging = None

            if "points_covered" in gt_line:
                points_covered = gt_line["points_covered"]
            else:
                points_covered = None

            # 在这里判断线是否需要分段成为不同的线，具有不同的id, 否则对聚类有影响
            # 将中间mask分段的线段分为不同的线段
            line_indexs = []
            start_indx = None
            for i, _mask in enumerate(mask):
                if _mask and start_indx is None:
                    start_indx = i
                elif (not _mask) and start_indx is not None:
                    end_indx = i - 1
                    line_indexs.append((start_indx, end_indx))
                    start_indx = None

                if i == len(mask) - 1:
                    end_indx = i
                    if start_indx is not None:
                        line_indexs.append((start_indx, end_indx))

            for j, line_index in enumerate(line_indexs):
                crop_gt_line = copy.deepcopy(gt_line)
                # points = points[mask]
                id_0, id_1 = line_index
                id_1 = id_1 + 1
                splint_points = points[id_0:id_1]

                if len(splint_points) < 2:
                    continue

                splint_points[:, 0] = splint_points[:, 0] - crop_x1
                splint_points[:, 1] = splint_points[:, 1] - crop_y1
                crop_gt_line["points"] = splint_points

                # 最后一个代表的是下个线段的属性
                if points_type is not None:
                    split_points_type = np.concatenate([points_type[id_0:id_1], points_type[id_1 - 1:id_1]], axis=0)
                    crop_gt_line["points_type"] = split_points_type

                if points_visible is not None:
                    split_points_visible = np.concatenate([points_visible[id_0:id_1], points_visible[id_1 - 1:id_1]], axis=0)
                    crop_gt_line["points_visible"] = split_points_visible

                if points_hanging is not None:
                    split_points_hanging = np.concatenate([points_hanging[id_0:id_1], points_hanging[id_1 - 1:id_1]], axis=0)
                    crop_gt_line["points_hanging"] = split_points_hanging

                if points_covered is not None:
                    split_points_covered = np.concatenate([points_covered[id_0:id_1], points_covered[id_1 - 1:id_1]], axis=0)
                    crop_gt_line["points_covered"] = split_points_covered

                # 代表新增的线
                if j > 0:
                    max_index = max_index + 1
                    max_line_idx = max_line_idx + 1
                    crop_gt_line["index"] = max_index
                    crop_gt_line["line_id"] = max_line_idx

                crop_gt_lines.append(crop_gt_line)

        return crop_gt_lines

    def transform(self, results: dict) -> dict:
        if np.random.rand() < self.prob:
            img_shape = results["img_shape"]

            crop_x1, crop_x2, crop_y1, crop_y2 = self.get_img_crop_size(img_shape, self.max_margin_scale)

            results['img_crop_coord'] = [crop_x1, crop_x2, crop_y1, crop_y2]
            results['img'] = results['img'][crop_y1:crop_y2, crop_x1:crop_x2, ...]
            results["img_shape"] = results['img'].shape[:2]

            # flip lane
            if results.get('eval_gt_lines', None) is not None:
                eval_gt_lines = results['eval_gt_lines']
                eval_gt_lines = self.crop_gt_lines(eval_gt_lines, crop_x1, crop_x2, crop_y1, crop_y2)
                results['eval_gt_lines'] = eval_gt_lines

            if results.get('gt_lines', None) is not None:
                gt_lines = results['gt_lines']
                gt_lines = self.crop_gt_lines(gt_lines, crop_x1, crop_x2, crop_y1, crop_y2)
                results['gt_lines'] = gt_lines

        return results

@TRANSFORMS.register_module()
# class SegLabelMapping(BaseTransform):
class GgldDetrRandomFlip(RandomFlip):
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map and
        keypoints."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'] = self._flip_keypoints(
                results['gt_keypoints'], img_shape, results['flip_direction'])

        # flip seg map
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = self._flip_seg_map(
                results['gt_seg_map'], direction=results['flip_direction'])
            results['swap_seg_labels'] = self.swap_seg_labels

        # flip lane
        if results.get('eval_ann_info', None) is not None:
            if results['eval_ann_info'].get('gt_lines', None) is not None:
                gt_lines = self._flip_lines(results['eval_ann_info']['gt_lines'],
                                            img_shape,
                                            direction=results['flip_direction'])

                results['eval_ann_info']['gt_lines'] = gt_lines
                results['eval_gt_lines'] = gt_lines

        if results.get('gt_lines', None) is not None:
            gt_lines = self._flip_lines(results['gt_lines'],
                img_shape,
                direction=results['flip_direction'])

            results['gt_lines'] = gt_lines

    def _flip_lines(self, lines: np.ndarray, img_shape: Tuple[int, int],
                   direction: str) -> np.ndarray:

        flipped = lines.copy()
        h, w = img_shape
        for i in range(len(lines)):
            line = lines[i]["points"]
            flip_line = flipped[i]["points"]

            if direction == 'horizontal':
                flip_line[:, 0] = w - line[:, 0]
                flip_line[:, :] = flip_line[::-1, :]

                if "points_type" in lines[i]:
                    lines[i]["points_type"] = lines[i]["points_type"][::-1, :]

                if "points_visible" in lines[i]:
                    lines[i]["points_visible"] = lines[i]["points_visible"][::-1, :]

                if "points_hanging" in lines[i]:
                    lines[i]["points_hanging"] = lines[i]["points_hanging"][::-1, :]

                if "points_covered" in lines[i]:
                    lines[i]["points_covered"] = lines[i]["points_covered"][::-1, :]

            # elif direction == 'vertical':
            #     flip_line[:, 1] = h - line[:, 1]
            #
            # elif direction == 'diagonal':
            #     flip_line[:, 0] = w - line[:, 0]
            #     flip_line[:, 1] = h - line[:, 1]
            else:
                raise ValueError(
                    f"Flipping direction must be 'horizontal', 'vertical', \
                      or 'diagonal', but got '{direction}'")
        return flipped


@TRANSFORMS.register_module()
# class SegLabelMapping(BaseTransform):
class GgldDetrColor(BaseTransform):
    def __init__(self, prob=0.5) -> None:

        self.prob = prob
        self.img_augs = self._get_augs()

    def transform(self, results: dict) -> dict:
        if np.random.rand() < self.prob:
            results['img'] = self.img_augs(image=results['img'])['image']
        return results

    def _get_augs(self):
        aug = Compose(
            [
                OneOf(
                    [
                        HueSaturationValue(hue_shift_limit=10,
                                           sat_shift_limit=10,
                                           val_shift_limit=10,
                                           p=0.5),
                        RandomBrightnessContrast(brightness_limit=0.2,
                                                 contrast_limit=0.2,
                                                 p=0.5)
                    ]
                ),
                OneOf(
                    [
                        Blur(blur_limit=3, p=0.5),
                        GaussianBlur(blur_limit=3, p=0.5),
                        MedianBlur(blur_limit=3, p=0.5),
                        MotionBlur(p=0.5)
                    ]
                ),

            ],
            p=1.0)
        return aug


@TRANSFORMS.register_module()
# class SegLabelMapping(BaseTransform):
class GgldDetrResize(Resize):
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1],
                                           self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)

        # add resize line
        self._resize_lines(results)
        self._resize_lines_eval(results)
        return results

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

    def _resize_lines(self, results: dict) -> None:
        if results.get('gt_lines', None) is not None:
            scale_factor = results['scale_factor']

            gt_lines = results["gt_lines"]
            for gt_line in gt_lines:
                points = gt_line["points"]
                points[:, 0] = points[:, 0] * scale_factor[0]
                points[:, 1] = points[:, 1] * scale_factor[1]

                if self.clip_object_border:
                    points[:, 1] = np.clip(points[:, 1], 0, results['img_shape'][0])
                    points[:, 0] = np.clip(points[:, 0], 0, results['img_shape'][1])

            results['gt_lines'] = gt_lines

    def _resize_lines_eval(self, results: dict) -> None:
        if results.get('eval_gt_lines', None) is not None:
            scale_factor = results['scale_factor']

            eval_gt_lines = results["eval_gt_lines"]
            for gt_line in eval_gt_lines:
                points = gt_line["points"]
                points[:, 0] = points[:, 0] * scale_factor[0]
                points[:, 1] = points[:, 1] * scale_factor[1]

                if self.clip_object_border:
                    points[:, 1] = np.clip(points[:, 1], 0, results['img_shape'][0])
                    points[:, 0] = np.clip(points[:, 0], 0, results['img_shape'][1])

            results['eval_gt_lines'] = eval_gt_lines


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        print(data)
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class GgldDetrLineMapsGenerate(BaseTransform):
    # 用来产生gbld所需的训练信息
    def __init__(self,
                 gt_down_scales: Optional[Tuple[int, int]] = None,
                 num_classes: int = 1,
                 ) -> None:
        self.gt_down_scales = gt_down_scales
        self.num_classes = num_classes

    def _gen_line_map(self, gt_lines, map_size):
        line_map = np.zeros(map_size, dtype=np.uint8)
        line_map_id = np.zeros(map_size, dtype=np.uint8)
        line_map_cls = np.zeros(map_size, dtype=np.uint8)
        for gt_line in gt_lines:
            label = gt_line['label']
            line_points = gt_line['points']
            index = gt_line['index'] + 1     # 序号从0开始的
            line_id = gt_line['line_id'] + 1
            category_id = gt_line['category_id'] + 1

            pre_point = line_points[0]
            for cur_point in line_points[1:]:
                x1, y1 = round(pre_point[0]), round(pre_point[1])
                x2, y2 = round(cur_point[0]), round(cur_point[1])
                cv2.line(line_map, (x1, y1), (x2, y2), (index,))
                cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,))

                # cls_value = self.class_dic[label] + 1
                cv2.line(line_map_cls, (x1, y1), (x2, y2), (category_id,))
                pre_point = cur_point

        return line_map, line_map_id, line_map_cls

    def _gen_gt_line_maps(self, line_map, line_map_id, line_map_cls, grid_size):
        line_map_h, line_map_w = line_map.shape
        gt_map_h, gt_map_w = math.ceil(line_map_h / grid_size), math.ceil(
            line_map_w / grid_size
        )
        gt_confidence = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_offset_x = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_offset_y = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_line_index = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_line_id = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)

        gt_line_cls = np.zeros((self.num_classes, gt_map_h, gt_map_w), dtype=np.float32)

        for y in range(0, gt_map_h):
            for x in range(0, gt_map_w):
                start_x, end_x = x * grid_size, (x + 1) * grid_size
                end_x = end_x if end_x < line_map_w else line_map_w
                start_y, end_y = y * grid_size, (y + 1) * grid_size
                end_y = end_y if end_y < line_map_h else line_map_h
                grid = line_map[start_y:end_y, start_x:end_x]

                grid_id = line_map_id[start_y:end_y, start_x:end_x]
                grid_cls = line_map_cls[start_y:end_y, start_x:end_x]

                confidence = 1 if np.any(grid) else 0
                gt_confidence[0, y, x] = confidence
                if confidence == 1:
                    ys, xs = np.nonzero(grid)
                    offset_y, offset_x = sorted(
                        zip(ys, xs), key=lambda p: (p[0], -p[1]), reverse=True
                    )[0]
                    gt_offset_x[0, y, x] = offset_x / (grid_size - 1)
                    gt_offset_y[0, y, x] = offset_y / (grid_size - 1)
                    gt_line_index[0, y, x] = grid[offset_y, offset_x]
                    gt_line_id[0, y, x] = grid_id[offset_y, offset_x]

                    cls = grid_cls[offset_y, offset_x]
                    if cls > 0:
                        cls_indx = int(cls - 1)
                        gt_line_cls[cls_indx, y, x] = 1

        foreground_mask = gt_confidence.astype(np.uint8)

        # expand foreground mask
        kernel = np.ones((3, 3), np.uint8)
        foreground_expand_mask = cv2.dilate(foreground_mask[0], kernel)
        foreground_expand_mask = np.expand_dims(foreground_expand_mask.astype(np.uint8), axis=0)

        ignore_mask = np.zeros((1, gt_map_h, gt_map_w), dtype=np.uint8)
        # top, bottom = self.line_map_range
        ignore_mask[0, 0:-1, :] = 1     # 手动设置有效范围

        return gt_confidence, gt_offset_x, gt_offset_y, \
               gt_line_index, ignore_mask, foreground_mask, \
               gt_line_id, gt_line_cls, foreground_expand_mask

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

    def filter_near_same_points(self, line_points):
        mask = [True] * len(line_points)
        pre_point = line_points[0]
        for i, cur_point in enumerate(line_points[1:]):
            if np.all(cur_point == pre_point):
                mask[i + 1] = False
            else:
                pre_point = cur_point

        line_points = line_points[mask]
        return line_points

    def get_orient_sin_cos(self, orient):
        # 将MEBOW的格式转为[-pi, pi], 将角度的范围转换为kitti数据的格式
        # 以x为0度方向(水平向右), 范围：-pi~pi，顺时针为正，逆时针为负
        if orient > 180:
            orient = orient - 360

        orient = -orient      # 将逆时针方向修改为顺时针方向
        orient = orient - 90  # 将垂直向上为0度改成以水平向右为0度
        if orient > 180:      # 将范围限制到-pi~pi的范围内
            orient = orient - 360
        if orient < -180:
            orient = orient + 360

        orient = orient / 180 * math.pi

        orient_sin = math.sin(orient)
        orient_cos = math.cos(orient)
        return orient_sin, orient_cos

    def _gen_gt_orient_maps(self, gt_lines, map_size, down_scale):
        down_scale_map_size = (int(map_size[0]/down_scale), int(map_size[1]/down_scale))
        orient_map_mask = np.zeros(down_scale_map_size, dtype=np.uint8)
        orient_map_sin = np.zeros(down_scale_map_size, dtype=np.float32)
        orient_map_cos = np.zeros(down_scale_map_size, dtype=np.float32)

        for gt_line in gt_lines:
            label = gt_line['label']
            line_points = gt_line['points']
            line_points = line_points/down_scale

            # 过滤重复的点
            line_points = self.filter_near_same_points(line_points)

            index = gt_line['index'] + 1     # 序号从0开始的
            line_id = gt_line['line_id'] + 1
            category_id = gt_line['category_id'] + 1

            pre_point = line_points[0]
            for i, cur_point in enumerate(line_points[1:]):
                orient = self.cal_points_orient(pre_point, cur_point)
                c_x, c_y = int(pre_point[0]), int(pre_point[1])
                c_x = min(max(0, c_x), down_scale_map_size[1]-1)
                c_y = min(max(0, c_y), down_scale_map_size[0]-1)

                if orient != -1:
                    # print(c_x, c_y, pre_point[0], pre_point[1], gt_line['points'][i+1, 0], gt_line['points'][i+1, 1])
                    orient_map_mask[c_y, c_x] = 1

                    orient_sin, orient_cos = self.get_orient_sin_cos(orient)

                    # 转为三角函数 sin、cos的形式
                    orient_map_sin[c_y, c_x] = orient_sin
                    orient_map_cos[c_y, c_x] = orient_cos

                pre_point = cur_point

        orient_map_mask = np.expand_dims(orient_map_mask, axis=0)
        orient_map_sin = np.expand_dims(orient_map_sin, axis=0)
        orient_map_cos = np.expand_dims(orient_map_cos, axis=0)

        return orient_map_mask, orient_map_sin, orient_map_cos

    def transform(self, results: dict) -> dict:
        map_size = results["img_shape"]
        if "gt_lines" in results:
            gt_lines = results["gt_lines"]
        else:
            gt_lines = results["eval_gt_lines"]

        # 对gt_lines进行排序, 按照点数的数量从小到大排序，最后再绘制长的线，保证在降采样的分辨率上，如果有重叠的位置，保证长的优先
        gt_lines = sorted(gt_lines, key=lambda x: x["points"].shape[0])

        gt_line_maps_stages = []
        line_map, line_map_id, line_map_cls = self._gen_line_map(gt_lines, map_size)

        for gt_down_scale in self.gt_down_scales:
            gt_line_maps = self._gen_gt_line_maps(line_map, line_map_id, line_map_cls, gt_down_scale)

            gt_confidence = torch.from_numpy(gt_line_maps[0])
            gt_offset_x = torch.from_numpy(gt_line_maps[1])
            gt_offset_y = torch.from_numpy(gt_line_maps[2])
            gt_line_index = torch.from_numpy(gt_line_maps[3])
            ignore_mask = torch.from_numpy(gt_line_maps[4])
            foreground_mask = torch.from_numpy(gt_line_maps[5])
            gt_line_id = torch.from_numpy(gt_line_maps[6])
            gt_line_cls = torch.from_numpy(gt_line_maps[7])
            foreground_expand_mask = torch.from_numpy(gt_line_maps[8])

            orient_map_mask, orient_map_sin, orient_map_cos = self._gen_gt_orient_maps(gt_lines, map_size, gt_down_scale)
            orient_map_mask = torch.from_numpy(orient_map_mask)
            orient_map_sin = torch.from_numpy(orient_map_sin)
            orient_map_cos = torch.from_numpy(orient_map_cos)


            gt_line_maps = {
                            "gt_confidence": gt_confidence,
                            "gt_offset_x": gt_offset_x,
                            "gt_offset_y": gt_offset_y,
                            "gt_line_index": gt_line_index,
                            "ignore_mask": ignore_mask,
                            "foreground_mask": foreground_mask,
                            "gt_line_id": gt_line_id,
                            "gt_line_cls": gt_line_cls,
                            "foreground_expand_mask": foreground_expand_mask,
                            "orient_map_mask": orient_map_mask,
                            "orient_map_sin": orient_map_sin,
                            "orient_map_cos": orient_map_cos,
                            }

            gt_line_maps_stages.append(gt_line_maps)

        results['gt_line_maps_stages'] = gt_line_maps_stages
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_down_scales={self.gt_down_scales}, '
        return repr_str


@TRANSFORMS.register_module()
# 相对上个版本, 采用分段产生类别、可见属性、悬空属性和被草遮挡属性等
class GgldDetrLineMapsGenerateV2(BaseTransform):
    # 用来产生gbld所需的训练信息
    def __init__(self,
                 gt_down_scales: Optional[Tuple[int, int]] = None,
                 num_classes: int = 1,
                 filter_small_line=False,
                 filter_length=10,
                 ) -> None:
        self.gt_down_scales = gt_down_scales
        self.num_classes = num_classes

        self.filter_small_line = filter_small_line
        self.filter_length = filter_length

    def _gen_line_map(self, gt_lines, map_size):
        line_map = np.zeros(map_size, dtype=np.uint8)
        line_map_id = np.zeros(map_size, dtype=np.uint8)
        line_map_cls = np.zeros(map_size, dtype=np.uint8)

        line_map_visible = np.zeros(map_size, dtype=np.uint8)
        line_map_hanging = np.zeros(map_size, dtype=np.uint8)
        line_map_covered = np.zeros(map_size, dtype=np.uint8)

        thickness = 1
        if map_size[1] > 240:
            thickness = 2

        if map_size[1] > 480:
            thickness = 4

        for gt_line in gt_lines:
            label = gt_line['label']
            line_points = gt_line['points']
            index = gt_line['index'] + 1     # 序号从0开始的
            line_id = gt_line['line_id'] + 1
            category_id = gt_line['category_id'] + 1

            line_points_type = gt_line['points_type']
            line_points_visible = gt_line['points_visible']
            line_points_hanging = gt_line['points_hanging']
            line_points_covered = gt_line['points_covered']

            pre_point = line_points[0]
            pre_point_type = line_points_type[0]
            pre_point_visible = line_points_visible[0]
            pre_point_hanging = line_points_hanging[0]
            pre_point_covered = line_points_covered[0]

            for cur_point, cur_point_type, cur_point_visible, cur_point_hanging, cur_point_covered in \
                    zip(line_points[1:], line_points_type[1:], line_points_visible[1:],
                        line_points_hanging[1:], line_points_covered[1:]):

                x1, y1 = round(pre_point[0]), round(pre_point[1])
                x2, y2 = round(cur_point[0]), round(cur_point[1])
                cv2.line(line_map, (x1, y1), (x2, y2), (index,), thickness=thickness)
                cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,), thickness=thickness)

                # cls_value = self.class_dic[label] + 1
                # cv2.line(line_map_cls, (x1, y1), (x2, y2), (category_id,))

                # 以起点的属性为准
                # 修改为分段类别
                cv2.line(line_map_cls, (x1, y1), (x2, y2), (int(pre_point_type[0]) + 1,), thickness=thickness)

                # 新增可见、悬空和被草遮挡的属性预测
                if pre_point_visible[0] > 0:
                    cv2.line(line_map_visible, (x1, y1), (x2, y2), (1,), thickness=thickness)

                if pre_point_hanging[0] > 0:
                    cv2.line(line_map_hanging, (x1, y1), (x2, y2), (1,), thickness=thickness)

                if pre_point_covered[0] > 0:
                    cv2.line(line_map_covered, (x1, y1), (x2, y2), (1,), thickness=thickness)

                pre_point = cur_point
                pre_point_type = cur_point_type
                pre_point_visible = cur_point_visible
                pre_point_hanging = cur_point_hanging
                pre_point_covered = cur_point_covered

        return line_map, line_map_id, line_map_cls, line_map_visible, line_map_hanging, line_map_covered

    def _gen_gt_line_maps(self, line_map, line_map_id, line_map_cls,
                          line_map_visible, line_map_hanging, line_map_covered, grid_size):
        line_map_h, line_map_w = line_map.shape
        gt_map_h, gt_map_w = math.ceil(line_map_h / grid_size), math.ceil(
            line_map_w / grid_size
        )
        gt_confidence = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_offset_x = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_offset_y = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_line_index = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_line_id = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)

        gt_line_cls = np.zeros((self.num_classes, gt_map_h, gt_map_w), dtype=np.float32)

        # 新增可见、悬空、被草遮挡的confidence预测
        gt_confidence_visible = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_confidence_hanging = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_confidence_covered = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)

        for y in range(0, gt_map_h):
            for x in range(0, gt_map_w):
                start_x, end_x = x * grid_size, (x + 1) * grid_size
                end_x = end_x if end_x < line_map_w else line_map_w
                start_y, end_y = y * grid_size, (y + 1) * grid_size
                end_y = end_y if end_y < line_map_h else line_map_h
                grid = line_map[start_y:end_y, start_x:end_x]

                grid_id = line_map_id[start_y:end_y, start_x:end_x]
                grid_cls = line_map_cls[start_y:end_y, start_x:end_x]

                confidence = 1 if np.any(grid) else 0
                gt_confidence[0, y, x] = confidence
                if confidence == 1:
                    ys, xs = np.nonzero(grid)
                    offset_y, offset_x = sorted(
                        zip(ys, xs), key=lambda p: (p[0], -p[1]), reverse=True
                    )[0]
                    if grid_size != 1:
                        gt_offset_x[0, y, x] = offset_x / (grid_size - 1)
                        gt_offset_y[0, y, x] = offset_y / (grid_size - 1)

                        # 设置成
                        # gt_confidence[0, y, x] = confidence - min(offset_x, offset_y)/grid_size

                    gt_line_index[0, y, x] = grid[offset_y, offset_x]
                    gt_line_id[0, y, x] = grid_id[offset_y, offset_x]

                    cls = grid_cls[offset_y, offset_x]
                    if cls > 0:
                        cls_indx = int(cls - 1)
                        gt_line_cls[cls_indx, y, x] = 1

                # gt_confidence_visible
                if np.any(line_map_visible[start_y:end_y, start_x:end_x]):
                    gt_confidence_visible[0, y, x] = 1

                # gt_confidence_hanging
                if np.any(line_map_hanging[start_y:end_y, start_x:end_x]):
                    gt_confidence_hanging[0, y, x] = 1

                # gt_confidence_covered
                if np.any(line_map_covered[start_y:end_y, start_x:end_x]):
                    gt_confidence_covered[0, y, x] = 1

        foreground_mask = gt_confidence.astype(np.uint8)

        # expand foreground mask
        kernel = np.ones((3, 3), np.uint8)
        foreground_expand_mask = cv2.dilate(foreground_mask[0], kernel)
        foreground_expand_mask = np.expand_dims(foreground_expand_mask.astype(np.uint8), axis=0)

        ignore_mask = np.zeros((1, gt_map_h, gt_map_w), dtype=np.uint8)
        # top, bottom = self.line_map_range
        ignore_mask[0, 0:-1, :] = 1     # 手动设置有效范围

        return gt_confidence, gt_offset_x, gt_offset_y, \
               gt_line_index, ignore_mask, foreground_mask, \
               gt_line_id, gt_line_cls, foreground_expand_mask, \
               gt_confidence_visible, gt_confidence_hanging, gt_confidence_covered

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

    def filter_near_same_points(self, line_points):
        mask = [True] * len(line_points)
        pre_point = line_points[0]
        for i, cur_point in enumerate(line_points[1:]):
            if np.all(cur_point == pre_point):
                mask[i + 1] = False
            else:
                pre_point = cur_point

        line_points = line_points[mask]
        return line_points

    def get_orient_sin_cos(self, orient):
        # 将MEBOW的格式转为[-pi, pi], 将角度的范围转换为kitti数据的格式
        # 以x为0度方向(水平向右), 范围：-pi~pi，顺时针为正，逆时针为负
        if orient > 180:
            orient = orient - 360

        orient = -orient      # 将逆时针方向修改为顺时针方向
        orient = orient - 90  # 将垂直向上为0度改成以水平向右为0度
        if orient > 180:      # 将范围限制到-pi~pi的范围内
            orient = orient - 360
        if orient < -180:
            orient = orient + 360

        orient = orient / 180 * math.pi

        orient_sin = math.sin(orient)
        orient_cos = math.cos(orient)
        return orient_sin, orient_cos

    def _gen_gt_orient_maps(self, gt_lines, map_size, down_scale):
        down_scale_map_size = (int(map_size[0]/down_scale), int(map_size[1]/down_scale))
        orient_map_mask = np.zeros(down_scale_map_size, dtype=np.uint8)
        orient_map_sin = np.zeros(down_scale_map_size, dtype=np.float32)
        orient_map_cos = np.zeros(down_scale_map_size, dtype=np.float32)

        for gt_line in gt_lines:
            label = gt_line['label']
            line_points = gt_line['points']
            line_points = line_points/down_scale

            # 过滤重复的点
            line_points = self.filter_near_same_points(line_points)

            index = gt_line['index'] + 1     # 序号从0开始的
            line_id = gt_line['line_id'] + 1
            category_id = gt_line['category_id'] + 1

            pre_point = line_points[0]
            for i, cur_point in enumerate(line_points[1:]):
                orient = self.cal_points_orient(pre_point, cur_point)
                c_x, c_y = int(pre_point[0]), int(pre_point[1])
                c_x = min(max(0, c_x), down_scale_map_size[1]-1)
                c_y = min(max(0, c_y), down_scale_map_size[0]-1)

                if orient != -1:
                    # print(c_x, c_y, pre_point[0], pre_point[1], gt_line['points'][i+1, 0], gt_line['points'][i+1, 1])
                    orient_map_mask[c_y, c_x] = 1

                    orient_sin, orient_cos = self.get_orient_sin_cos(orient)

                    # 转为三角函数 sin、cos的形式
                    orient_map_sin[c_y, c_x] = orient_sin
                    orient_map_cos[c_y, c_x] = orient_cos

                pre_point = cur_point

        orient_map_mask = np.expand_dims(orient_map_mask, axis=0)
        orient_map_sin = np.expand_dims(orient_map_sin, axis=0)
        orient_map_cos = np.expand_dims(orient_map_cos, axis=0)

        return orient_map_mask, orient_map_sin, orient_map_cos
    #
    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def _gen_gt_point_maps(self, gt_lines, map_size, down_scale):
        down_scale_map_size = (int(map_size[0]/down_scale), int(map_size[1]/down_scale))
        point_map_mask = np.zeros(down_scale_map_size, dtype=np.uint8)
        point_map_confidence = np.zeros(down_scale_map_size, dtype=np.float32)
        point_map_index = np.zeros(down_scale_map_size, dtype=np.float32)

        map_height, map_width = down_scale_map_size[0], down_scale_map_size[1]

        for gt_line in gt_lines:
            label = gt_line['label']
            line_points = gt_line['points']
            line_points = line_points/down_scale

            # 过滤重复的点
            line_points = self.filter_near_same_points(line_points)

            index = gt_line['index'] + 1     # 序号从0开始的
            start_point = line_points[0]
            end_point = line_points[-1]

            middle_point = line_points[len(line_points)//2]

            # dist_h = abs(start_point[1] - end_point[1])
            # dist_w = abs(start_point[0] - end_point[0])

            dist_h = abs(max(line_points[:, 1]) - min(line_points[:, 1]))
            dist_w = abs(max(line_points[:, 0]) - min(line_points[:, 0]))
            radius = self.gaussian_radius((math.ceil(dist_h), math.ceil(dist_w)))
            radius = max(0, int(radius))

            # for _point in [start_point, end_point, middle_point]:
            for _point in [start_point, end_point]:
                diameter = 2 * radius + 1
                gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
                # import matplotlib.pyplot as plt
                # plt.imshow(gaussian)
                # plt.show()
                x, y = int(_point[0]), int(_point[1])

                left, right = min(x, radius), min(map_width - x, radius + 1)
                top, bottom = min(y, radius), min(map_height - y, radius + 1)

                k = 1
                region_heatmap = point_map_confidence[y - top:y + bottom, x - left:x + right]
                masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
                if min(masked_gaussian.shape) > 0 and min(region_heatmap.shape) > 0:  # TODO debug
                    np.maximum(region_heatmap, masked_gaussian * k, out=region_heatmap)

                region_index = point_map_index[y - top:y + bottom, x - left:x + right]
                region_mask = point_map_mask[y - top:y + bottom, x - left:x + right]

                if min(masked_gaussian.shape) > 0 and min(region_heatmap.shape) > 0:
                    region_index[region_heatmap > 0.2] = index
                    region_mask[region_heatmap > 0.2] = 1

                # point_map_index[y - top:y + bottom, x - left:x + right] = index
                # point_map_mask[y - top:y + bottom, x - left:x + right] = 1

                # c_x, c_y = int(_point[0]), int(_point[1])
                # c_x = min(max(0, c_x), down_scale_map_size[1]-1)
                # c_y = min(max(0, c_y), down_scale_map_size[0]-1)
                # point_map_mask[c_y, c_x] = 1
                # point_map_confidence[c_y, c_x] = 1
                # point_map_index[c_y, c_x] = index

        point_map_mask = np.expand_dims(point_map_mask, axis=0)
        point_map_confidence = np.expand_dims(point_map_confidence, axis=0)
        point_map_index = np.expand_dims(point_map_index, axis=0)

        return point_map_mask, point_map_confidence, point_map_index

    def filter_small_lines(self, gt_lines):
        filter_gt_lines = []
        for gt_line in gt_lines:
            line_points = gt_line["points"]
            points_0 = line_points[:-1]
            points_1 = line_points[1:]

            points_dist = np.sqrt((points_0[:, 0] - points_1[:, 0]) ** 2
                                + (points_0[:, 1] - points_1[:, 1]) ** 2)

            line_dist = np.sum(points_dist)
            if line_dist > self.filter_length:
                filter_gt_lines.append(gt_line)
        return gt_lines


    def transform(self, results: dict) -> dict:
        map_size = results["img_shape"]
        if "gt_lines" in results:
            gt_lines = results["gt_lines"]
        else:
            gt_lines = results["eval_gt_lines"]

        if self.filter_small_line:
            gt_lines = self.filter_small_lines(gt_lines)

        # 对gt_lines进行排序, 按照点数的数量从小到大排序，最后再绘制长的线，保证在降采样的分辨率上，如果有重叠的位置，保证长的优先
        gt_lines = sorted(gt_lines, key=lambda x: x["points"].shape[0])

        gt_line_maps_stages = []
        line_map, line_map_id, line_map_cls, line_map_visible,\
        line_map_hanging, line_map_covered = self._gen_line_map(gt_lines, map_size)

        for gt_down_scale in self.gt_down_scales:
            gt_line_maps = self._gen_gt_line_maps(line_map, line_map_id, line_map_cls,
                                                  line_map_visible, line_map_hanging, line_map_covered, gt_down_scale)

            gt_confidence = torch.from_numpy(gt_line_maps[0])
            gt_offset_x = torch.from_numpy(gt_line_maps[1])
            gt_offset_y = torch.from_numpy(gt_line_maps[2])
            gt_line_index = torch.from_numpy(gt_line_maps[3])
            ignore_mask = torch.from_numpy(gt_line_maps[4])
            foreground_mask = torch.from_numpy(gt_line_maps[5])
            gt_line_id = torch.from_numpy(gt_line_maps[6])
            gt_line_cls = torch.from_numpy(gt_line_maps[7])
            foreground_expand_mask = torch.from_numpy(gt_line_maps[8])

            gt_confidence_visible = torch.from_numpy(gt_line_maps[9])
            gt_confidence_hanging = torch.from_numpy(gt_line_maps[10])
            gt_confidence_covered = torch.from_numpy(gt_line_maps[11])

            orient_map_mask, orient_map_sin, orient_map_cos = self._gen_gt_orient_maps(gt_lines, map_size, gt_down_scale)
            orient_map_mask = torch.from_numpy(orient_map_mask)
            orient_map_sin = torch.from_numpy(orient_map_sin)
            orient_map_cos = torch.from_numpy(orient_map_cos)

            # 对曲线起点和终点进行预测
            point_map_mask, point_map_confidence, point_map_index = self._gen_gt_point_maps(gt_lines, map_size, gt_down_scale)
            point_map_mask = torch.from_numpy(point_map_mask)
            point_map_confidence = torch.from_numpy(point_map_confidence)
            point_map_index = torch.from_numpy(point_map_index)

            gt_line_maps = {
                            "gt_confidence": gt_confidence,
                            "gt_offset_x": gt_offset_x,
                            "gt_offset_y": gt_offset_y,
                            "gt_line_index": gt_line_index,
                            "ignore_mask": ignore_mask,
                            "foreground_mask": foreground_mask,
                            "gt_line_id": gt_line_id,
                            "gt_line_cls": gt_line_cls,
                            "foreground_expand_mask": foreground_expand_mask,
                            "orient_map_mask": orient_map_mask,
                            "orient_map_sin": orient_map_sin,
                            "orient_map_cos": orient_map_cos,

                            # 新增可见、悬空和被草遮挡属性
                            "gt_confidence_visible": gt_confidence_visible,
                            "gt_confidence_hanging": gt_confidence_hanging,
                            "gt_confidence_covered": gt_confidence_covered,

                            # 新增曲线的起点和终点的heatmap和emb预测
                            "point_map_mask": point_map_mask,
                            "point_map_confidence": point_map_confidence,
                            "point_map_index": point_map_index,
                            }

            gt_line_maps_stages.append(gt_line_maps)

        results['gt_line_maps_stages'] = gt_line_maps_stages
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_down_scales={self.gt_down_scales}, '
        return repr_str


@TRANSFORMS.register_module()
class GgldDetrLineCapture(BaseTransform):
    def __init__(self,
                 gt_down_scales: Optional[Tuple[int, int]] = None,
                 num_classes: int = 1,
                 filter_small_line=False,
                 filter_length=10,
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_num=20,  # 曲线采样的点数
                 padding_value=-10000,
                 ) -> None:
        self.gt_down_scales = gt_down_scales
        self.num_classes = num_classes

        self.filter_small_line = filter_small_line
        self.filter_length = filter_length
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

    def _gen_line_map(self, gt_lines, map_size):
        line_map = np.zeros(map_size, dtype=np.uint8)
        line_map_id = np.zeros(map_size, dtype=np.uint8)
        line_map_cls = np.zeros(map_size, dtype=np.uint8)

        line_map_visible = np.zeros(map_size, dtype=np.uint8)
        line_map_hanging = np.zeros(map_size, dtype=np.uint8)
        line_map_covered = np.zeros(map_size, dtype=np.uint8)

        thickness = 1
        if map_size[1] > 240:
            thickness = 2

        if map_size[1] > 480:
            thickness = 4

        for gt_line in gt_lines:
            label = gt_line['label']
            line_points = gt_line['points']
            index = gt_line['index'] + 1     # 序号从0开始的
            line_id = gt_line['line_id'] + 1
            category_id = gt_line['category_id'] + 1

            line_points_type = gt_line['points_type']
            line_points_visible = gt_line['points_visible']
            line_points_hanging = gt_line['points_hanging']
            line_points_covered = gt_line['points_covered']

            pre_point = line_points[0]
            pre_point_type = line_points_type[0]
            pre_point_visible = line_points_visible[0]
            pre_point_hanging = line_points_hanging[0]
            pre_point_covered = line_points_covered[0]

            for cur_point, cur_point_type, cur_point_visible, cur_point_hanging, cur_point_covered in \
                    zip(line_points[1:], line_points_type[1:], line_points_visible[1:],
                        line_points_hanging[1:], line_points_covered[1:]):

                x1, y1 = round(pre_point[0]), round(pre_point[1])
                x2, y2 = round(cur_point[0]), round(cur_point[1])
                cv2.line(line_map, (x1, y1), (x2, y2), (index,), thickness=thickness)
                cv2.line(line_map_id, (x1, y1), (x2, y2), (line_id,), thickness=thickness)

                # cls_value = self.class_dic[label] + 1
                # cv2.line(line_map_cls, (x1, y1), (x2, y2), (category_id,))

                # 以起点的属性为准
                # 修改为分段类别
                cv2.line(line_map_cls, (x1, y1), (x2, y2), (int(pre_point_type[0]) + 1,), thickness=thickness)

                # 新增可见、悬空和被草遮挡的属性预测
                if pre_point_visible[0] > 0:
                    cv2.line(line_map_visible, (x1, y1), (x2, y2), (1,), thickness=thickness)

                if pre_point_hanging[0] > 0:
                    cv2.line(line_map_hanging, (x1, y1), (x2, y2), (1,), thickness=thickness)

                if pre_point_covered[0] > 0:
                    cv2.line(line_map_covered, (x1, y1), (x2, y2), (1,), thickness=thickness)

                pre_point = cur_point
                pre_point_type = cur_point_type
                pre_point_visible = cur_point_visible
                pre_point_hanging = cur_point_hanging
                pre_point_covered = cur_point_covered

        return line_map, line_map_id, line_map_cls, line_map_visible, line_map_hanging, line_map_covered

    def _gen_gt_line_maps(self, line_map, line_map_id, line_map_cls,
                          line_map_visible, line_map_hanging, line_map_covered, grid_size):
        line_map_h, line_map_w = line_map.shape
        gt_map_h, gt_map_w = math.ceil(line_map_h / grid_size), math.ceil(
            line_map_w / grid_size
        )
        gt_confidence = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_offset_x = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_offset_y = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_line_index = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_line_id = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)

        gt_line_cls = np.zeros((self.num_classes, gt_map_h, gt_map_w), dtype=np.float32)

        # 新增可见、悬空、被草遮挡的confidence预测
        gt_confidence_visible = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_confidence_hanging = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)
        gt_confidence_covered = np.zeros((1, gt_map_h, gt_map_w), dtype=np.float32)

        for y in range(0, gt_map_h):
            for x in range(0, gt_map_w):
                start_x, end_x = x * grid_size, (x + 1) * grid_size
                end_x = end_x if end_x < line_map_w else line_map_w
                start_y, end_y = y * grid_size, (y + 1) * grid_size
                end_y = end_y if end_y < line_map_h else line_map_h
                grid = line_map[start_y:end_y, start_x:end_x]

                grid_id = line_map_id[start_y:end_y, start_x:end_x]
                grid_cls = line_map_cls[start_y:end_y, start_x:end_x]

                confidence = 1 if np.any(grid) else 0
                gt_confidence[0, y, x] = confidence
                if confidence == 1:
                    ys, xs = np.nonzero(grid)
                    offset_y, offset_x = sorted(
                        zip(ys, xs), key=lambda p: (p[0], -p[1]), reverse=True
                    )[0]
                    if grid_size != 1:
                        gt_offset_x[0, y, x] = offset_x / (grid_size - 1)
                        gt_offset_y[0, y, x] = offset_y / (grid_size - 1)

                        # 设置成
                        # gt_confidence[0, y, x] = confidence - min(offset_x, offset_y)/grid_size

                    gt_line_index[0, y, x] = grid[offset_y, offset_x]
                    gt_line_id[0, y, x] = grid_id[offset_y, offset_x]

                    cls = grid_cls[offset_y, offset_x]
                    if cls > 0:
                        cls_indx = int(cls - 1)
                        gt_line_cls[cls_indx, y, x] = 1

                # gt_confidence_visible
                if np.any(line_map_visible[start_y:end_y, start_x:end_x]):
                    gt_confidence_visible[0, y, x] = 1

                # gt_confidence_hanging
                if np.any(line_map_hanging[start_y:end_y, start_x:end_x]):
                    gt_confidence_hanging[0, y, x] = 1

                # gt_confidence_covered
                if np.any(line_map_covered[start_y:end_y, start_x:end_x]):
                    gt_confidence_covered[0, y, x] = 1

        foreground_mask = gt_confidence.astype(np.uint8)

        # expand foreground mask
        kernel = np.ones((3, 3), np.uint8)
        foreground_expand_mask = cv2.dilate(foreground_mask[0], kernel)
        foreground_expand_mask = np.expand_dims(foreground_expand_mask.astype(np.uint8), axis=0)

        ignore_mask = np.zeros((1, gt_map_h, gt_map_w), dtype=np.uint8)
        # top, bottom = self.line_map_range
        ignore_mask[0, 0:-1, :] = 1     # 手动设置有效范围

        return gt_confidence, gt_offset_x, gt_offset_y, \
               gt_line_index, ignore_mask, foreground_mask, \
               gt_line_id, gt_line_cls, foreground_expand_mask, \
               gt_confidence_visible, gt_confidence_hanging, gt_confidence_covered

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

    def filter_near_same_points(self, line_points):
        mask = [True] * len(line_points)
        pre_point = line_points[0]
        for i, cur_point in enumerate(line_points[1:]):
            if np.all(cur_point == pre_point):
                mask[i + 1] = False
            else:
                pre_point = cur_point

        line_points = line_points[mask]
        return line_points

    def get_orient_sin_cos(self, orient):
        # 将MEBOW的格式转为[-pi, pi], 将角度的范围转换为kitti数据的格式
        # 以x为0度方向(水平向右), 范围：-pi~pi，顺时针为正，逆时针为负
        if orient > 180:
            orient = orient - 360

        orient = -orient      # 将逆时针方向修改为顺时针方向
        orient = orient - 90  # 将垂直向上为0度改成以水平向右为0度
        if orient > 180:      # 将范围限制到-pi~pi的范围内
            orient = orient - 360
        if orient < -180:
            orient = orient + 360

        orient = orient / 180 * math.pi

        orient_sin = math.sin(orient)
        orient_cos = math.cos(orient)
        return orient_sin, orient_cos

    def _gen_gt_orient_maps(self, gt_lines, map_size, down_scale):
        down_scale_map_size = (int(map_size[0]/down_scale), int(map_size[1]/down_scale))
        orient_map_mask = np.zeros(down_scale_map_size, dtype=np.uint8)
        orient_map_sin = np.zeros(down_scale_map_size, dtype=np.float32)
        orient_map_cos = np.zeros(down_scale_map_size, dtype=np.float32)

        for gt_line in gt_lines:
            label = gt_line['label']
            line_points = gt_line['points']
            line_points = line_points/down_scale

            # 过滤重复的点
            line_points = self.filter_near_same_points(line_points)

            index = gt_line['index'] + 1     # 序号从0开始的
            line_id = gt_line['line_id'] + 1
            category_id = gt_line['category_id'] + 1

            pre_point = line_points[0]
            for i, cur_point in enumerate(line_points[1:]):
                orient = self.cal_points_orient(pre_point, cur_point)
                c_x, c_y = int(pre_point[0]), int(pre_point[1])
                c_x = min(max(0, c_x), down_scale_map_size[1]-1)
                c_y = min(max(0, c_y), down_scale_map_size[0]-1)

                if orient != -1:
                    # print(c_x, c_y, pre_point[0], pre_point[1], gt_line['points'][i+1, 0], gt_line['points'][i+1, 1])
                    orient_map_mask[c_y, c_x] = 1

                    orient_sin, orient_cos = self.get_orient_sin_cos(orient)

                    # 转为三角函数 sin、cos的形式
                    orient_map_sin[c_y, c_x] = orient_sin
                    orient_map_cos[c_y, c_x] = orient_cos

                pre_point = cur_point

        orient_map_mask = np.expand_dims(orient_map_mask, axis=0)
        orient_map_sin = np.expand_dims(orient_map_sin, axis=0)
        orient_map_cos = np.expand_dims(orient_map_cos, axis=0)

        return orient_map_mask, orient_map_sin, orient_map_cos
    #
    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def _gen_gt_point_maps(self, gt_lines, map_size, down_scale):
        down_scale_map_size = (int(map_size[0]/down_scale), int(map_size[1]/down_scale))
        point_map_mask = np.zeros(down_scale_map_size, dtype=np.uint8)
        point_map_confidence = np.zeros(down_scale_map_size, dtype=np.float32)
        point_map_index = np.zeros(down_scale_map_size, dtype=np.float32)

        map_height, map_width = down_scale_map_size[0], down_scale_map_size[1]

        for gt_line in gt_lines:
            label = gt_line['label']
            line_points = gt_line['points']
            line_points = line_points/down_scale

            # 过滤重复的点
            line_points = self.filter_near_same_points(line_points)

            index = gt_line['index'] + 1     # 序号从0开始的
            start_point = line_points[0]
            end_point = line_points[-1]

            middle_point = line_points[len(line_points)//2]

            # dist_h = abs(start_point[1] - end_point[1])
            # dist_w = abs(start_point[0] - end_point[0])

            dist_h = abs(max(line_points[:, 1]) - min(line_points[:, 1]))
            dist_w = abs(max(line_points[:, 0]) - min(line_points[:, 0]))
            radius = self.gaussian_radius((math.ceil(dist_h), math.ceil(dist_w)))
            radius = max(0, int(radius))

            # for _point in [start_point, end_point, middle_point]:
            for _point in [start_point, end_point]:
                diameter = 2 * radius + 1
                gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
                # import matplotlib.pyplot as plt
                # plt.imshow(gaussian)
                # plt.show()
                x, y = int(_point[0]), int(_point[1])

                left, right = min(x, radius), min(map_width - x, radius + 1)
                top, bottom = min(y, radius), min(map_height - y, radius + 1)

                k = 1
                region_heatmap = point_map_confidence[y - top:y + bottom, x - left:x + right]
                masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
                if min(masked_gaussian.shape) > 0 and min(region_heatmap.shape) > 0:  # TODO debug
                    np.maximum(region_heatmap, masked_gaussian * k, out=region_heatmap)

                region_index = point_map_index[y - top:y + bottom, x - left:x + right]
                region_mask = point_map_mask[y - top:y + bottom, x - left:x + right]

                if min(masked_gaussian.shape) > 0 and min(region_heatmap.shape) > 0:
                    region_index[region_heatmap > 0.2] = index
                    region_mask[region_heatmap > 0.2] = 1

                # point_map_index[y - top:y + bottom, x - left:x + right] = index
                # point_map_mask[y - top:y + bottom, x - left:x + right] = 1

                # c_x, c_y = int(_point[0]), int(_point[1])
                # c_x = min(max(0, c_x), down_scale_map_size[1]-1)
                # c_y = min(max(0, c_y), down_scale_map_size[0]-1)
                # point_map_mask[c_y, c_x] = 1
                # point_map_confidence[c_y, c_x] = 1
                # point_map_index[c_y, c_x] = index

        point_map_mask = np.expand_dims(point_map_mask, axis=0)
        point_map_confidence = np.expand_dims(point_map_confidence, axis=0)
        point_map_index = np.expand_dims(point_map_index, axis=0)

        return point_map_mask, point_map_confidence, point_map_index

    def filter_small_lines(self, gt_lines):
        filter_gt_lines = []
        for gt_line in gt_lines:
            line_points = gt_line["points"]
            points_0 = line_points[:-1]
            points_1 = line_points[1:]

            points_dist = np.sqrt((points_0[:, 0] - points_1[:, 0]) ** 2
                                + (points_0[:, 1] - points_1[:, 1]) ** 2)

            line_dist = np.sum(points_dist)
            if line_dist > self.filter_length:
                filter_gt_lines.append(gt_line)
        return gt_lines

    def transform(self, results: dict) -> dict:
        if "gt_lines" in results:
            gt_lines = results["gt_lines"]
        else:
            gt_lines = results["eval_gt_lines"]

        map_size = results["img_shape"]

        if self.filter_small_line:
            gt_lines = self.filter_small_lines(gt_lines)

        # 对gt_lines进行排序, 按照点数的数量从小到大排序，最后再绘制长的线，保证在降采样的分辨率上，如果有重叠的位置，保证长的优先
        gt_lines = sorted(gt_lines, key=lambda x: x["points"].shape[0])

        # 产生不同stages的gt
        gt_line_maps_stages = []
        line_map, line_map_id, line_map_cls, line_map_visible,\
        line_map_hanging, line_map_covered = self._gen_line_map(gt_lines, map_size)

        for gt_down_scale in self.gt_down_scales:
            gt_line_maps = self._gen_gt_line_maps(line_map, line_map_id, line_map_cls,
                                                  line_map_visible, line_map_hanging, line_map_covered, gt_down_scale)

            gt_confidence = torch.from_numpy(gt_line_maps[0])
            gt_offset_x = torch.from_numpy(gt_line_maps[1])
            gt_offset_y = torch.from_numpy(gt_line_maps[2])
            gt_line_index = torch.from_numpy(gt_line_maps[3])
            ignore_mask = torch.from_numpy(gt_line_maps[4])
            foreground_mask = torch.from_numpy(gt_line_maps[5])
            gt_line_id = torch.from_numpy(gt_line_maps[6])
            gt_line_cls = torch.from_numpy(gt_line_maps[7])
            foreground_expand_mask = torch.from_numpy(gt_line_maps[8])

            gt_confidence_visible = torch.from_numpy(gt_line_maps[9])
            gt_confidence_hanging = torch.from_numpy(gt_line_maps[10])
            gt_confidence_covered = torch.from_numpy(gt_line_maps[11])

            orient_map_mask, orient_map_sin, orient_map_cos = self._gen_gt_orient_maps(gt_lines, map_size, gt_down_scale)
            orient_map_mask = torch.from_numpy(orient_map_mask)
            orient_map_sin = torch.from_numpy(orient_map_sin)
            orient_map_cos = torch.from_numpy(orient_map_cos)
            
            # 对曲线起点和终点进行预测
            point_map_mask, point_map_confidence, point_map_index = self._gen_gt_point_maps(gt_lines, map_size, gt_down_scale)
            point_map_mask = torch.from_numpy(point_map_mask)
            point_map_confidence = torch.from_numpy(point_map_confidence)
            point_map_index = torch.from_numpy(point_map_index)

            gt_line_maps = {
                            "gt_confidence": gt_confidence,
                            "gt_offset_x": gt_offset_x,
                            "gt_offset_y": gt_offset_y,
                            "gt_line_index": gt_line_index,
                            "ignore_mask": ignore_mask,
                            "foreground_mask": foreground_mask,
                            "gt_line_id": gt_line_id,
                            "gt_line_cls": gt_line_cls,
                            "foreground_expand_mask": foreground_expand_mask,
                            "orient_map_mask": orient_map_mask,
                            "orient_map_sin": orient_map_sin,
                            "orient_map_cos": orient_map_cos,

                            # 新增可见、悬空和被草遮挡属性
                            "gt_confidence_visible": gt_confidence_visible,
                            "gt_confidence_hanging": gt_confidence_hanging,
                            "gt_confidence_covered": gt_confidence_covered,

                            # 新增曲线的起点和终点的heatmap和emb预测
                            "point_map_mask": point_map_mask,
                            "point_map_confidence": point_map_confidence,
                            "point_map_index": point_map_index,
                            }

            gt_line_maps_stages.append(gt_line_maps)

        gt_line_labels = []
        gt_line_instances = []
        for gt_line in gt_lines:
            line_points_type = gt_line['points_type']

            # line_points_type为每个点的类别属性，得到统计最多的类别
            line_points_type = np.array([int(_type[0]) for _type in line_points_type])
            line_cls = np.argmax(np.bincount(line_points_type.astype(np.int32)))
            gt_line_labels.append(line_cls)

            line_points = gt_line['points']

            instance = LineString(np.array(line_points))
            gt_line_instances.append(instance)

        patch_size = results["img_shape"]
        gt_line_instances = GBLDDetrInstanceLines(gt_line_instances, gt_line_labels, self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num, self.padding_value,
                                            patch_size=patch_size)

        results['gt_line_maps_stages'] = gt_line_maps_stages
        results['gt_line_instances_stages'] = [gt_line_instances] * len(gt_line_maps_stages)    # 在不同的stage中是同样的，对应原图的输入大小
        results['gt_line_labels_stages'] = [np.array(gt_line_labels)] * len(gt_line_maps_stages)  # 同上
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_down_scales={self.gt_down_scales}, '
        return repr_str

@TRANSFORMS.register_module()
class PackGbldDetrMono2dInputs(BaseTransform):
    INPUTS_KEYS = ['points', 'img']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_line_maps_stages',
        'gt_bboxes',
        'gt_bboxes_labels',
        'gt_line_instances_stages',
        'gt_line_labels_stages',
        'gt_bboxes',
        'gt_labels'
    ]

    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    ]

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pcd_rotation_angle', 'lidar_path',
                            'transformation_3d_flow', 'trans_mat',
                            'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                            'cam2global', 'crop_offset', 'img_crop_offset',
                            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                            'num_ref_frames', 'num_views', 'ego2global',
                            'axis_align_matrix', "gt_lines", "eval_gt_lines", "sample_idx")
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    def _remove_prefix(self, key: str) -> str:
        if key.startswith('gt_'):
            key = key[3:]
        return key

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif 'images' in results:
                if len(results['images'].keys()) == 1:
                    cam_type = list(results['images'].keys())[0]
                    # single-view image
                    if key in results['images'][cam_type]:
                        data_metas[key] = results['images'][cam_type][key]
                else:
                    # multi-view image
                    img_metas = []
                    cam_types = list(results['images'].keys())
                    for cam_type in cam_types:
                        if key in results['images'][cam_type]:
                            img_metas.append(results['images'][cam_type][key])
                    if len(img_metas) > 0:
                        data_metas[key] = img_metas
            elif 'lidar_points' in results:
                if key in results['lidar_points']:
                    data_metas[key] = results['lidar_points'][key]
        data_sample.set_metainfo(data_metas)

        inputs = {}
        for key in self.keys:
            #
            if key == 'gt_labels_stages':
                gt_labels_stages = []
                for gt_line_labels in results['gt_line_labels_stages']:
                    if len(gt_line_labels) != 0:
                        gt_labels_stages.append(to_tensor(gt_line_labels))
                    else:
                        gt_labels_stages.append(to_tensor([]))
                gt_instances['gt_labels_stages'] = gt_labels_stages
                # gt_instances['gt_labels_stages'] = [to_tensor(gt_line_labels) for gt_line_labels in results['gt_line_labels_stages']]

            # box的形式xyxy
            if key == 'gt_bboxes_stages':
                gt_bboxes_stages = []
                for gt_line_instances in results['gt_line_instances_stages']:
                    if len(gt_line_instances) != 0:
                        gt_bboxes_stages.append(gt_line_instances.bbox)
                    else:
                        # print("zeroooooooo", results["img_name"], results["ann_path"])
                        gt_bboxes_stages.append([])
                gt_instances['gt_bboxes_stages'] = gt_bboxes_stages
                # gt_instances['gt_bboxes_stages'] = [gt_line_instances.bbox for gt_line_instances in results['gt_line_instances_stages']]

            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key in ["gt_line_maps_stages", 'gt_line_instances_stages', 'gt_line_labels_stages']:
                        gt_instances[key] = results[key]

                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_instances_stages = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        if "img_path" in results:
            data_sample.img_path = results["img_path"]
        else:
            data_sample.img_path = None
        #
        # if "img_shape" in results:
        #     data_sample.img_shape = results['img_shape']
        # else:
        #     data_sample.img_shape = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str