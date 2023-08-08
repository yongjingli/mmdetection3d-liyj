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
class GgldLoadLines(BaseTransform):
    def __init__(self, name="load_lines") -> None:
        self.name = name

    def transform(self, results: dict) -> dict:
        has_labels = False
        if results.get('eval_ann_info', None) is not None:
            if results['eval_ann_info'].get('gt_lines', None) is not None:
                eval_gt_lines = results['eval_ann_info']['gt_lines']
                results['eval_gt_lines'] = eval_gt_lines
                results['ori_eval_gt_lines'] = copy.deepcopy(eval_gt_lines)
                has_labels = True

        if results.get('ann_info', None) is not None:
            if results['ann_info'].get('gt_lines', None) is not None:
                gt_lines = results['ann_info']['gt_lines']
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
class GgldRandomCrop(BaseTransform):
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
        for gt_line in gt_lines:
            points = gt_line["points"]

            mask_x = np.bitwise_and(points[:, 0] > (crop_x1 - 1), points[:, 0] < crop_x2)
            mask_y = np.bitwise_and(points[:, 1] > (crop_y1 - 1), points[:, 1] < crop_y2)

            mask = np.bitwise_and(mask_x, mask_y)
            points = points[mask]
            if len(points) < 2:
                continue

            points[:, 0] = points[:, 0] - crop_x1
            points[:, 1] = points[:, 1] - crop_y1
            gt_line["points"] = points

            crop_gt_lines.append(gt_line)

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
class GgldRandomFlip(RandomFlip):
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
            elif direction == 'vertical':
                flip_line[:, 1] = h - line[:, 1]
            elif direction == 'diagonal':
                flip_line[:, 0] = w - line[:, 0]
                flip_line[:, 1] = h - line[:, 1]
            else:
                raise ValueError(
                    f"Flipping direction must be 'horizontal', 'vertical', \
                      or 'diagonal', but got '{direction}'")

        return flipped


@TRANSFORMS.register_module()
# class SegLabelMapping(BaseTransform):
class GgldColor(BaseTransform):
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
class GgldResize(Resize):
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
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')



@TRANSFORMS.register_module()
class PackGbldMono2dInputs(BaseTransform):
    INPUTS_KEYS = ['points', 'img']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_line_maps_stages',
        # 'gt_bboxes',
        # 'gt_bboxes_labels',
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
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
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
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == "gt_line_maps_stages":
                        gt_instances[key] = results[key]
                    # else:
                    #     gt_instances[key] = results[key]
                    # if key == 'gt_bboxes_labels':
                    #     gt_instances['labels'] = results[key]
                    # else:
                    #     gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

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


@TRANSFORMS.register_module()
class GgldLineMapsGenerate(BaseTransform):
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

    def transform(self, results: dict) -> dict:
        map_size = results["img_shape"]
        gt_lines = results["gt_lines"]

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
                            }

            gt_line_maps_stages.append(gt_line_maps)

        results['gt_line_maps_stages'] = gt_line_maps_stages
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_down_scales={self.gt_down_scales}, '
        return repr_str