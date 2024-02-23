# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, List, Union

import numpy as np
import os
import json
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from typing import Callable, List, Optional, Set, Union
from mmengine.dataset import BaseDataset
import copy

@DATASETS.register_module()
class GbldDetrMono2dDataset(BaseDataset):
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 load_type: str = 'frame_based',
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 **kwargs) -> None:
        
        self.test_mode = test_mode
        self.load_eval_anns = load_eval_anns
        self.load_type = load_type
        self.backend_args = backend_args
        self.idx = 0

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def parse_ann_info(self, info: dict) -> dict:
        ann_path = info["ann_path"]
        with open(ann_path, 'r') as f:
            anns = json.load(f)

        ann_info = dict()

        remap_anns = []
        for i, shape in enumerate(anns['shapes']):
            line_id = int(shape['id'])
            label = shape['label']

            if label not in self.metainfo["classes"]:
                continue

            category_id = self.metainfo["classes"].index(label)

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
                point_type[0] = self.metainfo["classes"].index(point_type[0])

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

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        info = super().parse_data_info(info)

        img_path = info["img"]
        ann_path = info["ann"]

        # img_path = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/debug_overfit/train/images/1696921603.610147.jpg"
        # ann_path = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/debug_overfit/train/jsons/1696921603.610147.json"

        # TODO DEBUG
        # img_path = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/debug_overfit_gbld_detr/train/images/1696924422.61974.jpg"
        # ann_path = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/debug_overfit_gbld_detr/train/jsons/1696924422.61974.json"

        img_name = os.path.split(img_path)[-1]
        ann_name = os.path.split(ann_path)[-1]

        camera_info = dict()
        camera_info["img_name"] = img_name
        camera_info["ann_name"] = ann_name
        camera_info["img_path"] = img_path
        camera_info["ann_path"] = ann_path
        camera_info['sample_idx'] = self.idx
        self.idx = self.idx + 1

        if not self.test_mode:
            camera_info['ann_info'] = self.parse_ann_info(camera_info)
        if self.test_mode and self.load_eval_anns:
            camera_info['eval_ann_info'] = \
                self.parse_ann_info(camera_info)
        return camera_info