# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv.cnn import Scale
from mmdet.models.utils import multi_apply, select_single_mlvl
from mmengine.model import normal_init
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import limit_period, points_img2cam, xywhr2xyxyr
from mmdet3d.utils import (ConfigType, InstanceList, OptConfigType,
                           OptInstanceList)

from mmdet3d.models.dense_heads.anchor_free_mono3d_head import AnchorFreeMono3DHead
from mmengine.model import bias_init_with_prob, normal_init
RangeType = Sequence[Tuple[int, int]]
from typing import Any, List, Sequence, Tuple, Union
from mmcv.cnn import ConvModule
from mmdet3d.structures.det3d_data_sample import SampleList

INF = 1e8
from mmengine.model import BaseModule


@MODELS.register_module()
class GBLDMono2DHead(BaseModule):
    """Anchor-free head used in FCOS3D.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: True.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: True.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to
            https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: True.
        centerness_alpha (float): Parameter used to adjust the intensity
            attenuation from the center to the periphery. Default: 2.5.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dir (:obj:`ConfigDict` or dict): Config of direction classification loss.
        loss_attr (:obj:`ConfigDict` or dict): Config of attribute classification loss.
        loss_centerness (:obj:`ConfigDict` or dict): Config of centerness loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        centerness_branch (tuple[int]): Channels for centerness branch.
            Default: (64, ).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """  # noqa: E501

    def __init__(self,
                 # feat
                 num_classes: int,
                 in_channels: int,
                 # feat_channels: int = 256,
                 # stacked_convs: int = 4,

                 dcn_on_last_conv: bool = False,

                 seg_branch: Sequence[int] = (128, 64),
                 offset_branch: Sequence[int] = (128, 64),
                 seg_emb_branch: Sequence[int] = (128, 64),
                 connect_emb_branch: Sequence[int] = (128, 64),
                 cls_branch: Sequence[int] = (128, 64),

                 num_seg: int = 1,
                 num_offset: int = 2,
                 num_seg_emb: int = 1,
                 num_connect_emb: int = 1,

                 conv_bias: Union[bool, str] = 'auto',
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,

                 # predict
                 in_feat_index: Sequence[int] = (0, 1, 2, 3),
                 strides: Sequence[int] = (4, 8, 16, 32),
                 gbld_decode: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     confident_t=0.2),

                 stage_loss_weight=(1, 1, 1, 1),    # 设置不同stage的loss weight

                 loss_seg: ConfigType = dict(
                     type='GbldSegLoss',
                     focal_loss_gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),

                 loss_offset: ConfigType = dict(
                     type='GbldOffsetLoss',
                     loss_weight=1.0),

                 loss_seg_emb: ConfigType = dict(
                     type='GbldEmbLoss',
                     pull_margin=0.5,
                     push_margin=1.0,
                     loss_weight=1.0),

                 loss_connect_emb: ConfigType = dict(
                     pull_margin=0.5,
                     push_margin=1.0,
                     loss_weight=1.0),

                 loss_cls: ConfigType = dict(
                     type='GbldClsLoss',
                     num_classes=1,
                     loss_weight=1.0),

                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super(GBLDMono2DHead, self).__init__(init_cfg=init_cfg)
        # 配置记录
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv

        self.seg_branch = seg_branch
        self.offset_branch = offset_branch
        self.seg_emb_branch = seg_emb_branch
        self.connect_emb_branch = connect_emb_branch
        self.cls_branch = cls_branch

        self.num_seg = num_seg
        self.num_offset = num_offset
        self.num_seg_emb = num_seg_emb
        self.num_connect_emb = num_connect_emb

        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # 初始化
        self._init_predictor()
        self.init_weights()

        # predict
        self.in_feat_index = in_feat_index
        self.strides = strides
        self.gbld_decode = MODELS.build(gbld_decode)


        # loss
        self.stage_loss_weight = torch.tensor(stage_loss_weight)
        self.loss_seg = MODELS.build(loss_seg)
        self.loss_offset = MODELS.build(loss_offset)
        self.loss_seg_emb = MODELS.build(loss_seg_emb)
        self.loss_connect_emb = MODELS.build(loss_connect_emb)
        self.loss_cls = MODELS.build(loss_cls)

    def _init_predictor(self):
        # seg
        self.conv_seg_prev = self._init_branch(
            conv_channels=self.seg_branch,
            conv_strides=(1,) * len(self.seg_branch))
        self.conv_seg = nn.Conv2d(self.seg_branch[-1], self.num_seg, 1)

        # offset
        self.conv_offset_prev = self._init_branch(
            conv_channels=self.offset_branch,
            conv_strides=(1,) * len(self.offset_branch))
        self.conv_offset = nn.Conv2d(self.offset_branch[-1], self.num_offset, 1)

        # seg_emb
        self.conv_seg_emb_prev = self._init_branch(
            conv_channels=self.seg_emb_branch,
            conv_strides=(1,) * len(self.seg_emb_branch))
        self.conv_seg_emb = nn.Conv2d(self.seg_emb_branch[-1], self.num_seg_emb, 1)

        # connect_emb
        self.conv_connect_emb_prev = self._init_branch(
            conv_channels=self.connect_emb_branch,
            conv_strides=(1,) * len(self.connect_emb_branch))
        self.conv_connect_emb = nn.Conv2d(self.connect_emb_branch[-1], self.num_connect_emb, 1)

        # cls
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1,) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.num_classes, 1)

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.in_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.in_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        return conv_before_pred

    def init_weights(self):
        for modules in [self.conv_seg_prev, self.conv_offset_prev, self.conv_seg_emb_prev,
                        self.conv_connect_emb_prev, self.conv_cls_prev,]:
            for m in modules:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        for module in [self.conv_seg, self.conv_offset, self.conv_seg_emb,
                        self.conv_connect_emb, self.conv_cls]:
            normal_init(module, std=0.01, bias=bias_cls)

    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        x_neck_select = []
        for feat_index in self.in_feat_index:
            x_neck_select.append(x[feat_index])

        # 设置只输出单层的结果
        if torch.onnx.is_in_onnx_export():
            x_neck_select = x_neck_select[0]
            return self.forward_single(x_neck_select)

        # return multi_apply(self.forward_single, x)
        return multi_apply(self.forward_single, x_neck_select)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        # seg
        clone_seg_feat = x.clone()
        for conv_dir_seg_prev_layer in self.conv_seg_prev:
            clone_seg_feat = conv_dir_seg_prev_layer(clone_seg_feat)
        dir_seg_pred = self.conv_seg(clone_seg_feat)

        # offset
        clone_offset_feat = x.clone()
        for conv_dir_offset_prev_layer in self.conv_offset_prev:
            clone_offset_feat = conv_dir_offset_prev_layer(clone_offset_feat)
        dir_offset_pred = self.conv_offset(clone_offset_feat)

        # seg emb
        clone_seg_emb_feat = x.clone()
        for conv_dir_seg_emb_prev_layer in self.conv_seg_emb_prev:
            clone_seg_emb_feat = conv_dir_seg_emb_prev_layer(clone_seg_emb_feat)
        dir_seg_emb_pred = self.conv_seg_emb(clone_seg_emb_feat)

        # connect emb
        clone_connect_emb_feat = x.clone()
        for conv_dir_connect_emb_prev_layer in self.conv_connect_emb_prev:
            clone_connect_emb_feat = conv_dir_connect_emb_prev_layer(clone_connect_emb_feat)
        dir_connect_emb_pred = self.conv_connect_emb(clone_connect_emb_feat)

        # cls
        clone_cls_feat = x.clone()
        for conv_dir_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_dir_cls_prev_layer(clone_cls_feat)
        dir_cls_pred = self.conv_cls(clone_cls_feat)

        return dir_seg_pred, dir_offset_pred, dir_seg_emb_pred, dir_connect_emb_pred, dir_cls_pred

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x)
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        return predictions

    def predict_by_feat(self,
                        seg_pred: List[Tensor],
                        offset_pred: List[Tensor],
                        seg_emb_pred: List[Tensor],
                        connect_emb_pred: List[Tensor],
                        cls_pred: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: OptConfigType = None,
                        rescale: bool = False) -> InstanceList:

        assert len(seg_pred) == len(offset_pred) == len(seg_emb_pred) == \
            len(connect_emb_pred) == len(cls_pred)
        num_levels = len(seg_pred)

        batch_results = []
        # 在batch的维度上进行解析
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            # print(img_meta)
            # exit(1)
            seg_pred_list = select_single_mlvl(seg_pred, img_id)
            offset_pred_list = select_single_mlvl(offset_pred, img_id)
            seg_emb_pred_list = select_single_mlvl(seg_emb_pred, img_id)
            connect_emb_pred_list = select_single_mlvl(connect_emb_pred, img_id)
            cls_pred_list = select_single_mlvl(cls_pred, img_id)

            stages_result = []
            for i in range(num_levels):
                self.gbld_decode.grid_size = self.strides[i]
                stages_result.append(self.gbld_decode(seg_pred_list[i], offset_pred_list[i],
                                                      seg_emb_pred_list[i], connect_emb_pred_list[i],
                                                      cls_pred_list[i]))
            if rescale:
                # 将结果返回到原图的尺寸上
                scale_factor = img_meta["scale_factor"]
                # ori_shape = img_meta["ori_shape"]
                w_scale, h_scale = scale_factor
                # ori_img_h, ori_img_w = ori_shape

                for stage_result in stages_result:
                    pred_lines = stage_result[0]

                    for pred_line in pred_lines:
                        pred_line[:, 0] = pred_line[:, 0] / w_scale
                        pred_line[:, 1] = pred_line[:, 1] / h_scale

            results = InstanceData()
            results.stages_result = stages_result
            batch_results.append(results)
        return batch_results

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:
        outs = self(x)
        batch_gt_instances_3d = []
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (batch_gt_instances_3d, batch_gt_instances,
                              batch_img_metas, batch_gt_instances_ignore)

        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
            self,
            seg_preds: List[Tensor],
            offset_preds: List[Tensor],
            seg_emb_preds: List[Tensor],
            connect_emb_preds: List[Tensor],
            cls_preds: List[Tensor],

            batch_gt_instances_3d: InstanceList,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:

        assert len(seg_preds) == len(offset_preds) == len(seg_emb_preds) == len(
            connect_emb_preds) == len(cls_preds)

        #  输出不同分辨率的预测与生成gt的数量保持一致
        assert len(seg_preds) == len(batch_gt_instances[0]["gt_line_maps_stages"]),  "output stages num != gt stages num"

        num_levels = len(seg_preds)
        loss_seg = []
        loss_offset = []
        loss_seg_emb = []
        loss_connect_emb = []
        loss_cls = []

        stage_batch_gt_segs = []
        stage_batch_gt_offsets = []
        stage_batch_gt_line_indexs = []
        stage_batch_gt_ignore_masks = []
        stage_batch_gt_foreground_masks = []
        stage_batch_gt_line_ids = []
        stage_batch_gt_line_clses = []

        for i in range(num_levels):
            batch_gt_segs = []
            batch_gt_offsets = []
            batch_gt_line_indexs = []
            batch_gt_ignore_masks = []
            batch_gt_foreground_masks = []
            batch_gt_line_ids = []
            batch_gt_line_clses = []

            for gt_instances in batch_gt_instances:
                gt_line_maps_stages = gt_instances["gt_line_maps_stages"]
                batch_gt_segs.append(gt_line_maps_stages[i]["gt_confidence"])

                gt_offset_x = gt_line_maps_stages[i]["gt_offset_x"]
                gt_offset_y = gt_line_maps_stages[i]["gt_offset_y"]
                batch_gt_offsets.append(torch.concat([gt_offset_x, gt_offset_y], dim=0))

                batch_gt_line_indexs.append(gt_line_maps_stages[i]["gt_line_index"])

                batch_gt_ignore_masks.append(gt_line_maps_stages[i]["ignore_mask"])
                batch_gt_foreground_masks.append(gt_line_maps_stages[i]["foreground_mask"])
                batch_gt_line_ids.append(gt_line_maps_stages[i]["gt_line_id"])
                batch_gt_line_clses.append(gt_line_maps_stages[i]["gt_line_cls"])

            batch_gt_segs = torch.stack(batch_gt_segs, dim=0)
            batch_gt_offsets = torch.stack(batch_gt_offsets, dim=0)
            batch_gt_line_indexs = torch.stack(batch_gt_line_indexs, dim=0)
            batch_gt_ignore_masks = torch.stack(batch_gt_ignore_masks, dim=0)
            batch_gt_foreground_masks = torch.stack(batch_gt_foreground_masks, dim=0)
            batch_gt_line_ids = torch.stack(batch_gt_line_ids, dim=0)
            batch_gt_line_clses = torch.stack(batch_gt_line_clses, dim=0)

            stage_batch_gt_segs.append(batch_gt_segs)
            stage_batch_gt_offsets.append(batch_gt_offsets)
            stage_batch_gt_line_indexs.append(batch_gt_line_indexs)
            stage_batch_gt_ignore_masks.append(batch_gt_ignore_masks)
            stage_batch_gt_foreground_masks.append(batch_gt_foreground_masks)
            stage_batch_gt_line_ids.append(batch_gt_line_ids)
            stage_batch_gt_line_clses.append(batch_gt_line_clses)

        for i in range(num_levels):
            seg_pred = seg_preds[i]                                # [B, 1, H, W]
            offset_pred = offset_preds[i]                          # [B, 2, H, W]
            seg_emb_pred = seg_emb_preds[i]                        # [B, 1, H, W]
            connect_emb_pred = connect_emb_preds[i]                # [B, 1, H, W]
            cls_pred = cls_preds[i]                                # [B, cls_num, H, W]

            gt_seg = stage_batch_gt_segs[i]
            gt_offset = stage_batch_gt_offsets[i]
            gt_line_index = stage_batch_gt_line_indexs[i]
            gt_ignore_mask = stage_batch_gt_ignore_masks[i]
            gt_foreground_mask = stage_batch_gt_foreground_masks[i]
            gt_line_id = stage_batch_gt_line_ids[i]
            gt_line_cls = stage_batch_gt_line_clses[i]

            # set gt divice
            gt_seg = gt_seg.to(seg_pred.device)
            gt_offset = gt_offset.to(seg_pred.device)
            gt_line_index = gt_line_index.to(seg_pred.device)
            gt_ignore_mask = gt_ignore_mask.to(seg_pred.device)
            gt_foreground_mask = gt_foreground_mask.to(seg_pred.device)
            gt_line_id = gt_line_id.to(seg_pred.device)
            gt_line_cls = gt_line_cls.float().to(seg_pred.device)

            _loss_seg = self.loss_seg(seg_pred, gt_seg, gt_ignore_mask,)
            _loss_offset = self.loss_offset(offset_pred, gt_offset, gt_foreground_mask)
            _seg_emb_pred = self.loss_seg_emb(seg_emb_pred, gt_line_index, gt_foreground_mask)

            _connect_emb_pred = self.loss_connect_emb(connect_emb_pred, gt_line_id, gt_foreground_mask)
            _cls_pred = self.loss_cls(cls_pred, gt_line_cls, gt_foreground_mask)

            loss_seg.append(_loss_seg)
            loss_offset.append(_loss_offset)
            loss_seg_emb.append(_seg_emb_pred)
            loss_connect_emb.append(_connect_emb_pred)
            loss_cls.append(_cls_pred)

        # 输出不同分辨率的预测与设置的stage_weight数量应该保持一致
        assert len(loss_seg) == len(self.stage_loss_weight), "output stages num != stage_loss_weight"
        loss_seg = [_loss * _stage_weight for _loss, _stage_weight in zip(loss_seg, self.stage_loss_weight)]
        loss_offset = [_loss * _stage_weight for _loss, _stage_weight in zip(loss_offset, self.stage_loss_weight)]
        loss_seg_emb = [_loss * _stage_weight for _loss, _stage_weight in zip(loss_seg_emb, self.stage_loss_weight)]
        loss_connect_emb = [_loss * _stage_weight for _loss, _stage_weight in zip(loss_connect_emb, self.stage_loss_weight)]
        loss_cls = [_loss * _stage_weight for _loss, _stage_weight in zip(loss_cls, self.stage_loss_weight)]

        loss_seg = sum(loss_seg)
        loss_offset = sum(loss_offset)
        loss_seg_emb = sum(loss_seg_emb)
        loss_connect_emb = sum(loss_connect_emb)
        loss_cls = sum(loss_cls)

        loss_dict = dict(
            loss_seg=loss_seg,
            loss_offset=loss_offset,
            loss_seg_emb=loss_seg_emb,
            loss_connect_emb=loss_connect_emb,
            loss_cls=loss_cls,)

        return loss_dict