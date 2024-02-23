import copy

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models import (DETR, ChannelMapper, DetDataPreprocessor, DETRHead, ResNet)
from mmengine.registry import MODELS as MMENGINMODELS
from mmdet3d.registry import MODELS
import torch.nn as nn
from typing import Union
from mmengine import ConfigDict
from mmdet.models.layers.positional_encoding import SinePositionalEncoding
from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoder, DetrTransformerDecoder
import torch
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList
from typing import List, Optional, Sequence, Tuple
from torch import Tensor
from typing import Dict, Tuple
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmdet.models.utils.misc import samplelist_boxtype2tensor


@MODELS.register_module()
class GBLDDetrMono2DHead(BaseModule):
    r"""Implementation of `DETR: End-to-End Object Detection with Transformers.

    <https://arxiv.org/pdf/2005.12872>`_.

    Code is modified from the `official github repo
    <https://github.com/facebookresearch/detr>`_.
    """
    def __init__(self,
                 num_queries: int = 100,
                 transform_feat_idx: int = -1,
                 with_seg: bool = False,  # 是否进行seg的预测
                 select_gt_stages_index: OptConfigType = None,
                 select_gt_stages_maps: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_decode: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 seg_decode: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: Union[dict, ConfigDict] = None) -> None:

        super(GBLDDetrMono2DHead, self).__init__(init_cfg=init_cfg)

        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        self.transform_feat_idx = transform_feat_idx
        self.num_queries = num_queries
        self.select_gt_stages_index = select_gt_stages_index
        self.select_gt_stages_maps = select_gt_stages_maps
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding

        self.bbox_decode = MMENGINMODELS.build(bbox_decode)

        self._init_layers()
        self.init_weights()

        # 是否进行seg的预测
        self.with_seg = with_seg
        # 修改为transforms的输入特征
        # seg_decode["in_feat_index"] = [transform_feat_idx]
        # seg_decode["strides"] = [seg_decode["strides"][transform_feat_idx]]
        # seg_decode["stage_loss_weight"] = [seg_decode["stage_loss_weight"][transform_feat_idx]]
        if self.with_seg:
            self.seg_decode = MMENGINMODELS.build(seg_decode)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = DetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def select_stage_instance(self, batch_data_samples, device=None):
        # 从gt_instances_stages选择其中一个stage作为gt_instance
        # device = self.query_embedding.device
        for data_samples in batch_data_samples:
            gt_instances_stages = data_samples.gt_instances_stages
            data_samples.gt_instances = InstanceData()
            for key, key_n in self.select_gt_stages_maps.items():
                if key in gt_instances_stages:
                    data_samples.gt_instances[key_n] = gt_instances_stages[key][self.select_gt_stages_index]
                    if (device is not None and
                            isinstance(data_samples.gt_instances[key_n], torch.Tensor)):
                        data_samples.gt_instances[key_n] = data_samples.gt_instances[key_n].to(device)

        return batch_data_samples

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:

        # 需要先计算seg的loss，transform只选择一个stage的gt，会修改instance的数值
        if self.with_seg:
            losses_seg = self.seg_decode.loss(x, batch_data_samples)

        batch_data_samples = self.select_stage_instance(batch_data_samples, device=x[0].device)
        head_inputs_dict = self.forward_transformer(x, batch_data_samples)

        losses = self.bbox_decode.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if self.with_seg:
            for key, loss in losses_seg.items():
                losses[key] = loss
        return losses

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        # img_feats = self.extract_feat(batch_inputs)

        batch_data_samples = self.select_stage_instance(batch_data_samples, device=x[0].device)
        head_inputs_dict = self.forward_transformer(x,
                                                    batch_data_samples)
        results_list = self.bbox_decode.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        if self.with_seg:
            # seg_results_list = self.seg_decode.predict(x, batch_data_samples=batch_data_samples)
            outs = self.seg_decode(x)
            for batch_i, results in enumerate(results_list):
                # results.seg = outs[0][self.transform_feat_idx][batch_i]
                # results.metainfo["seg"] = outs[0][self.transform_feat_idx][batch_i].sigmoid()
                results.seg = [None for _ in range(len(results.labels))]
                results.seg[0] = outs[0][self.transform_feat_idx][batch_i].sigmoid()

        # batch_data_samples = self.add_pred_to_datasample(
        #     batch_data_samples, results_list)
        return results_list

    def _forward(
            self,
            x: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:

        # img_feats = self.extract_feat(batch_inputs)
        batch_data_samples = self.select_stage_instance(batch_data_samples, device=x[0].device)

        head_inputs_dict = self.forward_transformer(x,
                                                    batch_data_samples)
        results = self.bbox_decode.forward(**head_inputs_dict)

        if self.with_seg:
            # seg_results_list = self.seg_decode.predict(x, batch_data_samples=batch_data_samples)
            outs = self.seg_decode(x)
            for batch_i, results in enumerate(results):
                # results.seg = outs[0][self.transform_feat_idx][batch_i]
                # results.metainfo["seg"] = outs[0][self.transform_feat_idx][batch_i].sigmoid()
                results.seg = [None for _ in range(len(results.labels))]
                results.seg[0] = outs[0][self.transform_feat_idx][batch_i].sigmoid()

        return results

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        # 从不同stage的特征选择其中一个特征进行transform
        img_feats_select = [img_feats[self.transform_feat_idx]]

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats_select, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        # 'memory_mask': 自注意力机制的特征的mask, mask为0代表有效位置（pre_transformer时得到）
        # 'memory_pos':  图像特征的位置编码（pre_transformer时得到）

        # 'memory':      进行自注意力后的图像特征(forward_encoder后得到)
        # 'query_pos':   （pre_decoder时得到，提早定义的query:self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)）
        # 'query':       根据query的长度全部设置为0(pre_decoder时得到)
        decoder_inputs_dict.update(tmp_dec_in)

#       query (Tensor): The queries of decoder inputs, has shape (bs, num_queries, dim).
#       query_pos (Tensor): The positional queries of decoder inputs, has shape (bs, num_queries, dim).
#       memory (Tensor): The output embeddings of the Transformer encoder, has shape (bs, num_feat_points, dim).
#       memory_mask (Tensor): ByteTensor, the padding mask of the memory, has shape (bs, num_feat_points).
#       memory_pos (Tensor): The positional embeddings of memory, has shape (bs, num_feat_points, dim).

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)

        # torch.Size([6, 2, 100, 256])
        # 6代表不同阶段hidden输出,与decode的num_layer有关
        # 2为batchsize的大小
        # 100为num_query
        # 256为query的特征维度
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such as
                `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """
        # 目前只是取最后一个特征进行预测, 最后一个为最小的特征
        feat = img_feats[-1]  # NOTE img_feats contains only one feature.
        # feat = img_feats  # NOTE img_feats contains only one feature.
        batch_size, feat_dim, _, _ = feat.shape
        # construct binary masks which for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            # masks = None
            masks = feat.new_zeros((batch_size, feat.shape[-2:][0], feat.shape[-2:][1]))
            # [batch_size, embed_dim, h, w]
            # pos_embed = self.positional_encoding(masks, input=feat)
            pos_embed = self.positional_encoding(masks)
        else:
            masks = feat.new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero values represent
            # ignored positions, while zero values mean valid positions.

            masks = F.interpolate(
                masks.unsqueeze(1),
                size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks)

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [bs, h*w, c]
        feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        # [bs, h, w] -> [bs, h*w]
        if masks is not None:
            masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        encoder_inputs_dict = dict(
            feat=feat, feat_mask=masks, feat_pos=pos_embed)
        decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, query_pos=feat_pos,
            key_padding_mask=feat_mask)  # for self_attn
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory'.
            - head_inputs_dict (dict): The keyword args dictionary of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """

        batch_size = memory.size(0)  # (bs, num_feat_points, dim)
        query_pos = self.query_embedding.weight
        # (num_queries, dim) -> (bs, num_queries, dim)
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        query = torch.zeros_like(query_pos)

        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output.

            - hidden_states (Tensor): Has shape
              (num_decoder_layers, bs, num_queries, dim)
        """

        #       query (Tensor): The input query, has shape (bs, num_queries, dim).
        #       key (Tensor): The input key, has shape (bs, num_keys, dim).
        #       value (Tensor): The input value with the same shape as `key`.
        #       query_pos (Tensor): The positional encoding for `query`, with thesame shape as `query`.
        #       key_pos (Tensor): The positional encoding for `key`, with the same shape as `key`.
        #       key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn` input. ByteTensor, has shape (bs, num_value).

        hidden_states = self.decoder(
            query=query,
            key=memory,
            value=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask)  # for cross_attn

        head_inputs_dict = dict(hidden_states=hidden_states)
        return head_inputs_dict

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples