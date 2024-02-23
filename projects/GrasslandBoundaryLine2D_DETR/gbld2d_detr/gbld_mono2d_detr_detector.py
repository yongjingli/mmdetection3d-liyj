# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptInstanceList
from mmdet.models.utils.misc import samplelist_boxtype2tensor


@MODELS.register_module()
class GBLDDetrMono2Dtector(SingleStageDetector):
    """Base class for monocular 3D single-stage detectors.

    Monocular 3D single-stage detectors directly and densely predict bounding
    boxes on the output features of the backbone+neck.
    """
    # 主要修改head
    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each image. Defaults to None.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each image. Defaults to None.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are 2D prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        for i, data_sample in enumerate(data_samples):
            # data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]

        samplelist_boxtype2tensor(data_samples)
        return data_samples

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs_dict (dict): Contains 'img' key
                with image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        batch_imgs = batch_inputs_dict['imgs']
        # get flops
        # batch_imgs = batch_inputs_dict
        x = self.backbone(batch_imgs)
        if self.with_neck:
            x = self.neck(x)
        return x

    # TODO: Support test time augmentation
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        pass
