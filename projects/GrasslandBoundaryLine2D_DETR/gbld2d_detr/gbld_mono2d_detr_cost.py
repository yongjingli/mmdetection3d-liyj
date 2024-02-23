from abc import abstractmethod
from typing import Optional, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from typing import List, Optional, Union
from mmengine import ConfigDict
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from scipy.optimize import linear_sum_assignment

# from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps, bbox_xyxy_to_cxcywh
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmengine.registry import TASK_UTILS



@TASK_UTILS.register_module()
class GBLDDETROrderPtsCost(BaseMatchCost):
    def __init__(self, weight: Union[float, int] = 1) -> None:
        super().__init__(weight=weight)
        self.name = "GBLDDETROrderPtsCost"

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``scores`` inside is
                predicted classification logits, of shape
                (num_queries, num_class).
            gt_instances (:obj:`InstanceData`): ``labels`` inside should have
                shape (num_gt, ).
            img_meta (Optional[dict]): _description_. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        # dict(type='OrderedPtsL1Cost',
        #      weight=5)
        # self.pts_cost(pts_pred_interpolated, normalized_gt_pts)
        # pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        # pts_cost, order_index = torch.min(pts_cost_ordered, 2)  # 求出order里最小的一个 [50, 8]

        pred_pts = pred_instances.pts  # num_query, num_pts, 2
        # gt_pts = gt_instances.gt_line_instances.shift_fixed_num_sampled_points_v2
        gt_pts = gt_instances.gt_shift_points
        num_gts, num_orders, num_pts, num_coords = gt_pts.shape

        pred_pts = pred_pts.view(pred_pts.size(0), -1)
        gt_pts = gt_pts.flatten(2).view(num_gts * num_orders, -1)
        pts_cost = torch.cdist(pred_pts, gt_pts, p=1)    # num_query, num_gts * num_orders

        pts_cost = pts_cost.view(pred_pts.size(0), num_gts, num_orders)
        pts_cost, order_index = torch.min(pts_cost, 2)   # 得出pst_cost为instance-cost, order_index为具体的某个order
        return pts_cost, order_index

        # pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        # pts_cost, order_index = torch.min(pts_cost_ordered, 2)  # 求出order里最小的一个 [50, 8]

        # pred_scores = pred_scores.softmax(-1)
        # cls_cost = -pred_scores[:, gt_labels]

        # num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # # import pdb;pdb.set_trace()
        # bbox_pred = bbox_pred.view(bbox_pred.size(0),-1)
        # gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders,-1)
        # bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        # return bbox_cost * self.weight

        # return cls_cost * self.weight
        return 1


@TASK_UTILS.register_module()
class GBLDDETRHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the one-to-one
    matching, the un-matched are treated as backgrounds. Thus each query
    prediction will be assigned with `0` or a positive integer indicating the
    ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
    """

    # def __init__(
    #     self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
    #                              ConfigDict]
    # ) -> None:
    #
    #     if isinstance(match_costs, dict):
    #         match_costs = [match_costs]
    #     elif isinstance(match_costs, list):
    #         assert len(match_costs) > 0, \
    #             'match_costs must not be a empty list.'
    #
    #     self.match_costs = [
    #         TASK_UTILS.build(match_cost) for match_cost in match_costs
    #     ]

    def __init__(self,
                 cls_cost=dict(type='mmdet.ClassificationCost', weight=1.),
                 reg_cost=dict(type='mmdet.BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='mmdet.IoUCost', weight=0.0),
                 pts_cost=dict(type='GBLDDETROrderPtsCost', weight=1.0),):

        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)
        self.pts_cost = TASK_UTILS.build(pts_cost)


    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places. It may includes ``masks``, with shape
                (n, h, w) or (n, l).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                ``labels``, with shape (k, ) and ``masks``, with shape
                (k, h, w) or (k, l).
            img_meta (dict): Image information.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. compute weighted cost
        cls_cost = self.cls_cost(pred_instances=pred_instances,
                                 gt_instances=gt_instances,
                                 img_meta=img_meta)

        reg_cost = self.reg_cost(pred_instances=pred_instances,
                                 gt_instances=gt_instances,
                                 img_meta=img_meta)

        iou_cost = self.iou_cost(pred_instances=pred_instances,
                                 gt_instances=gt_instances,
                                 img_meta=img_meta)

        pts_cost, order_index = self.pts_cost(
                                 pred_instances=pred_instances,
                                 gt_instances=gt_instances,
                                 img_meta=img_meta)  # pts_pred_interpolated[50， 20， 2], normalized_gt_pts[8， 19， 20， 2]

        cost = cls_cost + reg_cost + iou_cost + pts_cost
        # pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        # pts_cost, order_index = torch.min(pts_cost_ordered, 2)  # 求出order里最小的一个 [50, 8]


        #
        # cost_list = []
        # for match_cost in self.match_costs:
        #     cost = match_cost(
        #         pred_instances=pred_instances,
        #         gt_instances=gt_instances,
        #         img_meta=img_meta)
        #     cost_list.append(cost)
        # cost = torch.stack(cost_list).sum(dim=0)

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels), order_index