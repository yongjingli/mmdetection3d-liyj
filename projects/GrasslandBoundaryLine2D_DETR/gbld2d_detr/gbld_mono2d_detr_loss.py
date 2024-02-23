import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

@MODELS.register_module()
class GbldDetrSegLoss(nn.Module):
    def __init__(self,
                 focal_loss_gamma=2.0,
                 alpha=0.25,
                 loss_weight=1.0,
                 use_dist_weight=False,
                 max_dist_weight=5.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(GbldDetrSegLoss, self).__init__()
        self.focal_loss_gamma = focal_loss_gamma        # focal_loss_gamma
        self.alpha = alpha                              # 暂时没用
        self.loss_weight = loss_weight                  # 该loss的权重
        self.use_dist_weight = use_dist_weight          # 是否使用距离加权, y越大权重越大
        self.max_dist_weight = max_dist_weight          # 最大y的权重

    def forward(self,
                seg_pred,
                gt_seg,
                gt_ignore_mask,):

        # confidence loss
        gt_ignore_mask = gt_ignore_mask > 0
        if gt_ignore_mask.sum() > 0:
            bce_loss = F.binary_cross_entropy_with_logits(seg_pred, gt_seg, reduction="none")
            p = torch.exp(-1.0 * bce_loss)
            seg_loss = F.binary_cross_entropy_with_logits(
                seg_pred, gt_seg, reduction="none", pos_weight=torch.tensor(5.0)
            )
            seg_loss = torch.pow(1.0 - p, self.focal_loss_gamma) * seg_loss

            # # 加强对近处的loss
            if self.use_dist_weight:
                N, C, H, W = seg_loss.shape
                dist_weight_col = torch.arange(1, self.max_dist_weight, (self.max_dist_weight - 1) / H)
                dist_weight = dist_weight_col.repeat(1, W).reshape(W, H).transpose(1, 0)
                dist_weight = dist_weight.to(seg_loss.device)
                dist_weight.requires_grad = False
                seg_loss = seg_loss * dist_weight

            seg_loss = seg_loss[gt_ignore_mask].mean()

            # Line confidence loss (Dice Loss)
            dice_loss = 0.0
            eps = 1.0e-5
            dice_loss_weight = 5.0
            # dice_loss_weight = 0.0
            seg_pred = torch.sigmoid(seg_pred)
            for pred, gt, msk in zip(seg_pred, gt_seg, gt_ignore_mask):
                if msk.sum() == 0:
                    continue
                pred = pred[msk]
                gt = gt[msk]
                positive_loss = 1 - ((pred * gt).sum() + eps) / (pred.pow(2).sum() + gt.pow(2).sum() + eps)
                dice_loss += positive_loss
                pred = 1 - pred
                gt = 1 - gt
                negative_loss = - ((pred * gt).sum() + eps) / (pred.pow(2).sum() + gt.pow(2).sum() + eps)
                # negative_loss = ((pred * gt).sum() + eps) / (pred.pow(2).sum() + gt.pow(2).sum() + eps)
                dice_loss += negative_loss

                # dice_loss = max(dice_loss, 0)
                # print("dice_loss:", dice_loss, "positive_loss:", positive_loss, "negative_loss:", negative_loss)

            seg_loss += dice_loss_weight * dice_loss / seg_pred.shape[0]
        else:
            seg_loss = torch.tensor(0.0)
        seg_loss = seg_loss * self.loss_weight
        return seg_loss


@MODELS.register_module()
class GbldDetrOffsetLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0):
        super(GbldDetrOffsetLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                offset_pred,
                gt_offset,
                gt_foreground_mask,):
        # offset losses
        gt_foreground_mask = gt_foreground_mask > 0
        # gt_foreground_expand_mask = gt_foreground_expand_mask > 0
        if gt_foreground_mask.sum() > 0:
            # offset loss
            # pred_offset_x = torch.sigmoid(pred_offset_x)
            # pred_offset_y = torch.sigmoid(pred_offset_y)
            #
            # offset_loss_x = F.mse_loss(pred_offset_x, gt_offset_x, reduction="none")
            # offset_loss_x = offset_loss_x[gt_foreground_mask]
            #
            # offset_loss_y = F.mse_loss(pred_offset_y, gt_offset_y, reduction="none")
            # offset_loss_y = offset_loss_y[gt_foreground_mask]
            # offset_loss = offset_loss_x.mean() + offset_loss_y.mean()

            pred_offset = torch.sigmoid(offset_pred)
            offset_loss = F.mse_loss(pred_offset, gt_offset, reduction="none")
            offset_loss = offset_loss[gt_foreground_mask.expand(-1, 2, -1, -1)]
            offset_loss = offset_loss.mean()

        else:
            offset_loss = torch.tensor(0.0)
        offset_loss = offset_loss * self.loss_weight

        return offset_loss


@MODELS.register_module()
class GbldDetrEmbLoss(nn.Module):
    def __init__(self,
                pull_margin=0.5,
                push_margin=1.0,
                loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(GbldDetrEmbLoss, self).__init__()
        self.push_margin = push_margin
        self.pull_margin = pull_margin
        self.loss_weight = loss_weight

    def forward(self,
                pred_emb,
                gt_line_index,
                gt_foreground_mask,):

        # 对于seg_emb, 为pred_emb, gt_line_index, gt_foreground_mask
        # 对于connect_emb, pred_emb_id, gt_line_id, gt_foreground_mask

        # embedding_losses
        embedding_loss = torch.tensor(0.0)
        embedding_loss = embedding_loss.to(pred_emb.device)
        if gt_foreground_mask.sum() > 0:
            # num_chn = end_p - start_p
            num_chn = 1

            pull_loss, push_loss = 0.0, 0.0
            for pred, gt, msk in zip(pred_emb, gt_line_index, gt_foreground_mask):
                N = msk.sum()
                if N == 0:
                    continue
                gt = gt[msk]
                gt_square = torch.zeros(N, N)
                for i in range(N):
                    gt_row = gt.clone()
                    gt_row[gt == gt[i]] = 1
                    gt_row[gt != gt[i]] = 2
                    gt_square[i] = gt_row
                msk = msk.expand(num_chn, -1, -1)
                pred = pred[msk]
                pred_row = pred.view(num_chn, 1, N).expand(num_chn, N, N)
                pred_col = pred.view(num_chn, N, 1).expand(num_chn, N, N)
                pred_sqrt = torch.norm(pred_col - pred_row, dim=0)
                pred_dist = pred_sqrt[gt_square == 1] - self.pull_margin
                pred_dist[pred_dist < 0] = 0
                pull_loss += pred_dist.mean()
                if gt_square.max() == 2:
                    pred_dist = self.push_margin - pred_sqrt[gt_square == 2]
                    pred_dist[pred_dist < 0] = 0
                    push_loss += pred_dist.mean()
            embedding_loss += (pull_loss + push_loss) / pred_emb.shape[0]

        else:
            embedding_loss += torch.tensor(0.0)

        embedding_loss = embedding_loss * self.loss_weight
        return embedding_loss


@MODELS.register_module()
class GbldDetrClsLoss(nn.Module):
    def __init__(self,
                 num_classes=1,
                 loss_weight=1.0):
        super(GbldDetrClsLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.cls_loss = torch.nn.BCELoss(reduction='none')

    def forward(self,
                pred_cls,
                gt_line_cls,
                gt_foreground_mask,):

        if gt_foreground_mask.sum() > 0:
            pred_cls = torch.sigmoid(pred_cls)
            cls_loss = self.cls_loss(pred_cls, gt_line_cls)
            cls_loss = cls_loss[gt_foreground_mask.expand(-1, self.num_classes, -1, -1)]
            cls_loss = cls_loss.mean()

        else:
            cls_loss = torch.tensor(0.0)
        cls_loss = cls_loss * self.loss_weight
        return cls_loss


@MODELS.register_module()
class GbldDetrOrientLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0):
        super(GbldDetrOrientLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                orient_pred,
                gt_orient,
                gt_orient_mask,):
        # offset losses
        gt_orient_mask = gt_orient_mask > 0
        # sin和cos同时不为0的位置为mask
        # gt_orient_mask = torch.bitwise_and(gt_orient[:, 0:1, :, :] != 0, gt_orient[:, 1:2, :, :] != 0)

        # gt_foreground_expand_mask = gt_foreground_expand_mask > 0
        if gt_orient_mask.sum() > 0:
            # offset loss
            # pred_offset_x = torch.sigmoid(pred_offset_x)
            # pred_offset_y = torch.sigmoid(pred_offset_y)
            #
            # offset_loss_x = F.mse_loss(pred_offset_x, gt_offset_x, reduction="none")
            # offset_loss_x = offset_loss_x[gt_foreground_mask]
            #
            # offset_loss_y = F.mse_loss(pred_offset_y, gt_offset_y, reduction="none")
            # offset_loss_y = offset_loss_y[gt_foreground_mask]
            # offset_loss = offset_loss_x.mean() + offset_loss_y.mean()

            # 将sin和cos合并在一起计算loss
            orient_pred = torch.sigmoid(orient_pred) * 2 - 1
            orient_loss = F.mse_loss(orient_pred, gt_orient, reduction="none")
            orient_loss = orient_loss[gt_orient_mask.expand(-1, 2, -1, -1)]
            orient_loss = orient_loss.mean()

        else:
            orient_loss = torch.tensor(0.0)
        orient_loss = orient_loss * self.loss_weight

        return orient_loss


# https://github.com/IrohXu/lanenet-lane-detection-pytorch
# https://github.com/harryhan618/LaneNet
# 采用高维度的聚类loss, 聚类完的后处理可以采用k-means和dbscan
# k-means, https://github.com/Wizaron/instance-segmentation-pytorch/blob/master/code/lib/prediction.py
# dbscan, https://zhuanlan.zhihu.com/p/77043965

@MODELS.register_module()
class GbldDetrDiscriminativeLoss(nn.Module):
    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 loss_weight=1.0):
        super(GbldDetrDiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_weight = loss_weight
        assert self.norm in [1, 2]
    def forward(self, input, target):
        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, embedding, seg_gt):
        # batch_size = embedding.shape[0]
        batch_size, embed_dim, H, W = embedding.shape

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i[0]]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(
                    F.relu(torch.norm(embedding_i - mean_i.reshape(embed_dim, 1), dim=0, p=self.norm) - self.delta_var) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)
                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2, p=self.norm)  # shape (num_lanes, num_lanes)
                # 这里的距离越大loss越小，对角线上的距离为0，需要设置为大于等于self.delta_dist的数据，保证该部分的loss为0
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype,
                                        device=dist.device) * self.delta_dist  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_dist) ** 2) / (num_lanes * (num_lanes - 1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        discriminative_loss = (var_loss + dist_loss + reg_loss) / batch_size
        discriminative_loss = discriminative_loss * self.loss_weight
        return discriminative_loss