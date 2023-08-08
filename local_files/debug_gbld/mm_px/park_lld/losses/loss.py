import numpy as np
import torch
import torch.nn.functional as F

from xpilot_vision.tasks.base.losses.loss import BaseLoss
from xpilot_lightning.machine_learning.tasks.builder import LOSSES


@LOSSES.register_module
class AP_LLDLoss(BaseLoss):
    def __init__(self, global_config, task_config):
        BaseLoss.__init__(self, global_config, task_config)
        self.pred_channels = task_config.pred_channels
        self.gt_channels = task_config.gt_channels
        self.mask_channels = task_config.mask_channels
        self.register_buffer("line_weight", torch.tensor(task_config.line_weight))
        self.register_buffer("arrow_weight", torch.tensor(task_config.arrow_weight))
        self.focal_loss_gamma = task_config.focal_loss_gamma
        self.set_distance_decay(task_config)
        self.pull_margin = task_config.pull_margin
        self.push_margin = task_config.push_margin
        self.weight_list = task_config.weight_list

    def set_distance_decay(self, config):
        if len(config.crop_size) == 4 and len(config.decay_rate) == 4:
            self.use_distance_decay = True
            top, bottom, left, right = config.crop_size
            distance_decay = np.ones(bottom - top, dtype=np.float)
            start_y, end_y, min_d, max_d = config.decay_rate
            distance_decay[start_y:end_y] = np.linspace(
                min_d, max_d, end_y - start_y, endpoint=True
            )
            distance_decay = distance_decay[None, None, :, None].repeat(right - left, axis=3)
            self.register_buffer("distance_decay", torch.from_numpy(distance_decay))
        else:
            self.use_distance_decay = False

    def forward(self, y_hat, y, mask):
        zero = torch.tensor(0.0).to(y_hat)
        loss = {
            "Regression-position": zero.clone(),
            "Regression-cluster": zero.clone(),
            "Classify-type": zero.clone()
        }
        confidence_loss = zero.clone()
        offset_loss = zero.clone()
        embedding_loss = zero.clone()
        arrow_offset_loss = zero.clone()
        arrow_embedding_loss = zero.clone()
        arrow_type_loss = zero.clone()
        arrow_vertex_type_loss = zero.clone()

        mask = mask.to(torch.bool)
        id_m = 0
        start_m, end_m = self.mask_channels[id_m], self.mask_channels[id_m + 1]
        line_ignore_mask = mask[:, start_m:end_m, :, :]
        id_m += 1
        start_m, end_m = self.mask_channels[id_m], self.mask_channels[id_m + 1]
        line_mask = mask[:, start_m:end_m, :, :]
        id_m += 1
        start_m, end_m = self.mask_channels[id_m], self.mask_channels[id_m + 1]
        arrow_ignore_mask = mask[:, start_m:end_m, :, :]
        id_m += 1
        start_m, end_m = self.mask_channels[id_m], self.mask_channels[id_m + 1]
        arrow_mask = mask[:, start_m:end_m, :, :]

        # Line confidence loss
        if line_ignore_mask.sum() > 0:
            id_p, id_g = 0, 0
            start_p, end_p = self.pred_channels[id_p], self.pred_channels[id_p + 1]
            pred_conf = y_hat[:, start_p:end_p, :, :]
            start_g, end_g = self.gt_channels[id_g], self.gt_channels[id_g + 1]
            gt_conf = y[:, start_g:end_g, :, :]
            bce_loss = F.binary_cross_entropy_with_logits(pred_conf, gt_conf, reduction="none")
            p = torch.exp(-1.0 * bce_loss)
            confidence_loss = F.binary_cross_entropy_with_logits(
                pred_conf, gt_conf, reduction="none", pos_weight=self.line_weight
            )
            confidence_loss = torch.pow(1.0 - p, self.focal_loss_gamma) * confidence_loss
            if self.use_distance_decay:
                distance_decay = (
                    self.distance_decay.clone().to(confidence_loss).expand_as(confidence_loss)
                )
                distance_decay[line_mask == 0] = 1.0
                confidence_loss = distance_decay * confidence_loss
            confidence_loss = confidence_loss[line_ignore_mask].mean()

            # Line confidence loss (Dice Loss)
            dice_loss = 0.0
            eps = 1.0e-5
            dice_loss_weight = 5.0
            pred_conf = torch.sigmoid(pred_conf)
            for pred, gt, msk in zip(pred_conf, gt_conf, line_ignore_mask):
                if msk.sum() == 0:
                    continue
                pred = pred[msk]
                gt = gt[msk]
                dice_loss += 1 - ((pred * gt).sum() + eps) / (pred.pow(2).sum() + gt.pow(2).sum() + eps)
                pred = 1 - pred
                gt = 1 - gt
                dice_loss += - ((pred * gt).sum() + eps) / (pred.pow(2).sum() + gt.pow(2).sum() + eps)
            confidence_loss += dice_loss_weight * dice_loss / y_hat.shape[0]
        else:
            id_p, id_g = 0, 0
            confidence_loss = zero.clone()

        if line_mask.sum() > 0:
            # Line offset loss
            id_p, id_g = id_p + 1, id_g + 1
            start_p, end_p = self.pred_channels[id_p], self.pred_channels[id_p + 1]
            pred_offset = y_hat[:, start_p:end_p, :, :]
            pred_offset = torch.sigmoid(pred_offset)
            start_g, end_g = self.gt_channels[id_g], self.gt_channels[id_g + 1]
            gt_offset = y[:, start_g:end_g, :, :]
            offset_loss = F.mse_loss(pred_offset, gt_offset, reduction="none")
            offset_loss = offset_loss[line_mask.expand(-1, 2, -1, -1)]
            offset_loss = offset_loss.mean()

            # Line embedding loss
            id_p, id_g = id_p + 1, id_g + 1
            start_p, end_p = self.pred_channels[id_p], self.pred_channels[id_p + 1]
            pred_embed = y_hat[:, start_p:end_p, :, :]
            start_g, end_g = self.gt_channels[id_g], self.gt_channels[id_g + 1]
            gt_embed = y[:, start_g:end_g, :, :]
            num_chn = end_p - start_p
            pull_loss, push_loss = 0.0, 0.0
            for pred, gt, msk in zip(pred_embed, gt_embed, line_mask):
                N = msk.sum()
                if N == 0:
                    continue
                gt = gt[msk]
                gt_square = torch.zeros(N, N).to(y_hat)
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
            embedding_loss = (pull_loss + push_loss) / y_hat.shape[0]
        else:
            id_p, id_g = id_p + 2, id_g + 2

        # Arrow confidence loss
        if arrow_ignore_mask.sum() > 0:
            id_p, id_g = id_p + 1, id_g + 1
            start_p, end_p = self.pred_channels[id_p], self.pred_channels[id_p + 1]
            pred_conf = y_hat[:, start_p:end_p, :, :]
            start_g, end_g = self.gt_channels[id_g], self.gt_channels[id_g + 1]
            gt_conf = y[:, start_g:end_g, :, :]
            bce_loss = F.binary_cross_entropy_with_logits(pred_conf, gt_conf, reduction="none")
            p = torch.exp(-1.0 * bce_loss)
            arrow_confidence_loss = F.binary_cross_entropy_with_logits(
                pred_conf, gt_conf, reduction="none", pos_weight=self.arrow_weight
            )
            arrow_confidence_loss = torch.pow(1.0 - p, self.focal_loss_gamma) * arrow_confidence_loss
            arrow_confidence_loss = arrow_confidence_loss[arrow_ignore_mask].mean()
        else:
            id_p, id_g = id_p + 1, id_g + 1
            arrow_confidence_loss = zero.clone()

        if arrow_mask.sum() > 0:
            # Arrow offset loss
            id_p, id_g = id_p + 1, id_g + 1
            start_p, end_p = self.pred_channels[id_p], self.pred_channels[id_p + 1]
            pred_offset = y_hat[:, start_p:end_p, :, :]
            pred_offset = torch.sigmoid(pred_offset)
            start_g, end_g = self.gt_channels[id_g], self.gt_channels[id_g + 1]
            gt_offset = y[:, start_g:end_g, :, :]
            arrow_offset_loss = F.mse_loss(pred_offset, gt_offset, reduction="none")
            arrow_offset_loss = arrow_offset_loss[arrow_mask.expand(-1, 2, -1, -1)]
            arrow_offset_loss = arrow_offset_loss.mean()

            # Arrow embedding loss
            id_p, id_g = id_p + 1, id_g + 1
            start_p, end_p = self.pred_channels[id_p], self.pred_channels[id_p + 1]
            pred_embed = y_hat[:, start_p:end_p, :, :]
            start_g, end_g = self.gt_channels[id_g], self.gt_channels[id_g + 1]
            gt_embed = y[:, start_g:end_g, :, :]
            num_chn = end_p - start_p
            pull_loss, push_loss = 0.0, 0.0
            for pred, gt, msk in zip(pred_embed, gt_embed, arrow_mask):
                N = msk.sum()
                if N == 0:
                    continue
                gt = gt[msk]
                gt_square = torch.zeros(N, N).to(y_hat)
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
            arrow_embedding_loss = (pull_loss + push_loss) / y_hat.shape[0]

            # Arrow type loss
            id_p, id_g = id_p + 1, id_g + 1
            start_p, end_p = self.pred_channels[id_p], self.pred_channels[id_p + 1]
            pred_type = y_hat[:, start_p:end_p, :, :]
            start_g, end_g = self.gt_channels[id_g], self.gt_channels[id_g + 1]
            gt_type = y[:, start_g:end_g, :, :].to(torch.long).squeeze(dim=1)
            arrow_type_loss = F.cross_entropy(pred_type, gt_type, reduction="none")
            arrow_type_loss = arrow_type_loss[arrow_mask.squeeze(dim=1)].mean()

            # Arrow vertex type loss
            id_p, id_g = id_p + 1, id_g + 1
            start_p, end_p = self.pred_channels[id_p], self.pred_channels[id_p + 1]
            pred_type = y_hat[:, start_p:end_p, :, :]
            start_g, end_g = self.gt_channels[id_g], self.gt_channels[id_g + 1]
            gt_type = y[:, start_g:end_g, :, :]
            bce_loss = F.binary_cross_entropy_with_logits(pred_type, gt_type, reduction="none")
            p = torch.exp(-1.0 * bce_loss)
            arrow_vertex_type_loss = torch.pow(1.0 - p, self.focal_loss_gamma) * bce_loss
            arrow_vertex_type_loss = arrow_vertex_type_loss[arrow_mask].mean()

        loss["Classify-confidence"] = (
            self.weight_list[0] * confidence_loss + self.weight_list[3] * arrow_confidence_loss
        )
        loss["Regression-position"] = (
            self.weight_list[1] * offset_loss + self.weight_list[4] * arrow_offset_loss
        )
        loss["Regression-cluster"] = (
            self.weight_list[2] * embedding_loss + self.weight_list[5] * arrow_embedding_loss
        )
        loss["Classify-type"] = (
            self.weight_list[6] * arrow_type_loss + self.weight_list[7] * arrow_vertex_type_loss
        )
        return sum(loss.values()), loss, None
