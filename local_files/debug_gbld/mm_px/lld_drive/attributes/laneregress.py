import numpy as np
from xpilot_vision.tasks.lld2lightning.attributes.builder import LANESUBATTRIBUTES
from xpilot_lightning.machine_learning.tasks.base_linear.attributes.base import BaseAttribute
from xpilot_lightning.machine_learning.network_modules.losses.mse_loss import mse_loss


@LANESUBATTRIBUTES.register_module()
class LaneRegressAttribute(BaseAttribute):
    def __init__(self, global_config, task_config, attribute_config):
        super().__init__(global_config, task_config, attribute_config)
        self.feature_names = self.task_config.reg_features_lane
        assert len(self.feature_names) == self.bits_number

    def label_to_vectors(self, label, augmentations, metadata, vectors, masks, label_processor):
        attribute_vector = np.zeros(self.bits_number)
        attribute_mask = np.zeros(self.bits_number)
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in self.task_config.ignore_feature_list:
                continue
            if feature_name in label:
                attribute_vector[i] = label[feature_name]
                attribute_mask[i] = 1.0
            else:
                attribute_vector[i] = self.task_config.ignore_value
                attribute_mask[i] = 0.0
        vectors[self.bits_range] = attribute_vector
        masks[self.bits_range] = attribute_mask
        return vectors, masks

    def vectors_to_dict(self, vectors, metadata, is_gt, result):
        attribute_vector = vectors[self.bits_range]
        for i, feature_name in enumerate(self.feature_names):
            feature_logit = attribute_vector[i]
            if is_gt and feature_logit == self.task_config.ignore_value:
                continue
            result[feature_name] = [feature_logit, int(feature_logit+0.5)]
        return result

    def loss(self, preds, trues, masks):
        loss = mse_loss(
            preds[:, self.bits_range],
            trues[:, self.bits_range],
            masks[:, self.bits_range],
            False
        )
        return {
            "loss_laneregress": self.bits_weights * loss / (preds.shape[0] * self.bits_number)
        }

    def visualize(self, image, processed_dict, scale):
        return image
