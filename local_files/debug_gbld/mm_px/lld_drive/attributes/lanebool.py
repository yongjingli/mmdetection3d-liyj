import numpy as np
from xpilot_lightning.machine_learning.tasks.base_linear.attributes.base import BaseAttribute
from xpilot_lightning.machine_learning.network_modules.losses.single_bit_loss import (
    single_bits_loss,
)
from xpilot_lightning.utilities.operations.sigmoid import sigmoid
from xpilot_lightning.utilities.operations.probability_decoder import probability_decoder
from xpilot_vision.tasks.lld2lightning.attributes.builder import LANESUBATTRIBUTES


@LANESUBATTRIBUTES.register_module()
class LaneBoolAttribute(BaseAttribute):
    def __init__(self, global_config, task_config, attribute_config):
        super().__init__(global_config, task_config, attribute_config)
        self.feature_names = self.task_config.bool_features_lane
        assert len(self.feature_names) == self.bits_number

    def label_to_vectors(self, label, augmentations, metadata, vectors, masks, label_processor):
        attribute_vector = np.zeros(self.bits_number)
        attribute_mask = np.zeros(self.bits_number)
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in self.task_config.ignore_feature_list:
                continue
            if feature_name == "exist":
                attribute_vector[i] = 1.0
                attribute_mask[i] = 1.0
            elif feature_name in label:
                attribute_vector[i] = 1.0 if label[feature_name] else 0.0
                attribute_mask[i] = 1.0
                if feature_name in ["merge2left", "merge2right"]:
                    attribute_mask[i] = (
                        1.0 + self.task_config.force_merge_pos_weight * attribute_vector[i]
                    )
            else:
                attribute_vector[i] = self.task_config.ignore_value
                attribute_mask[i] = 0.0
        vectors[self.bits_range] = attribute_vector
        masks[self.bits_range] = attribute_mask
        return vectors, masks

    def vectors_to_dict(self, vectors, metadata, is_gt, result):
        attribute_vector = vectors[self.bits_range]
        for i, feature_name in enumerate(self.feature_names):
            if is_gt:
                feature_prob = attribute_vector[i]
            else:
                feature_prob = sigmoid(attribute_vector[i])
            if is_gt and feature_prob == self.task_config.ignore_value:
                continue
            feature_bool = (feature_prob > 0.5).item()
            result[feature_name] = [feature_bool, [feature_prob]]
        return result

    def loss(self, preds, trues, masks):
        loss = single_bits_loss(
            preds[:, self.bits_range], trues[:, self.bits_range], masks[:, self.bits_range], False
        )
        return {"loss_lanebool": self.bits_weights * loss / (preds.shape[0] * self.bits_number)}

    def visualize(self, image, processed_dict, scale):
        return image
