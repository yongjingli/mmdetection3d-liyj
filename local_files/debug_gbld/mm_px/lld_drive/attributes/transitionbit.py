import numpy as np
from xpilot_vision.tasks.lld2lightning.attributes.builder import LLD2LIGHTNINGATTRIBUTES
from xpilot_lightning.machine_learning.tasks.base_linear.attributes.base import BaseAttribute
from xpilot_lightning.machine_learning.network_modules.losses.single_bit_loss import (
    single_bits_loss,
)
from xpilot_lightning.utilities.operations.sigmoid import sigmoid
from xpilot_lightning.utilities.operations.probability_decoder import probability_decoder


@LLD2LIGHTNINGATTRIBUTES.register_module()
class TransitionBitAttribute(BaseAttribute):
    def __init__(self, global_config, task_config, attribute_config):
        super().__init__(global_config, task_config, attribute_config)

    def label_to_vectors(self, label, augmentations, metadata, vectors, masks, label_processor):
        attribute_label = np.zeros(self.bits_number)
        attribute_mask = np.zeros(self.bits_number)
        for position, lane_label in label.items():
            lane_id = self.task_config.pos2id[position]
            for i, line_name in enumerate(self.task_config.line_names):
                idx = self.task_config.attributes["Lane"].linebool_index(
                    "cross2noncross", line_name
                )
                attribute_label[2 * lane_id + i] = lane_label["label"][idx]
                attribute_mask[2 * lane_id + i] = lane_label["mask"][idx]
        vectors[self.bits_range] = attribute_label
        masks[self.bits_range] = attribute_mask
        return vectors, masks

    def vectors_to_dict(self, vectors, metadata, is_gt, result):
        attribute_vector = vectors[self.bits_range]
        for i in range(self.task_config.number_of_lanes):
            for j, line_name in enumerate(self.task_config.line_names):
                if is_gt:
                    feature_prob = attribute_vector[2 * i + j]
                else:
                    feature_prob = sigmoid(attribute_vector[i])
                if is_gt and feature_prob == self.task_config.ignore_value:
                    continue
                feature_bool = probability_decoder(feature_prob)
                result["lanes"][i][line_name]["cross2noncross"] = [feature_bool, [feature_prob]]
        return result

    def loss(self, preds, trues, masks):
        loss = single_bits_loss(
            preds[:, self.bits_range], trues[:, self.bits_range], masks[:, self.bits_range], False
        )
        return {"loss_transition": self.bits_weights * loss / (preds.shape[0] * self.bits_number)}

    def visualize(self, image, processed_dict, scale):
        return image
