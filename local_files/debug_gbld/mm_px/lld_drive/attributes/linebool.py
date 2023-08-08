import numpy as np
from xpilot_lightning.machine_learning.tasks.base_linear.attributes.base import BaseAttribute
from xpilot_lightning.machine_learning.network_modules.losses.single_bit_loss import (
    single_bits_loss,
)
from xpilot_lightning.utilities.operations.sigmoid import sigmoid
from xpilot_lightning.utilities.operations.probability_decoder import probability_decoder
from xpilot_vision.tasks.lld2lightning.attributes.builder import LANESUBATTRIBUTES


@LANESUBATTRIBUTES.register_module()
class LineBoolAttribute(BaseAttribute):
    def __init__(self, global_config, task_config, attribute_config):
        super().__init__(global_config, task_config, attribute_config)
        self.feature_names = self.task_config.bool_features_line
        self.line_names = self.task_config.line_names
        assert len(self.feature_names) * 2 == self.bits_number

    # TODO: special hack for solid ahead
    def label_to_vectors(self, label, augmentations, metadata, vectors, masks, label_processor):
        attribute_vector = np.zeros(self.bits_number)
        attribute_mask = np.zeros(self.bits_number)
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in self.task_config.ignore_feature_list:
                continue
            for j, line_name in enumerate(self.line_names):
                if feature_name == "line_exist":
                    attribute_vector[2 * i + j] = 1.0
                    attribute_mask[2 * i + j] = 1.0
                elif line_name in label and feature_name in label[line_name]:
                    gt = label[line_name][feature_name]
                    if isinstance(gt, str):
                        gt = gt.lower() == "true"
                    if not isinstance(gt, bool):
                        raise ValueError(f"Invalid groud truth value {gt}")
                    attribute_vector[2 * i + j] = 1.0 if gt else 0.0
                    attribute_mask[2 * i + j] = 1.0
                else:
                    attribute_vector[2 * i + j] = self.task_config.ignore_value
                    attribute_mask[2 * i + j] = 0.0
        vectors[self.bits_range] = attribute_vector
        masks[self.bits_range] = attribute_mask
        return vectors, masks

    def vectors_to_dict(self, vectors, metadata, is_gt, result):
        attribute_vector = vectors[self.bits_range]
        for i, feature_name in enumerate(self.feature_names):
            for j, line_name in enumerate(self.line_names):
                if is_gt:
                    feature_prob = attribute_vector[2 * i + j]
                else:
                    feature_prob = sigmoid(attribute_vector[i])
                if is_gt and feature_prob == self.task_config.ignore_value:
                    continue
                feature_bool = (feature_prob > 0.5).item()
                result[line_name][feature_name] = [feature_bool, [feature_prob]]
        return result

    def loss(self, preds, trues, masks):
        loss = single_bits_loss(
            preds[:, self.bits_range], trues[:, self.bits_range], masks[:, self.bits_range], False
        )
        return {"loss_linebool": self.bits_weights * loss / (preds.shape[0] * self.bits_number)}

    def visualize(self, image, processed_dict, scale):
        return image
