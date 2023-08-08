import numpy as np
from xpilot_vision.tasks.lld2lightning.attributes.builder import LANESUBATTRIBUTES
from xpilot_lightning.machine_learning.tasks.base_linear.attributes.base import BaseAttribute
from xpilot_lightning.machine_learning.network_modules.losses.multi_bits_loss import multi_bits_loss
from xpilot_lightning.utilities.operations.one_hot_decoder import one_hot_decoder
from xpilot_lightning.utilities.operations.softmax import softmax


@LANESUBATTRIBUTES.register_module()
class LineTypeAttribute(BaseAttribute):
    def __init__(self, global_config, task_config, attribute_config):
        super().__init__(global_config, task_config, attribute_config)
        self.line_types = self.task_config.line_types
        self.name2id = dict(zip(self.line_types, range(len(self.line_types))))
        self.line_names = self.task_config.line_names
        assert len(self.line_types) * 2 == self.bits_number

    def label_to_vectors(self, label, augmentations, metadata, vectors, masks, label_processor):
        attribute_vector = np.zeros(self.bits_number)
        attribute_mask = np.zeros(self.bits_number)
        for j, line_name in enumerate(self.line_names):
            offset = j * len(self.line_types)
            line_label = label.get(line_name, {})
            line_type = line_label.get("line_type", None)
            if line_type is not None and line_type in self.name2id:
                attribute_vector[offset+self.name2id[line_type]] = 1.0
                attribute_mask[offset:offset+len(self.line_types)] = 1.0
        vectors[self.bits_range] = attribute_vector
        masks[self.bits_range] = attribute_mask
        return vectors, masks

    def vectors_to_dict(self, vectors, metadata, is_gt, result):
        attribute_vector = vectors[self.bits_range]
        for j, line_name in enumerate(self.line_names):
            offset = j * len(self.line_types)
            if is_gt:
                line_prob = attribute_vector[offset:offset +
                                             len(self.line_types)]
            else:
                line_prob = softmax(
                    attribute_vector[offset:offset + len(self.line_types)])
            if is_gt and line_prob.sum() == 0:
                continue
            line_category = self.line_types[one_hot_decoder(line_prob)]
            result[line_name]['line_type'] = [
                line_category, line_prob.tolist()]
        return result

    def loss(self, preds, trues, masks):
        loss = 0
        for j, line_name in enumerate(self.line_names):
            offset = j*len(self.line_types)
            start_idx = self.bits_range.start+offset
            end_idx = start_idx+len(self.line_types)
            loss += multi_bits_loss(
                preds[:, start_idx:end_idx],
                trues[:, start_idx:end_idx],
                masks[:, start_idx:end_idx],
                False
            )
        return {
            "loss_linetype": self.bits_weights*loss/(preds.shape[0]*len(self.line_names))
        }

    def visualize(self, image, processed_dict, scale):
        return image
