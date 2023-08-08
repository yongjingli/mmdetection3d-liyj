import numpy as np
import cv2
from xpilot_lightning.machine_learning.network_modules.losses.multi_bits_loss import multi_bits_loss
from xpilot_lightning.machine_learning.tasks.base_linear.attributes.base import BaseAttribute
from xpilot_lightning.utilities.operations.one_hot_decoder import one_hot_decoder
from xpilot_lightning.utilities.operations.softmax import softmax
from xpilot_vision.tasks.lld2lightning.attributes.builder import LLD2LIGHTNINGATTRIBUTES


@LLD2LIGHTNINGATTRIBUTES.register_module()
class GlobalInfoAttribute(BaseAttribute):
    def __init__(self, global_config, task_config, attribute_config):
        super().__init__(global_config, task_config, attribute_config)
        self.global_info = self.task_config.global_info
        assert len(self.global_info) == self.bits_number

    def label_to_vectors(self, label, augmentations, metadata, vectors, masks, label_processor):
        exist_idx = self.task_config.attributes["Lane"].lanebool_index("exist")
        global_label = np.zeros(self.bits_number)
        if label["center"]["exist"]:
            global_label[self.global_info.index("normal")] = 1.0
        elif label["assist1"]["exist"]:
            global_label[self.global_info.index("split")] = 1.0
        elif label["fork1"]["exist"]:
            global_label[self.global_info.index("fork")] = 1.0
        elif label["wide"]["exist"]:
            global_label[self.global_info.index("wide")] = 1.0
        else:
            global_label[self.global_info.index("no_lane")] = 1.0
        vectors[self.bits_range] = global_label
        masks[self.bits_range] = 1.0
        return vectors, masks

    def vectors_to_dict(self, vectors, metadata, is_gt, result):
        pred_prob = vectors[self.bits_range]
        if not is_gt:
            pred_prob = softmax(pred_prob)
        global_case = self.global_info[one_hot_decoder(pred_prob)]
        result["global"] = [global_case, pred_prob.tolist()]
        return result

    def loss(self, preds, trues, masks):
        loss = multi_bits_loss(
            preds[:, self.bits_range],
            trues[:, self.bits_range],
            masks[:, self.bits_range],
            False
        )
        return {
            "loss_global": self.bits_weights * loss / preds.shape[0]
        }

    def visualize(self, image, processed_dict, scale):
        global_info = processed_dict["global"][0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        image = cv2.putText(
            image,
            global_info[0],
            (0,90),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return image
