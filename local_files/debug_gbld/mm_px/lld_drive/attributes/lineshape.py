import numpy as np
from xpilot_vision.tasks.lld2lightning.attributes.builder import LANESUBATTRIBUTES
from xpilot_lightning.machine_learning.tasks.base_linear.attributes.base import BaseAttribute
from xpilot_vision.tasks.lld2lightning.attributes.utils import interpolate_x, finetune_row_end
from xpilot_lightning.machine_learning.network_modules.losses.smooth_l1_loss import smooth_l1_loss
import scipy


@LANESUBATTRIBUTES.register_module()
class LineShapeAttribute(BaseAttribute):
    def __init__(self, global_config, task_config, attribute_config):
        super().__init__(global_config, task_config, attribute_config)
        self.num_points_each_line = self.task_config.num_points_each_line
        self.line_names = self.task_config.line_names
        self.y_coords = (
            self.task_config.start_y
            - np.arange(self.num_points_each_line) * self.task_config.step_y
        )
        if self.num_points_each_line == 32:
            self.point_weights = [1] * 8 + [2] * 6 + [4] * 6 + [8] * 6 + [16] * 6
        elif self.num_points_each_line == 64:
            self.point_weights = [1] * 16 + [2] * 12 + [4] * 12 + [8] * 12 + [16] * 12
        else:
            raise ValueError("Invalid number of points per line")
        self.point_weights = [i / sum(self.point_weights) * 48 for i in self.point_weights]
        assert (
            self.num_points_each_line + 2
        ) * 2 == self.bits_number, "Invalid number of bits for line shape"

    def encode_single_line_scipy(self, xys):
        xys = np.array(xys)
        f = scipy.interpolate.interp1d(xys[:, 1], xys[:, 0], bounds_error=False, assume_sorted=True)
        x_coords = f(self.y_coords)
        valid_idxs = np.where(~np.isnan(x_coords))[0]
        if len(valid_idxs) == 0:
            return (
                np.zeros(self.num_points_each_line),
                np.zeros(self.num_points_each_line),
                0,
                0,
                False,
            )
        y_start = self.y_coords[valid_idxs[-1]]
        y_end = self.y_coords[valid_idxs[0]]
        x_masks = np.array(self.point_weights)
        x_masks[np.isnan(x_coords)] = 0.0
        x_coords[np.isnan(x_coords)] = 0.0
        return x_coords, x_masks, y_start, y_end, True

    def encode_single_line(self, xys):
        x_coords = np.zeros(self.num_points_each_line)
        x_masks = np.zeros(self.num_points_each_line)
        min_value, max_value = xys[0][1], xys[-1][1]
        y_start = max_value
        y_end = min_value
        has_point = False
        for i, target_y in enumerate(self.y_coords):
            if target_y < min_value or target_y > max_value:
                continue
            x = interpolate_x(xys, target_y)
            x_coords[i] = x
            if (
                0 < x < self.global_config.image_width
                and 0 < target_y < self.global_config.image_height
            ):
                x_masks[i] = self.point_weights[i]
                y_start = min(y_start, target_y)
                y_end = max(y_end, target_y)
                has_point = True
        return x_coords, x_masks, y_start, y_end, has_point

    def label_to_vectors(self, label, augmentations, metadata, vectors, masks, label_processor):
        attribute_vector = np.zeros(self.bits_number)
        attribute_mask = np.zeros(self.bits_number)
        for j, line_name in enumerate(self.line_names):
            offset = j * (2 + self.num_points_each_line)
            if line_name in label:
                # encode y_start, y_end
                xys = [(pts["x"], pts["y"]) for pts in label[line_name]["points"]]
                xys = sorted(xys, key=lambda tup: tup[1])
                # encode x coordinate
                x_coord, x_mask, y_start, y_end, has_point = self.encode_single_line(xys)
                attribute_vector[offset + 2 : offset + 2 + self.num_points_each_line] = x_coord
                attribute_mask[offset + 2 : offset + 2 + self.num_points_each_line] = x_mask
                if has_point:
                    y_end = finetune_row_end(
                        y_end, xys, self.global_config.image_width, self.global_config.image_height
                    )
                    attribute_mask[offset : offset + 2] = 1.0
                attribute_vector[offset : offset + 2] = y_start, y_end
                if y_end - y_start < 2:
                    attribute_mask[offset + 2 : offset + 2 + self.num_points_each_line] = 0.0
        vectors[self.bits_range] = attribute_vector
        masks[self.bits_range] = attribute_mask
        return vectors, masks

    def vectors_to_dict(self, vectors, metadata, is_gt, result, scale=2):
        attribute_vector = vectors[self.bits_range]
        for j, line_name in enumerate(self.line_names):
            offset = j * (2 + self.num_points_each_line)
            line_vector = attribute_vector[offset : offset + (2 + self.num_points_each_line)]
            pred_y_start = line_vector[0]
            pred_y_end = line_vector[1]
            result[line_name]["y_start"] = [pred_y_start]
            result[line_name]["y_end"] = [pred_y_end]
            points = []
            for target_y, pred_x in zip(self.y_coords, line_vector[2:]):
                if (pred_y_start <= target_y <= pred_y_end) and (
                    0 <= pred_x < self.global_config.image_width
                ):
                    target_y *= scale
                    pred_x *= scale
                    points.append({"x": pred_x, "y": target_y})
            result[line_name]["points"] = points
        return result

    def loss(self, preds, trues, masks):
        loss_position = smooth_l1_loss(
            preds[:, self.bits_range],
            trues[:, self.bits_range],
            masks[:, self.bits_range],
            False,
        )
        loss_gradient = 0
        for j, line_name in enumerate(self.line_names):
            offset = j * (2 + self.num_points_each_line)
            start_idx = self.bits_range.start + offset + 2
            end_idx = start_idx + self.num_points_each_line
            loss_gradient += smooth_l1_loss(
                preds[:, start_idx + 1 : end_idx] - preds[:, start_idx : end_idx - 1],
                trues[:, start_idx + 1 : end_idx] - trues[:, start_idx : end_idx - 1],
                (masks[:, start_idx + 1 : end_idx] * masks[:, start_idx : end_idx - 1]) ** 0.5,
                False,
            )
        return {
            "loss_lineposition": self.bits_weights["position"]
            * loss_position
            / (preds.shape[0] * self.bits_number),
            "loss_linegradient": self.bits_weights["gradient"]
            * loss_gradient
            / (preds.shape[0] * 2 * self.num_points_each_line),
        }

    def visualize(self, image, processed_dict, scale):
        return image
