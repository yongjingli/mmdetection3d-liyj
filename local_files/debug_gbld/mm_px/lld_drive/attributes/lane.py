from collections import defaultdict
import numpy as np
import cv2
from xpilot_lightning.machine_learning.tasks.base.attributes.builder import build_attribute
from xpilot_lightning.machine_learning.tasks.base_linear.attributes.base import BaseAttribute
from xpilot_vision.tasks.lld2lightning.attributes.utils import (
    flip_attributes,
    extend_to_boundary,
    fix_label_typo,
)
from xpilot_vision.tasks.lld2lightning.attributes.builder import LLD2LIGHTNINGATTRIBUTES
from xpilot_vision.tasks.lld2lightning.attributes.builder import LANESUBATTRIBUTES


@LLD2LIGHTNINGATTRIBUTES.register_module()
class LaneAttribute(BaseAttribute):
    def __init__(self, global_config, task_config, attribute_config):
        super().__init__(global_config, task_config, attribute_config)
        self.global_config = global_config
        self.task_config = task_config
        self.attribute_config = attribute_config
        self.sub_attributes = self.build_subattributes()

        self.lane_bool_start = self.sub_attributes["LaneBool"].bits_range.start
        self.line_bool_start = self.sub_attributes["LineBool"].bits_range.start
        self.lane_regress_start = self.sub_attributes["LaneRegress"].bits_range.start
        self.lineshape_attribute = self.sub_attributes["LineShape"]

    def build_subattributes(self):
        current_digit = 0
        sub_attributes = {}
        for attribute_name, attribute_config in self.attribute_config["sub_attributes"].items():
            attribute_config["bits_range"] = range(
                current_digit, current_digit + attribute_config["bits_number"]
            )
            current_digit += attribute_config["bits_number"]
            sub_attributes[attribute_name] = build_attribute(
                self.global_config,
                self.task_config,
                attribute_config,
                LANESUBATTRIBUTES,
                attribute_name,
            )
        assert (
            self.bits_number // self.task_config.number_of_lanes == current_digit
        ), "Bits number mismatch, please check config"
        self.bits_per_lane = current_digit
        return sub_attributes

    def lanebool_index(self, feature):
        return self.lane_bool_start + self.task_config.bool_features_lane.index(feature)

    def linebool_index(self, feature, line_name):
        return (
            self.line_bool_start
            + 2 * self.task_config.bool_features_line.index(feature)
            + self.task_config.line_names.index(line_name)
        )

    def laneregress_index(self, feature):
        return self.lane_regress_start + self.task_config.reg_features_lane.index(feature)

    def lineshape_range(self, line_name=None):
        if line_name is None:
            return self.lineshape_attribute.bits_range
        else:
            offset = self.task_config.line_names.index(line_name) * (
                2 + self.lineshape_attribute.num_points_each_line
            )
            all_lines_range = self.lineshape_attribute.bits_range
            return range(
                all_lines_range.start + offset,
                all_lines_range.start + offset + 2 + self.lineshape_attribute.num_points_each_line,
            )

    def lineshape_point_range(self, line_name):
        offset = self.task_config.line_names.index(line_name) * (
            2 + self.lineshape_attribute.num_points_each_line
        )
        all_lines_range = self.lineshape_attribute.bits_range
        return range(
            all_lines_range.start + offset + 2,
            all_lines_range.start + offset + 2 + self.lineshape_attribute.num_points_each_line,
        )

    def set_exist_mask(self, mask):
        mask[self.lanebool_index("exist")] = 1.0
        for line_name in self.task_config.line_names:
            mask[self.linebool_index("line_exist", line_name)] = 1.0

    def extend_unseen_points(self, lane):
        if not lane["label"][self.lanebool_index("exist")] > 0:
            return lane
        for line in self.task_config.line_names:
            point_range = self.lineshape_point_range(line)
            valid_point_idx = np.nonzero(lane["mask"][point_range])[0][:2]
            if len(valid_point_idx) < 2:
                continue
            valid_points = lane["label"][point_range][valid_point_idx]
            dx = valid_points[1] - valid_points[0]
            dy = valid_point_idx[1] - valid_point_idx[0]
            slope = dx / dy
            for idx in point_range:
                if lane["mask"][idx] != 0:
                    break
                lane["mask"][idx] = self.task_config.extend_points_weight
                lane["label"][idx] = valid_points[0] - slope * (
                    valid_point_idx[0] - idx + point_range.start
                )
        return lane

    def normalize_points(self, lane):
        point_range = self.lineshape_range()
        lane["label"][point_range] -= self.global_config.image_width / 2
        return lane

    def denormalize_points(self, vectors):
        point_range = self.lineshape_range()
        vectors[point_range] += self.global_config.image_width / 2
        return vectors

    def validate_json(self, lane):
        for line_name in self.task_config.line_names:
            if line_name in lane:
                for pt in lane[line_name]["points"]:
                    if pt["y"] < self.task_config.skyhigh_threshold:
                        raise ValueError("skyhigh point exist")
                if len(lane[line_name]["points"]) < 2:
                    raise ValueError(f"{line_name} has less than two points")
        if lane["position"] != "center" and lane.get("primary", False):
            raise ValueError("primary lane must be center")

    def get_empty_label(self):
        vector = np.zeros(self.bits_per_lane)
        mask = np.zeros(self.bits_per_lane)
        return vector, mask

    def augment_lane(self, label_dict, data_augments, label_processor):
        if data_augments.get("flip", False):
            label_dict = flip_attributes(label_dict)
        for line_name in self.task_config.line_names:
            if line_name in label_dict:
                new_line = []
                for point in label_dict[line_name]["points"]:
                    x, y = label_processor.process((point["x"], point["y"]), data_augments)
                    new_line.append({"x": x, "y": y})
                # TODO: check e38 lane lines
                new_line = extend_to_boundary(
                    new_line, self.global_config.image_width, self.global_config.image_height
                )
                label_dict[line_name]["points"] = new_line
        return label_dict

    def postprocess_lane_label(self, vector, mask):
        line_exist = []
        for line_name in self.task_config.line_names:
            point_range = self.lineshape_range(line_name)
            if mask[point_range.start] == 0:
                vector[self.linebool_index("line_exist", line_name)] = 0.0
                line_exist.append(False)
            else:
                vector[self.linebool_index("line_exist", line_name)] = 1.0
                line_exist.append(True)
        if not any(line_exist):
            vector[self.lanebool_index("exist")] = 0.0
            mask[:] = 0.0
            mask[self.lanebool_index("exist")] = 1.0
        return vector, mask

    def parse_single_lane(self, label, augmentations, metadata, label_processor):
        self.validate_json(label)
        vector, mask = self.get_empty_label()
        label = fix_label_typo(label)
        label = self.augment_lane(label, augmentations, label_processor)
        for attribute in self.sub_attributes.values():
            vector, mask = attribute.label_to_vectors(
                label, augmentations, metadata, vector, mask, label_processor
            )
        vector, mask = self.postprocess_lane_label(vector, mask)
        return vector, mask

    def label_to_vectors(self, label, augmentations, metadata, vectors, masks, label_processor):
        for position, lane_id in self.task_config.pos2id.items():
            lane_range = range(lane_id * self.bits_per_lane, (lane_id + 1) * self.bits_per_lane)
            vectors[lane_range] = label[position]["label"]
            masks[lane_range] = label[position]["mask"]
        return vectors, masks

    def vectors_to_dict(self, vectors, metadata, is_gt, result):
        lanes = []
        vectors = vectors[self.bits_range]
        for position, i in self.task_config.pos2id.items():
            lane_range = range(i * self.bits_per_lane, (i + 1) * self.bits_per_lane)
            vectors_lane = vectors[lane_range]
            vectors_lane = self.denormalize_points(vectors_lane)
            lane_dict = {}
            lane_dict["left_line"] = {}
            lane_dict["right_line"] = {}
            for attribute_obj in self.sub_attributes.values():
                lane_dict = attribute_obj.vectors_to_dict(vectors_lane, metadata, is_gt, lane_dict)
            lane_dict["position"] = position
            lanes.append(lane_dict)
        result["lanes"] = lanes
        return result

    def loss(self, preds, trues, masks):
        loss = defaultdict(float)
        # split 10 lanes and calculate loss separately
        preds = preds[:, self.bits_range]
        trues = trues[:, self.bits_range]
        masks = masks[:, self.bits_range]
        for i in range(self.task_config.number_of_lanes):
            lane_range = range(i * self.bits_per_lane, (i + 1) * self.bits_per_lane)
            preds_lane = preds[:, lane_range]
            trues_lane = trues[:, lane_range]
            masks_lane = masks[:, lane_range]
            for attribute_obj in self.sub_attributes.values():
                attribute_loss = attribute_obj.loss(preds_lane, trues_lane, masks_lane)
                for loss_name, loss_value in attribute_loss.items():
                    loss[loss_name] += loss_value
        for loss_name in loss:
            loss[loss_name] /= self.task_config.number_of_lanes
        return loss

    def visualize(self, image, processed_dict, scale):
        global_info = processed_dict["global"][0]
        lanes = processed_dict["lanes"]
        visualization_lanes = {
            "normal": ["center", "left", "right"],
            "split": ["assist1", "assist2"],
            "fork": ["fork1", "fork2"],
            "wide": ["wide", "wide_left", "wide_right"],
            "no_lane": [],
        }
        visualization_linetype = {
            "normal": ["center", "center"],
            "split": ["assist2", "assist1"],
            "wide": ["wide", "wide"],
        }
        lanes_should_draw = visualization_lanes[global_info]
        lanes_should_draw = [lane for lane in lanes_should_draw if lanes[self.task_config.pos2id[lane]]["exist"][0]]

        for target_lane in lanes_should_draw:
            for i, line_name in enumerate(self.task_config.line_names):
                line = lanes[self.task_config.pos2id[target_lane]][line_name]
                if not line["line_exist"][0]:
                    continue
                point_color = self.task_config.colors[2 * self.task_config.pos2id[target_lane] + i]
                rgb_color = (
                    int(point_color[0] * 255),
                    int(point_color[1] * 255),
                    int(point_color[2] * 255),
                )
                for point in line["points"]:
                    cv2.circle(image, (int(point["x"] / scale), int(point["y"] / scale)), 2, rgb_color, -1)
        
        if global_info in visualization_linetype:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2
            lanes_should_draw = visualization_linetype[global_info]
            line = lanes[self.task_config.pos2id[lanes_should_draw[0]]]["left_line"]
            image = cv2.putText(
                image,
                line["line_type"][0],
                (0,30),
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            line = lanes[self.task_config.pos2id[lanes_should_draw[1]]]["right_line"]
            image = cv2.putText(
                image,
                line["line_type"][0],
                (0,60),
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        return image
