from copy import deepcopy
import numpy as np
from xpilot_lightning.machine_learning.tasks.base_linear.preprocesses.preprocess import (
    LinearBasePreprocess,
)
from xpilot_lightning.machine_learning.tasks.builder import PREPROCESSES
from xpilot_lightning.data.dataloader_helpers import LabelProcessor
from xpilot_vision.tasks.lld2lightning.config_parsers.config_parser import LLD2LIGHTNINGConfigParser
import json


def segment_segment_check(pt1, pt2, pt3, pt4):
    """Check intersection between two line segments pt1---pt2,pt3---pt4

    Args:
        pt1 (tuple): Point in (x,y) format
        pt2 (tuple): Point in (x,y) format
        pt3 (tuple): Point in (x,y) format
        pt4 (tuple): Point in (x,y) format

    Returns:
        bool: If True, two line segments have intersection
    """
    A = np.array([[pt2[0] - pt1[0], pt3[0] - pt4[0]], [pt2[1] - pt1[1], pt3[1] - pt4[1]]])
    b = np.array([pt3[0] - pt1[0], pt3[1] - pt1[1]])
    if np.linalg.det(A) == 0:
        return False
    x = np.linalg.solve(A, b)
    return 0 <= x[0] <= 1 and 0 <= x[1] <= 1


def segment_rectangle_check(pt1, pt2, rect):
    """Check intersection between a line segment pt1---pt2 and a rectangle
       Case 1: the line segment is inside the rectangle
       Case 2: the line segment intersects any of four edges

    Args:
        pt1 (tuple): Point in (x,y) format
        pt2 (tuple): Point in (x,y) format
        rect (tuple): Vetices of the rectangle(up left and right bottom) in (x1,y1,x2,y2) format

    Returns:
        bool: If True, the line segment intersects with the rectangle
    """
    # case 1, inside the rectangle
    if (
        rect[0] <= pt1[0] <= rect[2]
        and rect[1] <= pt1[1] <= rect[3]
        and rect[0] <= pt2[0] <= rect[2]
        and rect[1] <= pt2[1] <= rect[3]
    ):
        return True
    # case 2,check intersection between line segment and rectangle
    # vertices in 1 --- 2 order
    #             |     |
    #             4 --- 3
    v1 = (rect[0], rect[1])
    v2 = (rect[2], rect[1])
    v3 = (rect[2], rect[3])
    v4 = (rect[0], rect[3])
    return (
        segment_segment_check(pt1, pt2, v1, v2)
        or segment_segment_check(pt1, pt2, v2, v3)
        or segment_segment_check(pt1, pt2, v3, v4)
        or segment_segment_check(pt1, pt2, v4, v1)
    )


@PREPROCESSES.register_module()
class LLD2LIGHTNINGPreProcessing(LinearBasePreprocess):
    def __init__(self, global_config, task_config: LLD2LIGHTNINGConfigParser):
        super().__init__(global_config, task_config)
        self.task_config = task_config
        self.lane_attribute = self.task_config.attributes["Lane"]

    def reshape_e38_line(self, line, cam_id, scale=2):
        new_points = []
        for point in line["points"]:
            new_point = LabelProcessor.reshape_e38(
                self.global_config, (point["x"], point["y"]), cam_id
            )
            new_points.append({"x": new_point[0], "y": new_point[1]})
        line["points"] = new_points
        return line

    def reshape_e38_and_reassign(self, line, cam_id, scale=2, line_type_h=300, margin=0):
        """Transform a line from e38 to e28 format and reassign line type, entirely imagined, cross2noncross.
           This function will first transform points on the line into e28 frame, then it will determine line type in the nearby region.
           It will determine entirely imagined and cross2noncross according to the whole line after cropping.

        Args:
            line (dict): A line in LLD's label
            cam_id (str): Camera id
            scale (int, optional): Label to image scale. Defaults to 2.
            line_type_h (int, optional): Height of the nearby region to determine line type, in E28 label scale. Defaults to 250.
            margin (int, optional): Margin between the nearby region and whole e28 image, to avoid too short line segments. Defaults to 20.

        Returns:
            dict or None: if None, this line disappear after cropping, else return the updated line
        """
        cropped_h, cropped_w, _ = self.global_config.original_image_shapes[cam_id]
        cropped_h *= scale
        cropped_w *= scale
        rect_nearby = [margin, cropped_h - line_type_h, cropped_w - margin, cropped_h - margin]
        rect_whole = [0, 0, cropped_w, cropped_h]
        current_type = line["line_type"]
        current_visible = not line["entirely_imagined"]
        new_points = []
        nearby_linetypes = []
        whole_linetypes = []
        whole_visible = False
        for point in line["points"]:
            new_point = LabelProcessor.reshape_e38(
                self.global_config, (point["x"], point["y"]), cam_id
            )
            if len(new_points) > 0:
                last_point = new_points[-1]
                last_point = [last_point["x"], last_point["y"]]
                if segment_rectangle_check(last_point, new_point, rect_nearby):
                    nearby_linetypes.append(current_type)
                    whole_linetypes.append(current_type)
                    if current_type != "inferenced_line":
                        whole_visible = whole_visible or current_visible
                elif segment_rectangle_check(last_point, new_point, rect_whole):
                    whole_linetypes.append(current_type)
                    if current_type != "inferenced_line":
                        whole_visible = whole_visible or current_visible
            new_points.append({"x": new_point[0], "y": new_point[1]})
            if "point_type" in point:
                current_visible = point["point_type"] == "invisible2visible"
            if "segment_type" in point:
                current_type = point["segment_type"]

        if len(whole_linetypes) == 0:
            # this line no longer exist
            return None
        # determine line type
        if len(nearby_linetypes) == 0:
            line["line_type"] = whole_linetypes[0]
        else:
            non_crossable_types = [
                x for x in nearby_linetypes if self.task_config.type2crossability[x] != "crossable"
            ]
            if len(non_crossable_types) == 0:
                line["line_type"] = nearby_linetypes[0]
            else:
                line["line_type"] = non_crossable_types[0]
        noncross_types = [
            x for x in whole_linetypes if self.task_config.type2crossability[x] == "non_crossable"
        ]
        line["all_noncross_types"] = set(noncross_types)
        line["entirely_imagined"] = not whole_visible
        line["cross2noncross"] = len(noncross_types) > 0
        line["points"] = new_points
        return line

    def assign_cross2noncross(self, lanes):
        """Assign cross2noncross by adjacent lines and num of lanes.

        Args:
            lanes (list): List of lanes

        Returns:
            list: Processed list of lanes
        """
        if len(lanes) <= 1:
            return lanes
        for lane1, lane2 in zip(lanes[:-1], lanes[1:]):
            if lane1["opposite_direction"] != lane2["opposite_direction"]:
                continue
            if "right_line" in lane1 and "left_line" in lane2:
                line1 = lane1["right_line"]
                line2 = lane2["left_line"]
                new_cross2noncross = line1["cross2noncross"] or line2["cross2noncross"]
                if line1["all_noncross_types"] != {"solid_dash"}:
                    line2["cross2noncross"] = new_cross2noncross
                if line2["all_noncross_types"] != {"solid_dash"}:
                    line1["cross2noncross"] = new_cross2noncross
        center_lane = [x for x in lanes if x["position"] == "center"]
        if len(center_lane) != 1:
            return lanes
        center_lane = center_lane[0]
        center_left, center_right = center_lane["num_left_lanes"], center_lane["num_right_lanes"]
        for lane in lanes:
            if lane["position"] == "left" and (not lane["opposite_direction"]):
                if center_left == 1 and "left_line" in lane:
                    lane["left_line"]["cross2noncross"] = True
            elif lane["position"] == "right" and (not lane["opposite_direction"]):
                if center_right == 1 and "right_line" in lane:
                    lane["right_line"]["cross2noncross"] = True
            elif lane["position"] == "center":
                if center_left == 0 and "left_line" in lane:
                    lane["left_line"]["cross2noncross"] = True
                if center_right == 0 and "right_line" in lane:
                    lane["right_line"]["cross2noncross"] = True
        return lanes

    def anno_e38(self, label, cam_id, scale=2):
        """Only transform points from e38 to e28 frame

        Args:
            label (dict): LLD label
            cam_id (str): Camera id
            scale (int, optional): Label to image ratio. Defaults to 2.

        Returns:
            dict: LLD label
        """
        # if not contain segment type, go back to reshape
        label_str = json.dumps(label)
        if not ("point_type" in label_str or "segment_type" in label_str):
            return self.reshape_e38(label, cam_id)
        for lane in label["lanes"]:
            for order in ["left_line", "right_line"]:
                if order in lane:
                    new_line = self.reshape_e38_and_reassign(
                        lane[order],
                        cam_id,
                        scale,
                        self.task_config.reanno_config[0],
                        self.task_config.reanno_config[1],
                    )
                    if not new_line:
                        # this line disappear
                        del lane[order]
                    else:
                        lane[order] = new_line
        label["lanes"] = self.assign_cross2noncross(label["lanes"])
        return label

    def reshape_e38(self, label, cam_id, scale=2):
        """Only transform points from e38 to e28 frame

        Args:
            label (dict): LLD label
            cam_id (str): Camera id
            scale (int, optional): Label to image ratio. Defaults to 2.

        Returns:
            dict: LLD label
        """
        for lane in label["lanes"]:
            for order in ["left_line", "right_line"]:
                if order in lane:
                    new_line = self.reshape_e38_line(lane[order], cam_id, scale)
                    lane[order] = new_line
        return label

    def update_anno(self, label_json, cam_id):
        update_json = deepcopy(label_json)
        if cam_id == "cam2" and self.task_config.reanno_e38:
            update_json = self.anno_e38(update_json, cam_id)
        else:
            update_json = self.reshape_e38(update_json, cam_id)
        return update_json

    def validate_label(self, lanes, eps=1):
        left_point_range = self.lane_attribute.lineshape_point_range("left_line")
        right_point_range = self.lane_attribute.lineshape_point_range("right_line")
        left_lane = lanes["left"]
        right_lane = lanes["right"]
        for position in lanes:
            lane = lanes[position]
            if np.any(
                (lane["label"][left_point_range] - eps > lane["label"][right_point_range])
                * lane["mask"][left_point_range]
                * lane["mask"][right_point_range]
            ):
                raise ValueError(f"{position} lane: right line should be to the left of left line")

            if np.any(
                (left_lane["label"][left_point_range] - eps > lane["label"][left_point_range])
                * left_lane["mask"][left_point_range]
                * lane["mask"][left_point_range]
            ):
                raise ValueError(
                    f"{position} lane: left lane should be to the left to all other lanes"
                )
            if np.any(
                (left_lane["label"][right_point_range] - eps > lane["label"][right_point_range])
                * left_lane["mask"][right_point_range]
                * lane["mask"][right_point_range]
            ):
                raise ValueError(
                    f"{position} lane: left lane should be to the left to all other lanes"
                )

            if np.any(
                (right_lane["label"][left_point_range] + eps < lane["label"][left_point_range])
                * right_lane["mask"][left_point_range]
                * lane["mask"][left_point_range]
            ):
                raise ValueError(
                    f"{position} lane: right line should be to the right of all other lanes"
                )
            if np.any(
                (right_lane["label"][right_point_range] + eps < lane["label"][right_point_range])
                * right_lane["mask"][right_point_range]
                * lane["mask"][right_point_range]
            ):
                raise ValueError(
                    f"{position} lane: right line should be to the right of all other lanes"
                )

    def get_empty_label(self):
        """Generate an empty label for a lane

        Returns:
            dict: empty lane
        """
        lane_vector, lane_mask = self.lane_attribute.get_empty_label()
        return {"label": lane_vector, "mask": lane_mask, "exist": False}

    def shared_exist(self, left_lane, right_lane):
        """check two lane shared line or not
        Args:
            left_lane: dict of left_lane
            right_lane: dict of right_lane
        """
        left_line_range = self.lane_attribute.lineshape_point_range("left_line")
        right_line_range = self.lane_attribute.lineshape_point_range("right_line")
        if not (left_lane["exist"] and right_lane["exist"]):
            return False
        for l_idx, r_idx in zip(left_line_range, right_line_range):
            if left_lane["mask"][r_idx] > 0 and right_lane["mask"][l_idx] > 0:
                if (
                    abs(left_lane["label"][r_idx] - right_lane["label"][l_idx])
                    < self.task_config.shared_dist_max
                ):
                    return True
                else:
                    return False
        return False

    def copy_geometry(self, lane):
        """Copy the geometry of a lane, and its lane exist, line exist

        Args:
            lane (dict): source lane

        Returns:
            dict: copied lane
        """
        vector, mask = self.lane_attribute.get_empty_label()
        shape_range = self.lane_attribute.lineshape_range()
        vector[shape_range] = lane["label"][shape_range]
        mask[shape_range] = lane["mask"][shape_range] * 0.5
        exist_idx = self.lane_attribute.lanebool_index("exist")
        vector[exist_idx] = lane["label"][exist_idx]
        for line_name in self.task_config.line_names:
            line_exist_idx = self.lane_attribute.linebool_index("line_exist", line_name)
            vector[line_exist_idx] = lane["label"][line_exist_idx]
        return {"label": vector, "mask": mask, "exist": False}

    def split_reassignment(self, lanes):
        """Reassign global case based on geometry, a image can be one of these
        three cases: normal, soft_split, hard_split.
        Activatied lanes
        Normal: Center, Left, Right
        Soft_split: Center, Left, Right, Assist1(geo), Assist2(geo)
        Hard_split: Assist1, Assist2

        Args:
            lanes (dict): encoded lanes

        Returns:
            dict: encoded lanes
        """
        center_x = self.global_config.image_width / 2
        tight_margain = range(
            int(center_x - self.task_config.split_hard_margin),
            int(center_x + self.task_config.split_hard_margin),
        )
        soft_margain = range(
            int(center_x - self.task_config.split_soft_margin),
            int(center_x + self.task_config.split_soft_margin),
        )

        if lanes["fork1"]["exist"] and lanes["fork2"]["exist"]:
            if lanes["fork1"]["label"][self.lane_attribute.lanebool_index("primary")]:
                center_name = "fork1"
            else:
                center_name = "fork2"
        else:
            center_name = "center"
        # case 1: labeller label image as normal
        if lanes[center_name]["exist"]:
            center_lane = lanes[center_name]
            first_left_point = center_lane["label"][
                self.lane_attribute.lineshape_point_range("left_line").start
            ]
            first_right_point = center_lane["label"][
                self.lane_attribute.lineshape_point_range("right_line").start
            ]
            if abs(first_left_point - center_x) < abs(first_right_point - center_x):
                candidate_lanes = [lanes["left"], lanes[center_name]]
                start_point = first_left_point
            else:
                candidate_lanes = [lanes[center_name], lanes["right"]]
                start_point = first_right_point
            if not self.shared_exist(*candidate_lanes):
                return lanes
            if int(start_point) in tight_margain:
                lanes["assist1"] = candidate_lanes[0]
                lanes["assist2"] = candidate_lanes[1]
                lanes[center_name] = self.get_empty_label()
                lanes["left"] = self.get_empty_label()
                lanes["right"] = self.get_empty_label()
            elif int(start_point) in soft_margain:
                lanes["assist1"] = self.copy_geometry(candidate_lanes[0])
                lanes["assist2"] = self.copy_geometry(candidate_lanes[1])
            return lanes
        # case 2: labeller label image as split
        elif (
            lanes["assist1"]["exist"]
            and lanes["assist2"]["exist"]
            and self.shared_exist(lanes["assist1"], lanes["assist2"])
        ):
            # print("label_as_split")
            assist1_right = lanes["assist1"]["label"][
                self.lane_attribute.lineshape_point_range("right_line").start
            ]
            assist2_left = lanes["assist2"]["label"][
                self.lane_attribute.lineshape_point_range("left_line").start
            ]
            if int(assist1_right) in tight_margain or int(assist2_left) in tight_margain:
                lanes["left"] = self.get_empty_label()
                lanes["right"] = self.get_empty_label()
            elif int(assist1_right) in soft_margain or int(assist2_left) in soft_margain:
                if assist1_right < center_x:
                    lanes["center"] = lanes["assist2"]
                    lanes["left"] = lanes["assist1"]
                    lanes["assist1"] = self.copy_geometry(lanes["left"])
                    lanes["assist2"] = self.copy_geometry(lanes["center"])
                else:
                    lanes["center"] = lanes["assist1"]
                    lanes["right"] = lanes["assist2"]
                    lanes["assist1"] = self.copy_geometry(lanes["center"])
                    lanes["assist2"] = self.copy_geometry(lanes["right"])
            else:
                if assist1_right < center_x:
                    lanes["center"] = lanes["assist2"]
                    lanes["left"] = lanes["assist1"]
                    lanes["assist1"] = self.get_empty_label()
                    lanes["assist2"] = self.get_empty_label()
                else:
                    lanes["center"] = lanes["assist1"]
                    lanes["right"] = lanes["assist2"]
                    lanes["assist1"] = self.get_empty_label()
                    lanes["assist2"] = self.get_empty_label()
        else:
            raise ValueError("assit with no shared line is not accpeted")
        return lanes

    def extend_to_bottom(self, xs, y_idxs):
        dx = xs[1] - xs[0]
        dy = y_idxs[1] - y_idxs[0]
        return xs[0] - dx * y_idxs[0] / dy

    def wide_reassignment(self, lanes):
        """Reassign global case based on geometry, a image can be one of these
        three cases: normal, soft_wide, hard_wide.
        Activatied lanes
        Normal: Center, Left, Right
        Soft_wide: Center, Left, Right, Wide(geo), Wide_left(geo), Wide_right(geo)
        Hard_wide: Wide, Wide_left, Wide_right

        Args:
            lanes (dict): encoded lanes

        Returns:
            dict: encoded lanes
        """
        if not (lanes["center"]["exist"] or lanes["fork1"]["exist"]):
            return lanes
        if lanes["center"]["exist"]:
            center_lane = "center"
        elif lanes["fork1"]["label"][self.lane_attribute.lanebool_index("primary")]:
            center_lane = "fork1"
        else:
            center_lane = "fork2"
        center_lane_label = lanes[center_lane]["label"]
        center_lane_mask = lanes[center_lane]["mask"]
        left_line_point_range = self.lane_attribute.lineshape_point_range("left_line")
        right_line_point_range = self.lane_attribute.lineshape_point_range("right_line")
        left_line_points = center_lane_label[left_line_point_range]
        right_line_points = center_lane_label[right_line_point_range]
        left_exist_idx = np.nonzero(center_lane_mask[left_line_point_range])[0]
        right_exist_idx = np.nonzero(center_lane_mask[right_line_point_range])[0]
        is_candidate = False
        if len(left_exist_idx) >= 2 and len(right_exist_idx) >= 2:
            left_bottom_x = self.extend_to_bottom(
                left_line_points[left_exist_idx[:2]], left_exist_idx[:2]
            )
            right_bottom_x = self.extend_to_bottom(
                right_line_points[right_exist_idx[:2]], right_exist_idx[:2]
            )
            lane_width = right_bottom_x - left_bottom_x
            is_candidate = right_bottom_x >= self.global_config.image_width or left_bottom_x < 0
        elif len(right_exist_idx) >= 2:
            right_bottom_x = self.extend_to_bottom(
                right_line_points[right_exist_idx[:2]], right_exist_idx[:2]
            )
            lane_width = abs(right_bottom_x)
            is_candidate = True
        elif len(left_exist_idx) >= 2:
            left_bottom_x = self.extend_to_bottom(
                left_line_points[left_exist_idx[:2]], left_exist_idx[:2]
            )
            lane_width = abs(self.global_config.image_width - left_bottom_x)
            is_candidate = True
        else:
            raise ValueError("Lane line has fewer than 2 points")

        if lane_width > self.task_config.wide_hard_margin and is_candidate:
            if "fork" not in center_lane:
                lanes["wide"] = lanes[center_lane]
                lanes[center_lane] = self.get_empty_label()
            lanes["wide_left"] = lanes["left"]
            lanes["wide_right"] = lanes["right"]
            lanes["left"] = self.get_empty_label()
            lanes["right"] = self.get_empty_label()
        elif lane_width > self.task_config.wide_soft_margin or (
            is_candidate and self.task_config.softwide_out_of_image
        ):
            lanes["wide"] = self.copy_geometry(lanes[center_lane])
            lanes["wide_left"] = self.copy_geometry(lanes["left"])
            lanes["wide_right"] = self.copy_geometry(lanes["right"])
        return lanes

    def reorder_left_and_right(self, lane_1, lane_2, reverse_order=False):
        """Reorder left and right based on point coordinates

        Args:
            lane_1 (dict): candidate lane
            lane_2 (dict): candidate lane
            reverse_order (bool, optional): If True, will check from last point.
                Defaults to False.

        Returns:
            reordered lanes
        """
        point_order = self.lane_attribute.lineshape_point_range("left_line")
        if reverse_order:
            point_order = reversed(point_order)
        for idx in point_order:
            if lane_1["mask"][idx] and lane_2["mask"][idx]:
                if lane_1["label"][idx] <= lane_2["label"][idx]:
                    return lane_1, lane_2
                else:
                    return lane_2, lane_1
        if reverse_order:
            assert False, "this image should be ignore"
        return lane_1, lane_2

    def copy_line_exist(self, dst_lane, dst_linename, src_lane, src_linename):
        dst_idx = self.lane_attribute.linebool_index("line_exist", dst_linename)
        src_idx = self.lane_attribute.linebool_index("line_exist", src_linename)
        dst_lane["label"][dst_idx] = src_lane["label"][src_idx]

    def line_exist_reseting(self, lanes):
        """Set line exist for ego lanes to avoid flickering, need to revisit this!

        Args:
            lanes (dict): Encoded lanes

        Returns:
            lanes (dict): Encoded lanes
        """
        if lanes["assist1"]["exist"]:
            self.copy_line_exist(lanes["center"], "left_line", lanes["assist1"], "left_line")
            self.copy_line_exist(lanes["center"], "right_line", lanes["assist2"], "right_line")
            self.copy_line_exist(lanes["fork1"], "left_line", lanes["assist1"], "left_line")
            self.copy_line_exist(lanes["fork1"], "right_line", lanes["assist2"], "right_line")
            self.copy_line_exist(lanes["fork2"], "left_line", lanes["assist1"], "left_line")
            self.copy_line_exist(lanes["fork2"], "right_line", lanes["assist2"], "right_line")
            self.copy_line_exist(lanes["wide"], "left_line", lanes["assist1"], "left_line")
            self.copy_line_exist(lanes["wide"], "right_line", lanes["assist2"], "right_line")
            return lanes
        elif lanes["center"]["exist"]:
            self.copy_line_exist(lanes["fork1"], "left_line", lanes["center"], "left_line")
            self.copy_line_exist(lanes["fork1"], "right_line", lanes["center"], "right_line")
            self.copy_line_exist(lanes["fork2"], "left_line", lanes["center"], "left_line")
            self.copy_line_exist(lanes["fork2"], "right_line", lanes["center"], "right_line")
            self.copy_line_exist(lanes["wide"], "left_line", lanes["center"], "left_line")
            self.copy_line_exist(lanes["wide"], "right_line", lanes["center"], "right_line")
        elif lanes["fork1"]["exist"]:
            self.copy_line_exist(lanes["center"], "left_line", lanes["fork1"], "left_line")
            self.copy_line_exist(lanes["center"], "right_line", lanes["fork1"], "right_line")
            self.copy_line_exist(lanes["wide"], "left_line", lanes["fork1"], "left_line")
            self.copy_line_exist(lanes["wide"], "right_line", lanes["fork1"], "right_line")
        elif lanes["wide"]["exist"]:
            self.copy_line_exist(lanes["center"], "left_line", lanes["wide"], "left_line")
            self.copy_line_exist(lanes["center"], "right_line", lanes["wide"], "right_line")
            self.copy_line_exist(lanes["fork1"], "left_line", lanes["wide"], "left_line")
            self.copy_line_exist(lanes["fork1"], "right_line", lanes["wide"], "right_line")
            self.copy_line_exist(lanes["fork2"], "left_line", lanes["wide"], "left_line")
            self.copy_line_exist(lanes["fork2"], "right_line", lanes["wide"], "right_line")
        else:
            return lanes
        self.copy_line_exist(lanes["assist1"], "left_line", lanes["center"], "left_line")
        self.copy_line_exist(lanes["assist1"], "right_line", lanes["center"], "left_line")
        self.copy_line_exist(lanes["assist2"], "left_line", lanes["center"], "right_line")
        self.copy_line_exist(lanes["assist2"], "right_line", lanes["center"], "right_line")
        return lanes

    def set_primary_bits(self, lanes):
        """Set primary bit for center lane, it's only valid in fork case.

        Args:
            lanes (dict): Encoded lanes

        Returns:
            lanes (dict): Encoded lanes
        """
        primary_idx = self.lane_attribute.lanebool_index("primary")
        if lanes["assist1"]["exist"]:
            lanes["assist1"]["label"][primary_idx] = 0
            lanes["assist2"]["label"][primary_idx] = 0
        elif lanes["fork1"]["exist"]:
            assert (
                lanes["fork1"]["label"][primary_idx] + lanes["fork2"]["label"][primary_idx] == 1
            ), "Fork should have only one primary lane"
        elif lanes["center"]["exist"]:
            lanes["center"]["label"][primary_idx] = 1
        elif lanes["wide"]["exist"]:
            lanes["wide"]["label"][primary_idx] = 1
        return lanes

    def clean_num_side_lanes(self, lanes):
        """Set number of side lanes for ego lane and ignore prediction from other lanes

        Args:
            lanes (dict): Encoded lanes

        Returns:
            lanes (dict): Encoded lanes
        """
        left_idx = self.lane_attribute.laneregress_index("num_left_lanes")
        right_idx = self.lane_attribute.laneregress_index("num_right_lanes")
        if lanes["assist1"]["exist"]:
            if lanes["assist1"]["mask"][left_idx] and lanes["assist2"]["mask"][left_idx]:
                pass
            elif lanes["assist1"]["mask"][left_idx]:
                lanes["assist2"]["mask"][left_idx] = lanes["assist1"]["mask"][left_idx]
                lanes["assist2"]["mask"][right_idx] = lanes["assist1"]["mask"][right_idx]
                if lanes["assist1"]["label"][right_idx] != -1:
                    lanes["assist1"]["label"][right_idx] -= 1
                lanes["assist2"]["label"][left_idx] = lanes["assist1"]["label"][left_idx]
                lanes["assist2"]["label"][right_idx] = lanes["assist1"]["label"][right_idx]
            elif lanes["assist2"]["mask"][self.lane_attribute.laneregress_index("num_left_lanes")]:
                lanes["assist1"]["mask"][left_idx] = lanes["assist2"]["mask"][left_idx]
                lanes["assist1"]["mask"][right_idx] = lanes["assist2"]["mask"][right_idx]
                if lanes["assist2"]["label"][left_idx] != -1:
                    lanes["assist2"]["label"][left_idx] -= 1
                lanes["assist1"]["label"][left_idx] = lanes["assist2"]["label"][left_idx]
                lanes["assist1"]["label"][right_idx] = lanes["assist2"]["label"][right_idx]
            skip_position = ["assist1", "assist2"]
        elif lanes["center"]["exist"]:
            skip_position = ["center"]
        elif lanes["wide"]["exist"]:
            skip_position = ["wide"]
        else:
            skip_position = []
        for position in self.task_config.pos2id:
            if position in skip_position:
                continue
            lanes[position]["label"][left_idx] = self.task_config.ignore_value
            lanes[position]["label"][right_idx] = self.task_config.ignore_value
            lanes[position]["mask"][left_idx] = 0
            lanes[position]["mask"][right_idx] = 0
        return lanes

    def encode_lanes(
        self, label: dict, augmentations: dict, metadata: dict, label_processor: LabelProcessor
    ):
        """Encode lanes in label json format into vectors, results will be saved as
        dict. Human labels are first parsed, then some geometry-based preprocessing
        will be applied to generate the label.

        Args:
            label (dict): label dict
            augmentations (dict): xpilot_lightning augmentation
            metadata (dict): meta data of the image
            label_processor (LabelProcessor): transform label coordinates

        Returns:
            lanes (dict): A dict of encoded lanes, key is lane position, value is a
            dict containing label, mask, exist bit.
        """
        lanes = {}

        # first determine labeller's global case
        # 2 center + split=true -> split
        # 2 center + split=false -> fork
        # 1 center -> normal
        split = [
            lane["split"]
            for lane in label["lanes"]
            if lane["position"] == "center" and not lane.get("emergency_lane", False)
        ]
        primary = [
            lane["primary"]
            for lane in label["lanes"]
            if lane["position"] == "center" and not lane.get("emergency_lane", False)
        ]
        if len(split) == 1:
            if split[0]:
                raise ValueError("Split must have two assitant lanes")
            if not primary[0]:
                raise ValueError("Center lane should be primary")
            labeller_global_case = "normal"
        elif len(split) == 2:
            if all(split):
                labeller_global_case = "split"
                if any(primary):
                    raise ValueError("Assit lanes should be non primary")
            elif not any(split):
                labeller_global_case = "fork"
                if not any(primary):
                    raise ValueError("Fork should have one primary")
            else:
                raise ValueError("Inconsistent split label in center lanes")
        else:
            raise ValueError("Invalid number of center lanes")

        # parse each lane's label
        for lane_label in label["lanes"]:
            if lane_label.get("emergency_lane", False):
                continue
            lane_vector, lane_mask = self.lane_attribute.parse_single_lane(
                lane_label, augmentations, metadata, label_processor
            )
            position = lane_label["position"]
            if position == "center":
                if labeller_global_case == "normal":
                    position = "center"
                elif labeller_global_case == "split":
                    position = "assist2" if "assist1" in lanes else "assist1"
                else:
                    position = "fork2" if "fork1" in lanes else "fork1"
            assert position not in lanes, f"Duplicated lane labels for {position}"
            lanes[position] = {
                "label": lane_vector,
                "mask": lane_mask,
                "exist": lane_vector[self.lane_attribute.lanebool_index("exist")] > 0,
            }

        # add place holder for other lanes
        for position in self.task_config.pos2id:
            if position not in lanes:
                lanes[position] = self.get_empty_label()

        # ensure left/right consistency in split and fork
        if labeller_global_case == "split":
            lanes["assist1"], lanes["assist2"] = self.reorder_left_and_right(
                lanes["assist1"], lanes["assist2"], reverse_order=False
            )
        if labeller_global_case == "fork":
            lanes["fork1"], lanes["fork2"] = self.reorder_left_and_right(
                lanes["fork1"], lanes["fork2"], reverse_order=True
            )

        # reassign split and wide by geometry
        if self.task_config.split_reassign:
            lanes = self.split_reassignment(lanes)
        if self.task_config.wide_reassign:
            lanes = self.wide_reassignment(lanes)

        # hard split
        if lanes["assist1"]["exist"]:
            ignore_exist_list = ["left", "right", "wide_left", "wide_right"]
        # hard wide
        elif lanes["wide"]["exist"]:
            ignore_exist_list = ["left", "right"]
        # soft split & normal
        elif (
            lanes["center"]["exist"]
            and not lanes["wide"]["label"][self.lane_attribute.lanebool_index("exist")]
        ):
            ignore_exist_list = ["wide_left", "wide_right"]
        else:
            ignore_exist_list = []

        # additional processing
        for position in self.task_config.pos2id:
            if position not in ignore_exist_list:
                self.lane_attribute.set_exist_mask(lanes[position]["mask"])
        lanes = self.set_primary_bits(lanes)
        lanes = self.clean_num_side_lanes(lanes)
        lanes = self.line_exist_reseting(lanes)
        self.validate_label(lanes)

        # extend to boundary and normalize by half image width
        for position in lanes:
            lanes[position] = self.lane_attribute.extend_unseen_points(lanes[position])
            lanes[position] = self.lane_attribute.normalize_points(lanes[position])

        return lanes

    def process(
        self, label: dict, augmentations: dict, metadata: dict, label_processor: LabelProcessor
    ) -> dict:
        vectors = np.zeros(self.task_config.bits_number, dtype=np.float32)
        masks = np.zeros(self.task_config.bits_number, dtype=np.float32)
        lanes = self.encode_lanes(label, augmentations, metadata, label_processor)
        for attribute_name, attribute_obj in self.task_config.attributes.items():
            vectors, masks = attribute_obj.label_to_vectors(
                lanes, augmentations, metadata, vectors, masks, label_processor
            )
        return {"label": vectors, "mask": masks}
