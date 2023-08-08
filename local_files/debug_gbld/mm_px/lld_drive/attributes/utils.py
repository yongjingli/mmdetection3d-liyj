import numpy as np


def fix_label_typo(label_dict):
    # Some AR220 has a typo that mistook "opposite_direction" as "oppsite_direction".
    if "oppsite_direction" in label_dict:
        label_dict["opposite_direction"] = label_dict["oppsite_direction"]
    # conservative , can change in the future
    if "merge2left" in label_dict:
        if label_dict["merge2left"] in ["yes", "no"]:
            label_dict["merge2left"] = label_dict["merge2left"] == "yes"
        else:
            del label_dict["merge2left"]
    if "merge2right" in label_dict:
        if label_dict["merge2right"] in ["yes", "no"]:
            label_dict["merge2right"] = label_dict["merge2right"] == "yes"
        else:
            del label_dict["merge2right"]
    return label_dict


def flip_attributes(label_dict):
    if label_dict["position"] == "left":
        label_dict["position"] = "right"
    elif label_dict["position"] == "right":
        label_dict["position"] = "left"
    if "merge2left" in label_dict and "merge2right" in label_dict:
        label_dict["merge2left"], label_dict["merge2right"] = (
            label_dict["merge2right"],
            label_dict["merge2left"],
        )
    # TODO: add processing for blocking bit in the future
    # It's safe since at least one line exists
    left_line = label_dict.get("left_line", None)
    right_line = label_dict.get("right_line", None)
    if left_line is None:
        del label_dict["right_line"]
    else:
        label_dict["right_line"] = left_line
    if right_line is None:
        del label_dict["left_line"]
    else:
        label_dict["left_line"] = right_line
    return label_dict


class Line(object):
    def __init__(self, pt1, pt2):
        if (pt1["x"] - pt2["x"]) == 0:
            self.slope = 1e9
        elif (pt1["y"] - pt2["y"]) == 0:
            self.slope = 1e-9
        else:
            self.slope = (pt1["y"] - pt2["y"]) / (pt1["x"] - pt2["x"])
        self.bias = pt1["y"] - self.slope * pt1["x"]

    def get_y(self, x):
        return self.slope * x + self.bias

    def get_x(self, y):
        return (y - self.bias) / self.slope


def dist(pt1, pt2):
    return ((pt1["x"] - pt2["x"]) ** 2 + (pt1["y"] - pt2["y"]) ** 2) ** 0.5


def extend_to_boundary(line, image_width, image_height):
    """Some of the lane lines is not labeled to the image boundary.
    This function force all the lane lines to end at image boundary.
    """
    line.sort(key=lambda pt: pt["y"], reverse=True)
    pt1 = line[0]
    for pt2 in line[1:]:
        if pt2["x"] != pt1["x"] and pt2["y"] != pt1["y"]:
            break
    pt_line = Line(pt1, pt2)
    intersect_pts = [
        {"x": 0, "y": pt_line.get_y(0)},
        {"x": image_width, "y": pt_line.get_y(image_width)},
        {"x": pt_line.get_x(0), "y": 0},
        {"x": pt_line.get_x(image_height), "y": image_height},
    ]
    idx = np.argmin(
        [
            dist(pt1, intersect_pts[0]),
            dist(pt1, intersect_pts[1]),
            dist(pt1, intersect_pts[2]),
            dist(pt1, intersect_pts[3]),
        ]
    )
    line.insert(0, intersect_pts[idx])
    return line


def get_left_right_points(xys, start_point):
    # sometime one y will have multiply x in all points set
    insert_idx = np.searchsorted(np.array(xys)[:, 1], start_point, side="right")
    left_idx = max(0, insert_idx - 1)
    right_idx = min(insert_idx, len(xys) - 1)
    for i in range(left_idx, -1, -1):
        if xys[i][1] != xys[left_idx][1]:
            break
        elif abs(xys[right_idx][0] - xys[i][0]) < abs(xys[right_idx][0] - xys[left_idx][0]):
            left_idx = i

    for i in range(right_idx, len(xys)):
        if xys[i][1] != xys[right_idx][1]:
            break
        elif abs(xys[left_idx][0] - xys[i][0]) < abs(xys[right_idx][0] - xys[left_idx][0]):
            right_idx = i

    return xys[left_idx], xys[right_idx]


def interpolate_x(xys, target_y):
    left_point, right_point = get_left_right_points(xys, target_y)
    (x1, y1) = left_point
    (x2, y2) = right_point

    if y1 == target_y:
        return x1
    k = (x2 - x1) / (y2 - y1)
    b = x1 - y1 * k
    return k * target_y + b


def finetune_row_end(start_y, xys, image_width, image_height):
    """
    return fintuned row_end value
    """
    (x1, y1), (x2, y2) = get_left_right_points(xys, start_y)
    if y2 == y1:
        return y1
    if x2 == x1:
        return image_height
    if x1 == start_y:
        return y1
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    left_intersection = b
    right_intersection = image_width * k + b
    return min(image_height, max(left_intersection, right_intersection))
