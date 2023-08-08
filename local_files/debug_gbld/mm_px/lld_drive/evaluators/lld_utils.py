import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import Polygon


def robust_div(a, b):
    if b == 0 or a == np.nan or b == np.nan:
        return np.nan
    else:
        return a / b

def calculate_f1_with_threshold(df, key, threshold):
    sub_df = df[[key + "_prob", key + "_true"]].dropna()
    if len(sub_df) == 0:
        return {
            "precision": np.nan, 
            "recall": np.nan, 
            "f1": np.nan, 
            "num_gt": 0, 
            "num_pred": 0,
            "fp_rate": np.nan,
            "num_negative": 0,
        }
    pred = sub_df[key + "_prob"].to_numpy()
    true = sub_df[key + "_true"].to_numpy()
    tp = ((pred > threshold) * (true > 0.5)).sum()
    num_pred = (pred > threshold).sum()
    num_gt = (true > 0.5).sum()
    p = robust_div(tp, num_pred)
    r = robust_div(tp, num_gt)
    f1 = robust_div(2 * p * r, p + r)
    num_negative = len(sub_df) - num_gt
    fp_rate = (num_pred - tp) / num_negative if num_negative > 0 else np.nan
    return {
        "precision": p, 
        "recall": r,
        "f1": f1, 
        "num_gt": num_gt, 
        "num_pred": num_pred, 
        "fp_rate": fp_rate,
        "num_negative": num_negative,
    }

def calculate_f1(df, key, value):
    sub_df = df[[key + "_pred", key + "_true"]].dropna()
    if len(sub_df) == 0:
        return {
            "precision": np.nan, 
            "recall": np.nan, 
            "f1": np.nan, 
            "num_gt": 0, 
            "num_pred": 0,
        }
    pred = sub_df[key + "_pred"]
    true = sub_df[key + "_true"]
    tp = ((pred == value) & (true == value)).sum()
    num_pred = (pred == value).sum()
    num_gt = (true == value).sum()
    p = robust_div(tp, num_pred)
    r = robust_div(tp, num_gt)
    f1 = robust_div(2 * p * r, p + r)
    num_negative = len(sub_df) - num_gt
    fp_rate = (num_pred - tp) / num_negative if num_negative > 0 else np.nan
    return {
        "precision": p, 
        "recall": r,
        "f1": f1, 
        "num_gt": num_gt, 
        "num_pred": num_pred, 
    }
    
def calculate_accuracy(df, key):
    sub_df = df[[key + "_pred", key + "_true"]].dropna()
    if len(sub_df) == 0:
        return {"accuracy": np.nan}
    pred = sub_df[key + "_pred"]
    true = sub_df[key + "_true"]
    return {"accuracy": (pred == true).mean()}


def calculate_rmse(df, key):
    sub_df = df[[key + "_pred", key + "_true"]].dropna()
    if len(sub_df) == 0:
        return {"rmse": np.nan}
    pred = sub_df[key + "_pred"]
    true = sub_df[key + "_true"]
    return {"rmse": np.sqrt(((pred - true) ** 2).mean())}


def build_valid_polygon(left_line, right_line):
    """Generate a valid polygon from left and right line points,
    will pop the last pair of points if the polygon is invalid

    Args:
        left_line (list): list of left line points
        right_line (list): list of right line points

    Returns:
        A valid polygon if it can be found, else return None
    """
    valid = False
    polygon = None
    while not valid and len(left_line) > 2 and len(right_line) > 2:
        polygon = Polygon(left_line + right_line[::-1])
        valid = polygon.is_valid
        left_line = left_line[:-1]
        right_line = right_line[:-1]
    if valid:
        return polygon
    else:
        return None


def cal_iou_v2(global_config, pred_lane, gt_lane, target_range=(-1, 10e8), scale=2):
    """Calculate the iou of prediction lines and gt lines(left and right)
    logic: use shapely.Polygon(left_line + right_line.reverse()) to create polygon,
    will pop the last pair of points if the polygon is invalid

    Args:
        pred_lines: a line object contains two line(pred)
        gt_lines: a line object contains two line(gt)

    Returns: value of intersection, union, iou

    """
    points = {
        "pred": {"left_line": [], "right_line": []},
        "gt": {"left_line": [], "right_line": []},
    }
    for source in ["pred", "gt"]:
        point_source = gt_lane if source == "gt" else pred_lane
        for position in ["left_line", "right_line"]:
            for point in point_source[position]["points"]:
                if target_range[0] < point["y"] <= target_range[1]:
                    points[source][position].append((point["x"], point["y"]))
            # if only one line exist then use corner as part of the polygon
            if len(gt_lane[position]["points"]) == 0:
                if position == "left_line":
                    points[source][position] = [(0, global_config.image_height * scale)]
                else:
                    points[source][position] = [
                        (global_config.image_width * scale, global_config.image_height * scale)
                    ]
    gt_poly = build_valid_polygon(points["gt"]["left_line"], points["gt"]["right_line"])
    pred_poly = build_valid_polygon(points["pred"]["left_line"], points["pred"]["right_line"])
    if gt_poly is None:
        return "invalid", "invalid", "invalid"
    if pred_poly is None:
        return "error", "error", "error"
    intersection = pred_poly.intersection(gt_poly).area
    union = pred_poly.union(gt_poly).area
    if union == 0:
        return 0, 0, 0
    return intersection, union, intersection / union


def cal_iou(global_config, pred_lane, gt_lane, target_range=(-1, 10e8), scale=2):
    """Calculate the iou of prediction lines and gt lines(left and right)
    logic: use shapely.Polygon(left_line + right_line.reverse()) to create polygon

    Args:
        pred_lines: a line object contains two line(pred)
        gt_lines: a line object contains two line(gt)

    Returns: value of intersection, union, iou

    """
    points = {
        "pred": {"left_line": [], "right_line": []},
        "gt": {"left_line": [], "right_line": []},
    }
    for source in ["pred", "gt"]:
        point_source = gt_lane if source == "gt" else pred_lane
        for position in ["left_line", "right_line"]:
            traversal_order = 1 if position == "left_line" else -1
            for point in point_source[position]["points"][::traversal_order]:
                if target_range[0] < point["y"] <= target_range[1]:
                    points[source][position].append((point["x"], point["y"]))
            # if only one line exist then use corner as part of the polygon
            if len(gt_lane[position]["points"][::traversal_order]) == 0:
                if position == "left_line":
                    points[source][position] = [(0, global_config.image_height * scale)]
                else:
                    points[source][position] = [
                        (global_config.image_width * scale, global_config.image_height * scale)
                    ]

    # TODO find out why there is always self-intersect
    try:
        gt_poly = Polygon(points["gt"]["left_line"] + points["gt"]["right_line"])
        pred_poly = Polygon(points["pred"]["left_line"] + points["pred"]["right_line"])
    except:
        return "error", "error", "error"
    if not gt_poly.is_valid:
        return "invalid", "invalid", "invalid"
    if not pred_poly.is_valid:
        return "error", "error", "error"
    intersection = pred_poly.intersection(gt_poly).area
    union = pred_poly.union(gt_poly).area
    if union == 0:
        return "invalid", "invalid", "invalid"
    return intersection, union, intersection / union


def upsample_points(line, y_start, y_end, num_points_each_line):
    """Upsample the pred line and gt line. Rules:
    1. create 1d-interpolate function from gt_line
    2. find the intersection of [y_end, y_start] and [min(gt_line_y), max(gt_line_y), use this range as the
    y points
    3. Use the y points from 2 and function from 1 to compute remaining x

    Args:
        line: Line obj
        y_start: max position of y
        y_end: min position of y
        num_points_each_line: maximum sampling points

    Returns: None

    """
    target_y_list = [
        y_start - i * (y_start - y_end) / (num_points_each_line - 1)
        for i in range(num_points_each_line)
    ]
    line_points = line.points

    line_y_list = [point[0] for point in line_points]
    line_x_list = [point[1] for point in line_points]

    y_limit_min = max(min(target_y_list), min(line_y_list))
    y_limit_max = min(max(target_y_list), max(line_y_list))

    target_y_list = [y for y in target_y_list if y_limit_max >= y >= y_limit_min]

    line_func = interp1d(line_y_list, line_x_list, fill_value="extrapolate")

    new_line_points = [(target_y, line_func(target_y).tolist()) for target_y in target_y_list]

    line.points = new_line_points
