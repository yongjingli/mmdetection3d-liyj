import cv2
import json
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from debug_utils import parse_ann_infov2, draw_orient, cal_points_orient


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]


def draw_gbld_lines_on_image(img, all_points, all_points_type, all_points_visible,
                             all_points_hanging, all_points_covered):
    # img = cv2.imread(img_path)
    # img_line = np.ones_like(img) * 255
    img_line = copy.deepcopy(img)
    # img_cls = np.ones_like(img) * 255
    img_cls = copy.deepcopy(img)
    img_vis = np.ones_like(img) * 255
    img_hang = np.ones_like(img) * 255
    img_covered = np.ones_like(img) * 255

    line_count = 0

    for points, points_type, points_visible, points_hanging, points_covered in \
            zip(all_points, all_points_type, all_points_visible, all_points_hanging, all_points_covered):

        pre_point = points[0]
        color = (255, 0, 0)
        for i, cur_point in enumerate(points[1:]):
            x1, y1 = int(pre_point[0]), int(pre_point[1])
            x2, y2 = int(cur_point[0]), int(cur_point[1])

            cv2.circle(img, (x1, y1), 1, color, 1)
            cv2.line(img, (x1, y1), (x2, y2), color, 3)
            pre_point = cur_point

        pre_point = points[0]
        pre_point_type = points_type[0]
        pre_point_vis = points_visible[0]
        pre_point_hang = points_hanging[0]
        pre_point_covered = points_covered[0]

        for cur_point, point_type, point_vis, point_hang, point_covered in zip(points[1:],
                                                                               points_type[1:],
                                                                               points_visible[1:],
                                                                               points_hanging[1:],
                                                                               points_covered[1:]):

            color = color_list[line_count % len(color_list)]
            x1, y1 = int(pre_point[0]), int(pre_point[1])
            x2, y2 = int(cur_point[0]), int(cur_point[1])

            cv2.circle(img_line, (x1, y1), 1, color, 1)
            cv2.line(img_line, (x1, y1), (x2, y2), color, 3)
            pre_point = cur_point

            # 点的属性全都是字符类型
            # img_cls
            # A ---- B -----C
            # A点的属性代表AB线的属性
            # B点的属性代表BC线的属性
            # cls_index = classes.index(pre_point_type[0])
            cls_index = int(point_type)
            color = color_list[cls_index]
            cv2.circle(img_cls, (x1, y1), 1, color, 1)
            cv2.line(img_cls, (x1, y1), (x2, y2), color, 3)
            pre_point_type = point_type

            # img_vis
            # 绿色为true, 蓝色为false
            # point_vis为true的情况下为可见
            color = (0, 255, 0) if pre_point_vis[0] == "true" or pre_point_vis[0] else (255, 0, 0)
            cv2.circle(img_vis, (x1, y1), 1, color, 1)
            cv2.line(img_vis, (x1, y1), (x2, y2), color, 3)
            pre_point_vis = point_vis

            # img_hang
            # point_hang为true的情况为悬空
            color = (0, 255, 0) if pre_point_hang[0] == "true" or pre_point_vis[0] else (255, 0, 0)
            cv2.circle(img_hang, (x1, y1), 1, color, 1)
            cv2.line(img_hang, (x1, y1), (x2, y2), color, 3)
            pre_point_hang = point_hang

            # img_covered
            # point_covered为true的情况为被草遮挡
            color = (0, 255, 0) if pre_point_covered[0] == "true" or pre_point_vis[0] else (255, 0, 0)
            cv2.circle(img_covered, (x1, y1), 1, color, 1)
            cv2.line(img_covered, (x1, y1), (x2, y2), color, 3)
            pre_point_covered = point_covered

        line_count = line_count + 1

    return img_line, img_cls, img_vis, img_hang, img_covered


def draw_gbld_pred(img_show, single_stage_result, colors=None):
    for line_id, curve_line in enumerate(single_stage_result):
        curve_line = np.array(curve_line)

        # poitns_cls = curve_line[:, 4]
        # 通过统计的方式得到整条线的类别
        # line_cls = np.argmax(np.bincount(poitns_cls.astype(np.int32)))
        # points[:, 4] = cls

        point_num = len(curve_line)
        pre_point = curve_line[0]

        for i, cur_point in enumerate(curve_line[1:]):
            x1, y1 = int(pre_point[0]), int(pre_point[1])
            x2, y2 = int(cur_point[0]), int(cur_point[1])

            if len(pre_point) >= 9:
                point_visible = pre_point[6]
                point_hanging = pre_point[7]
                point_covered = pre_point[8]
            else:
                point_visible = -1
                point_hanging = -1
                point_covered = -1
            # print("point_covered:", point_covered)

            if colors is None:
                point_cls = pre_point[4]
                color = color_list[int(point_cls)]
            else:

                color = colors[line_id]

            thickness = 3
            cv2.line(img_show, (x1, y1), (x2, y2), color, thickness, 8)
            line_orient = cal_points_orient(pre_point, cur_point)
            if -1 not in [point_visible, point_covered]:
                if point_visible < 0.2 and point_covered < 0.2:
                    cv2.circle(img_show, (x2, y2), thickness * 2, (0, 0, 0), thickness=2)

            if i % 50 == 0:
                if len(pre_point) > 5:
                    orient = pre_point[5]
                else:
                    orient = -1

                if orient != -1:
                    reverse = False  # 代表反向是否反了
                    orient_diff = abs(line_orient - orient)
                    if orient_diff > 180:
                        orient_diff = 360 - orient_diff

                    if orient_diff > 90:
                        reverse = True

                    # color = (0, 255, 0)
                    # if reverse:
                    #     color = (0, 0, 255)

                    # 绘制预测的方向
                    # 转个90度,指向草地
                    orient = orient + 90
                    if orient > 360:
                        orient = orient - 360
                    # img_origin = draw_orient(img_origin, pre_point, orient, arrow_len=30, color=color)
            if i == point_num // 2:
                line_orient = line_orient + 90
                if line_orient > 360:
                    line_orient = line_orient - 360
                img_show = draw_orient(img_show, pre_point, line_orient, arrow_len=50, color=color)

            pre_point = cur_point
    return img_show


def debug_show_draw_gbld_gt():
    img_path = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231025_mmdet3d_spline/test/images/1696991193.66982.jpg"
    ann_path = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231025_mmdet3d_spline/test/jsons/1696991193.66982.json"
    info = {"img_path": img_path,
            "ann_path": ann_path,
            }
    classes = [
        'road_boundary_line',
        'bushes_boundary_line',
        'fence_boundary_line',
        'stone_boundary_line',
        'wall_boundary_line',
        'water_boundary_line',
        'snow_boundary_line',
        'manhole_boundary_line',
        'others_boundary_line',
    ]
    ann_info = parse_ann_infov2(info, classes=classes)
    img = cv2.imread(img_path)
    gt_lines = ann_info["gt_lines"]

    all_points = []
    all_points_type = []
    all_points_visible = []
    all_points_hanging = []
    all_points_covered = []
    for gt_line in gt_lines:
        all_points.append(gt_line['points'])
        all_points_type.append(gt_line['points_type'])
        all_points_visible.append(gt_line['points_visible'])
        all_points_hanging.append(gt_line['points_hanging'])
        all_points_covered.append(gt_line['points_covered'])

    show_img_line_gt, show_img_cls_gt, show_img_vis_gt,\
    show_img_hang_gt, show_img_covered_gt = draw_gbld_lines_on_image(img, all_points,
                                                                     all_points_type,
                                                                     all_points_visible,
                                                                     all_points_hanging,
                                                                     all_points_covered,
                                                                     )


    plt.imshow(show_img_line_gt[:, :, ::-1])
    # plt.imshow(show_img_cls_gt[:, :, ::-1])
    # plt.imshow(show_img_vis_gt[:, :, ::-1])
    # plt.imshow(show_img_hang_gt[:, :, ::-1])
    # plt.imshow(show_img_covered_gt[:, :, ::-1])
    plt.show()


if __name__ == "__main__":
    print("Start")
    debug_show_draw_gbld_gt()
    print("End")


