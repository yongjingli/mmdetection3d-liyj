# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def debug_slim_heatmap(heatmap):
    img_h, img_w = heatmap.shape
    # heatmap_mask = heatmap > 0.2
    heatmap_mask = heatmap > 0.2


    ys, xs = np.nonzero(heatmap_mask)
    # ys, xs = np.flipud(ys), np.flipud(xs)   # 优先将y大的点排在前面

    xmin = min(xs)
    xmax = max(xs)

    ymin = min(ys)
    ymax = max(ys)
    # indices = np.argwhere(heatmap_mask)
    #
    # # 按照x坐标进行排序
    # sorted_indices = indices[np.argsort(indices[:, 0])]
    #
    # xmin = sorted_indices[0][0]
    # xmax = sorted_indices[-1][0]
    #
    # sorted_indices = indices[np.argsort(indices[:, 1])]
    # ymin = sorted_indices[0][1]
    # ymax = sorted_indices[-1][1]

    pixel_step = 1

    if xmax-xmin > ymax -ymin:
        order = 0
        step_num = (xmax - xmin) / pixel_step
    else:
        order = 1
        step_num = (ymax - ymin) / pixel_step
    step_num = int(step_num)

    def get_key_point(arr, order, fixed, start, end):
        index = []
        if order == 0:
            for j in range(start, end):
                if arr[j, fixed]:
                    index.append(j)
        else:
            for j in range(start, end):
                if arr[fixed, j]:
                    index.append(j)

        start = False
        end = False
        last_ind = -1
        start_ind = -1
        end_ind = -1
        keypoint = []
        if len(index) == 1:
            if order == 0:
                keypoint.append([index[0], fixed])
            else:
                keypoint.append([fixed, index[0]])

        elif len(index) > 1:
            for i in range(len(index)):
                if start == False:
                    start = True
                    start_ind = index[i]
                if i == 0:
                    start = True
                    last_ind = index[i]
                    start_ind = index[i]
                    continue
                if index[i] - last_ind > 1 or i == len(index) - 1:
                    end_ind = last_ind
                    start = False
                    if order == 0:
                        keypoint.append([int((start_ind + end_ind) / 2), fixed])
                    else:
                        keypoint.append([fixed, int((start_ind + end_ind) / 2)])
                    start_ind = index[i]

                last_ind = index[i]
        return keypoint
    all_keypoints = []

    if order == 0:
        x_index = xmin - pixel_step
        # x_index = xmin + 30
        for i in range(step_num+1):
            x_index = x_index+pixel_step
            # debug
            # x_index = 105

            keypoint = get_key_point(heatmap_mask, order, x_index, ymin, ymax)
            if len(keypoint) == 0:
                print(x_index)

            all_keypoints.extend(keypoint)
    else:
        y_index = ymin - pixel_step
        # x_index = xmin + 30
        for i in range(step_num + 1):
            y_index = y_index + pixel_step

            keypoint = get_key_point(heatmap_mask, order, y_index, xmin, xmax)
            all_keypoints.extend(keypoint)

    image_point = np.zeros((img_h, img_w), dtype=np.uint8)
    for keypoint in all_keypoints:
        image_point[keypoint[0], keypoint[1]] = 1

    image_debug = np.zeros((img_h, img_w), dtype=np.uint8)
    image_debug[ys, xs] = 1


    plt.subplot(3, 1, 1)
    plt.imshow(heatmap_mask)

    plt.subplot(3, 1, 2)
    plt.imshow(image_point)

    plt.subplot(3, 1, 3)
    image_debug[:, 94] = 1
    image_debug[:, 122] = 1
    image_debug[70, :] = 1
    image_debug[106, :] = 1
    plt.imshow(image_debug)
    plt.show()
    exit(1)


    # # 打印排序后的结果
    # for point in sorted_indices:
    #     print(point)
    #
    # for i in range(len(all_keypoints)):
    #     keyp = all_keypoints[i]
    #
    #     # 绘制方块
    #     color = 125  # 方块的颜色，这里使用绿色
    #     thickness = 2  # 方块边框的线条粗细
    #     # cv2.circle(image, center, radius, color, thickness)
    #     cv2.circle(image, keyp, thickness, color, thickness)

    # all_keypoints = np.array(all_keypoints)
    if order == 0:
        keypoint_indices = sorted(all_keypoints, key=lambda x: x[0])
        # keypoint_indices = indices[np.argsort(all_keypoints[:, 0])]
    else:
        keypoint_indices = sorted(all_keypoints, key=lambda x: x[1])
        # keypoint_indices = indices[np.argsort(all_keypoints[:, 1])]

    first_point = keypoint_indices[0]
    keypoint_indices.pop(0)
    point_list = []
    point_list.append(first_point)


    while len(keypoint_indices) > 0:
        min_distance = 10000
        min_index = -1
        for i in range(len(keypoint_indices)):
            dis_point = np.array(first_point) - np.array(keypoint_indices[i])
            distance = np.sqrt(dis_point[0]*dis_point[0] + dis_point[1]*dis_point[1])
            if distance < min_distance:
                min_distance = distance
                min_index = i
        if min_index != -1:
            point_list.append(keypoint_indices[min_index])
            first_point = keypoint_indices[min_index]
            keypoint_indices.pop(min_index)


    # 显示绘制结果
    # cv2.imshow('Square', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 显示绘制结果
    image_line = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    color = 10
    pre_point = point_list[0]
    for i in range(len(point_list)):
        if i != 0:
            keyp = point_list[i]
            # 绘制方块
            color = color + 2  # 方块的颜色，这里使用绿色
            thickness = 1  # 方块边框的线条粗细
            # cv2.circle(image, center, radius, color, thickness)
            cv2.circle(image_line, keyp, 1, color, 1)
            # cv2.line(image_line, pre_point, keyp, color, thickness, 8)

            pre_point = keyp
    plt.imshow(image_line)
    plt.show()
    # cv2.imshow('image_line', image_line)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



def max_pooling_heatmap(heatmap):
    t_heatmap = torch.from_numpy(np.expand_dims(heatmap, axis=0))


    max_pooling_col = nn.MaxPool2d((3, 1), stride=(1, 1), padding=[1, 0])
    max_pooling_row = nn.MaxPool2d((1, 3), stride=(1, 1), padding=[0, 1])
    max_pooling_both = nn.MaxPool2d((3, 3), stride=(1, 1), padding=[1, 1])


    heatmap_max_pooling_row = max_pooling_row(t_heatmap)
    mask = t_heatmap == heatmap_max_pooling_row
    heatmap_row = torch.zeros_like(t_heatmap)
    heatmap_row[mask] = t_heatmap[mask]

    heatmap_max_pooling_col = max_pooling_col(t_heatmap)
    mask = t_heatmap == heatmap_max_pooling_col
    heatmap_col = torch.zeros_like(t_heatmap)
    heatmap_col[mask] = t_heatmap[mask]

    heatmap_row_col = torch.bitwise_or(heatmap_row > 0.2, heatmap_col > 0.2)
    heatmap_row = heatmap_row.detach().cpu().numpy()
    heatmap_col = heatmap_col.detach().cpu().numpy()
    heatmap_row_col = heatmap_row_col.detach().cpu().numpy()

    plt.subplot(4, 1, 1)
    plt.imshow(t_heatmap[0])

    plt.subplot(4, 1, 2)
    plt.imshow(heatmap_row[0] > 0.2)

    plt.subplot(4, 1, 3)
    plt.imshow(heatmap_col[0] > 0.2)

    # heatmap_col_row = np.bitwise_or(heatmap_row[0] > 0.2, heatmap_col[0] > 0.2)
    plt.subplot(4, 1, 4)
    plt.imshow(heatmap_row_col[0])
    plt.show()


def get_line_key_point(arr, order, fixed, start, end):
    index = []
    if order == 0:
        for j in range(start, end):
            if arr[j, fixed]:
                index.append(j)
    else:
        for j in range(start, end):
            if arr[fixed, j]:
                index.append(j)

    start = False
    last_ind = -1
    start_ind = -1
    keypoint = []
    for i in range(len(index)):
        if start == False:
            start = True
            start_ind = index[i]
        if i == 0:
            start = True
            last_ind = index[i]
            start_ind = index[i]
            continue
        if index[i] - last_ind > 1 or i == len(index) - 1:
            end_ind = last_ind
            start = False
            if order == 0:
                keypoint.append([int((start_ind + end_ind) / 2), fixed])
            else:
                keypoint.append([fixed, int((start_ind + end_ind) / 2)])
            start_ind = index[i]

        last_ind = index[i]
    return keypoint


def get_slim_points(heatmap_mask, start_x, end_x, start_y, end_y, step, order):
    slim_points = []
    for x_index in range(start_x, end_x+1, step):
        keypoint = get_line_key_point(heatmap_mask, order, x_index, start_y, end_y)
        slim_points.extend(keypoint)
    slim_points = np.array(slim_points)
    return slim_points


def debug_slim_heatmap2(heatmap):
    img_h, img_w = heatmap.shape
    heatmap_mask = heatmap > 0

    ys, xs = np.nonzero(heatmap_mask)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    pixel_step = 1
    if xmax-xmin > ymax -ymin:
        order = 0
        slim_points = get_slim_points(heatmap_mask, xmin, xmax, ymin, ymax, pixel_step, order)
    else:
        order = 1
        slim_points = get_slim_points(heatmap_mask, ymin, ymax, xmin, xmax, pixel_step, order)

    ys, xs = slim_points[:, 0], slim_points[:, 1]

    image_point = np.zeros((img_h, img_w), dtype=np.uint8)
    for keypoint in slim_points:
        image_point[keypoint[0], keypoint[1]] = 1

    plt.imshow(image_point)
    plt.show()



if __name__ == "__main__":
    print("Start")
    # data_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1695030883240994950_img_heatmamp_line.npy"
    # data_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1695031238613012474_img_heatmamp_line.npy"
    data_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1695031405813262853_img_heatmamp_line.npy"
    # data_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/local_files/debug_gbld/data/c0bfbf61-2adf-40d3-85b1-ed9332a5fedb_front_camera_9201_img_heatmamp_line.npy"
    # 读取.npy文件
    heatmap = np.load(data_path)
    debug_slim_heatmap(heatmap)
    # max_pooling_heatmap(heatmap)
    # debug_slim_heatmap2(heatmap)
    print("End")
