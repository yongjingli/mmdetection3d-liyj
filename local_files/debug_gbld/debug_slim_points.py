import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
from skimage import morphology

def get_line_key_point(line, order, fixed):
    index = []
    if order == 0:
        for point in line:
            if int(point[0]) == int(fixed):
                index.append(int(point[1]))
    else:
        for point in line:
            if int(point[1]) == int(fixed):
                index.append(int(point[0]))

    index = np.sort(index)

    start = False
    last_ind = -1
    start_ind = -1
    keypoint = []
    if len(index) == 1:
        if order == 0:
            keypoint.append([fixed, index[0]])
        else:
            keypoint.append([index[0], fixed])

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
            # if abs(index[i] - last_ind) > 1 or i == len(index) - 1:
            if abs(index[i] - last_ind) > 1:
                end_ind = last_ind
                start = False
                if order == 0:
                    keypoint.append([fixed, int((start_ind + end_ind) / 2)])
                else:
                    keypoint.append([int((start_ind + end_ind) / 2), fixed])
                start_ind = index[i]

                if i == len(index) - 1:
                    if order == 0:
                        keypoint.append([fixed, int(index[i])])
                    else:
                        keypoint.append([int(index[i]), fixed])

            elif i == len(index) - 1:
                end_ind = index[i]
                start = False
                if order == 0:
                    keypoint.append([fixed, int((start_ind + end_ind) / 2)])
                else:
                    keypoint.append([int((start_ind + end_ind) / 2), fixed])
                start_ind = index[i]
            last_ind = index[i]

    return keypoint


def get_slim_points(line, start_x, end_x, start_y, end_y, step, order):
    slim_points = []
    for x_index in range(start_x, end_x+1, step):
        # x_index = 94
        keypoint = get_line_key_point(line, order, x_index)
        slim_points.extend(keypoint)
    return slim_points


def get_slim_lines(lines):
    slim_lines = []
    for line in lines:
        line = np.array(line)
        xs, ys = line[:, 0], line[:, 1],
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        pixel_step = 1
        if xmax-xmin > ymax -ymin:
            order = 0
            slim_points = get_slim_points(line, xmin, xmax, ymin, ymax, pixel_step, order)
        else:
            order = 1
            slim_points = get_slim_points(line, ymin, ymax, xmin, xmax, pixel_step, order)

        # 为了应对圆形的情况, 两个方向都保留
        # order = 0
        # slim_points_x = get_slim_points(line, xmin, xmax, ymin, ymax, pixel_step, order)
        # # else:
        # order = 1
        # slim_points_y = get_slim_points(line, ymin, ymax, xmin, xmax, pixel_step, order)
        # slim_points = slim_points_x + slim_points_y


        # 对y进行倒序排序
        # slim_points = np.array(slim_points)
        # xs, ys = slim_points[:, 0], slim_points[:, 1]
        # sorted_indices = np.argsort(ys)[::-1]
        # slim_points = slim_points[sorted_indices].tolist()

        # if len(slim_points) > 1:
        slim_lines.append(slim_points)
    return slim_lines


def debug_slim_points():
    data_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/local_files/debug_gbld/data/debug_slim_points_1.npy"
    raw_lines = np.load(data_path, allow_pickle=True)

    raw_line = raw_lines[4]
    img_h, img_w = 608//4,  960//4

    img_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for point in raw_line:
        x, y = point
        img_mask[y, x] = 1

    raw_line_slim = get_slim_lines([raw_line])[0]
    img_mask_slim = np.zeros((img_h, img_w), dtype=np.uint8)
    for point in raw_line_slim:
        x, y = point
        img_mask_slim[y, x] = 1

    # img_mask_debug = np.zeros((img_h, img_w), dtype=np.uint8)
    img_mask_debug = copy.deepcopy(img_mask)

    plt.subplot(3, 1, 1)
    plt.imshow(img_mask)

    plt.subplot(3, 1, 2)
    plt.imshow(img_mask_slim)

    plt.subplot(3, 1, 3)
    img_mask_debug[:, 94] = 1
    # img_mask_debug[:, 94] = img_mask_slim[:, 94]
    plt.imshow(img_mask_debug)

    plt.show()
    print("FFF")

#在c++的取最大概率的方式
# void TaskEdgeLld::PickHighConfPoints(lld::PointList2f &selected_points,
#                                                                                 lld::PointList2f & selected_points_out_){
#   lld::PointList2f selected_points_sort_x_ (selected_points);
#   lld::PointList2f selected_points_sort_y_ (selected_points);
#   std::sort(selected_points_sort_x_.begin(), selected_points_sort_x_.end(),  sort_point_x_dim );
#   std::sort(selected_points_sort_y_.begin(), selected_points_sort_y_.end(),  sort_point_y_dim );
#
#   int x_min{ (int)selected_points_sort_x_[0].pt().x() };
#   int y_min{ (int)selected_points_sort_y_[0].pt().y() };
#   int x_max{ (int)selected_points_sort_x_[selected_points_sort_x_.size()-1].pt().x() };
#   int y_max { (int)selected_points_sort_y_[selected_points_sort_y_.size()-1].pt().y() };
#   // printf("x_min:%d, x_max:%d, y_min:%d, y_max:%d \n", x_min, x_max, y_min, y_max);
#   int x_range{x_max-x_min};
#   int y_range{y_max-y_min};
#   int32_t size = selected_points.size();
#   if(x_range > y_range){
#     int32_t save_x_value {0}; //record max confidence point
#     for(int32_t i =0; i < size; i ++){
#       if( selected_points_sort_x_[i].pt().x() == selected_points_sort_x_[save_x_value].pt().x() ){
#         if( selected_points_sort_x_[i].pt().conf() >  selected_points_sort_x_[save_x_value].pt().conf() ){
#           save_x_value = i;
#         }
#       }
#       else{
#        selected_points_out_.push_back(selected_points_sort_x_[save_x_value]);
#         save_x_value = i;
#       }
#
#       if(i == size-1){
#          selected_points_out_.push_back(selected_points_sort_x_[save_x_value]);
#       }
#     }
#   }
#   else{ // x_range < y_range
#    int32_t save_y_value {0}; //record max confidence point
#     for(int32_t i =0; i < size; i ++){
#       if( selected_points_sort_y_[i].pt().y() == selected_points_sort_y_[save_y_value].pt().y() ){
#         if( selected_points_sort_y_[i].pt().conf() >  selected_points_sort_y_[save_y_value].pt().conf() ){
#           save_y_value = i;
#         }
#       }
#       else{
#        selected_points_out_.push_back(selected_points_sort_y_[save_y_value]);
#         save_y_value = i;
#       }
#
#       if(i == size-1){
#          selected_points_out_.push_back(selected_points_sort_y_[save_y_value]);
#       }
#     }
#   }
# }
#

def get_slim_lines_20231116(lines):
    slim_lines = []
    for line in lines:
        line = np.array(line)
        xs, ys = line[:, 0], line[:, 1],
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        pixel_step = 1
        x_len = xmax - xmin
        y_len = ymax - ymin
        ratio_len = max(x_len, y_len) / (min(x_len, y_len) + 1e-8)
        if ratio_len > 2:
            if x_len > y_len:
                order = 0
                slim_points = get_slim_points(line, xmin, xmax, ymin, ymax, pixel_step, order)
            else:
                order = 1
                slim_points = get_slim_points(line, ymin, ymax, xmin, xmax, pixel_step, order)
        else:
            order = 0
            slim_points_x = get_slim_points(line, xmin, xmax, ymin, ymax, pixel_step, order)
            order = 1
            slim_points_y = get_slim_points(line, ymin, ymax, xmin, xmax, pixel_step, order)
            slim_points = slim_points_x + slim_points_y

        # 对y进行倒序排序
        # slim_points = np.array(slim_points)
        # xs, ys = slim_points[:, 0], slim_points[:, 1]
        # sorted_indices = np.argsort(ys)[::-1]
        # slim_points = slim_points[sorted_indices].tolist()

        # if len(slim_points) > 1:
        slim_lines.append(slim_points)
    return slim_lines


def debug_slim_points2():
    data_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/local_files/debug_gbld/data/debug_slim_points_20231116.npz"
    raw_lines = np.load(data_path, allow_pickle=True)

    raw_line = raw_lines['arr_0']
    img_h, img_w = 608//4,  960//4

    img_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for point in raw_line:
        x, y = point
        img_mask[y, x] = 1

    raw_line_slim = get_slim_lines_20231116([raw_line])[0]
    img_mask_slim = np.zeros((img_h, img_w), dtype=np.uint8)
    for point in raw_line_slim:
        x, y = point
        img_mask_slim[y, x] = 1

    # img_mask_debug = np.zeros((img_h, img_w), dtype=np.uint8)
    img_mask_debug = copy.deepcopy(img_mask)

    # 对slim-mask进行闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素
    img_mask_slim_close = cv2.morphologyEx(img_mask_slim, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭运算1

    plt.subplot(3, 1, 1)
    plt.imshow(img_mask)

    plt.subplot(3, 1, 2)
    plt.imshow(img_mask_slim)

    plt.subplot(3, 1, 3)
    img_mask_debug[:, 94] = 1
    # img_mask_debug[:, 94] = img_mask_slim[:, 94]
    # plt.imshow(img_mask_debug)
    plt.imshow(img_mask_slim_close)

    plt.show()
    print("FFF")


def skeletonize_points(points):
    # Determine the bounding box of the points
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    # Create an empty grid
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    grid = np.zeros((height, width), dtype=np.bool8)

    # Set the pixels corresponding to the points to True
    grid[points[:, 1] - min_y, points[:, 0] - min_x] = True

    # Compute the distance transform
    distance = np.sqrt(np.square(np.arange(width).reshape(1, -1) - np.arange(width).reshape(-1, 1)) +
                       np.square(np.arange(height).reshape(1, -1) - np.arange(height).reshape(-1, 1)))

    # Threshold the distance transform to obtain the skeleton
    skeleton = distance <= 1

    return skeleton

def debug_slim_points3():
    data_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/local_files/debug_gbld/data/debug_slim_points_20231117.npy.npz"
    raw_lines = np.load(data_path, allow_pickle=True)

    raw_line = raw_lines['arr_1']
    img_h, img_w = 608//4,  960//4

    line = np.array(raw_line)
    xs, ys = line[:, 0], line[:, 1],
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # slim points
    if 0:
        pixel_step = 1
        order = 0
        slim_points_x = get_slim_points(line, xmin, xmax, ymin, ymax, pixel_step, order)
        order = 1
        slim_points_y = get_slim_points(line, ymin, ymax, xmin, xmax, pixel_step, order)
        slim_points = slim_points_x + slim_points_y
    else:
        slim_points = Zhang_Suen_thining_points(line)

    # slim point by thinging
    img_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for point in raw_line:
        x, y = point
        img_mask[y, x] = 1

    # img_mask = morphology.skeletonize(img_mask)
    # img_mask = img_mask.astype(np.uint8) * 255

    img_mask_thining = Zhang_Suen_thining_img(img_mask)

    # slim_points = skeletonize_points(np.array(slim_points))

    # plt.subplot(3, 1, 1)
    plt.imshow(img_mask)

    img_mask_slim = np.zeros((img_h, img_w), dtype=np.uint8)
    for point in slim_points:
        x, y = point
        img_mask_slim[y, x] = 1

    # plt.subplot(3, 1, 2)
    # plt.imshow(img_mask_slim)

    # plt.subplot(3, 1, 3)
    # img_mask_debug[:, 94] = 1
    # # img_mask_debug[:, 94] = img_mask_slim[:, 94]
    # plt.imshow(img_mask_thining)
    # plt.imshow(img_mask_slim_close)

    plt.show()

def get_point_neighbor(line, point, b_inv=True):
    # b_inv为true的时候,1代表neighbor不存在
    # 这里的计算对应的图像坐标系 y-1 为x2, 图像的实现方式
    # x9 x2 x3
    # x8 x1 x4
    # x7 x6 x5

    # 但是计算出来的线实际为y+1为x2, 修改y的offset来实现
    # x1, x2, x3, x4, x5, x6, x7, x8, x9
    x_offset = [0,  0,  1,  1,  1,  0,  -1, -1, -1]
    # y_offset = [0, -1, -1,  0,  1,  1,   1,  0, -1]    # y-1 为x2
    y_offset = [0,  1,  1, 0, -1, -1,  -1,  0, 1]    # y+1 为x2
    point_neighbors = []
    x_p, y_p = point[0], point[1]

    line_points = line
    for x_o, y_o in zip(x_offset, y_offset):
        x_n, y_n = x_p + x_o, y_p + y_o
        if b_inv:
            not_has_neighbot = 1 if [x_n, y_n] not in line_points else 0
            # if not_has_neighbot != 0:
            #     print("fffff")
            #     exit(1)
            point_neighbors.append(not_has_neighbot)
        else:
            has_neighbot = 1 if [x_n, y_n] in line_points else 0
            point_neighbors.append(has_neighbot)
    return point_neighbors

def zhang_suen_thining_condiction2(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    f1 = 0
    if (x3 - x2) == 1:
        f1 += 1
    if (x4 - x3) == 1:
        f1 += 1
    if (x5 - x4) == 1:
        f1 += 1
    if (x6 - x5) == 1:
        f1 += 1
    if (x7 - x6) == 1:
        f1 += 1
    if (x8 - x7) == 1:
        f1 += 1
    if (x9 - x8) == 1:
        f1 += 1
    if (x2 - x9) == 1:
        f1 += 1
    return f1

def Zhang_Suen_thining_points(line):
    # line = line.tolist()
    out = copy.deepcopy(line.tolist())
    while True:
        s1 = []
        s2 = []
        # x9 x2 x3
        # x8 x1 x4
        # x7 x6 x5
        for point in out:
            # condition 2
            x1, x2, x3, x4, x5, x6, x7, x8, x9 = get_point_neighbor(out, point, b_inv=True)
            f1 = zhang_suen_thining_condiction2(x1, x2, x3, x4, x5, x6, x7, x8, x9)
            if f1 != 1:
                continue

            # condition 3
            f2 = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9)
            if f2 < 2 or f2 > 6:
                continue

            # condition 4
            # x2 x4 x6
            if (x2 + x4 + x6) < 1:
                continue

            # x4 x6 x8
            if (x4 + x6 + x8) < 1:
                continue
            s1.append(point)

        # 将s1中的点去除
        out = [point for point in out if point not in s1]
        for point in out:
            x1, x2, x3, x4, x5, x6, x7, x8, x9 = get_point_neighbor(out, point, b_inv=True)
            f1 = zhang_suen_thining_condiction2(x1, x2, x3, x4, x5, x6, x7, x8, x9)

            if f1 != 1:
                continue

            f2 = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9)
            if f2 < 2 or f2 > 6:
                continue

            if (x2 + x4 + x6) < 1:
                continue

            if (x4 + x6 + x8) < 1:
                continue
            s2.append(point)

        # 将s2中的点去除
        out = [point for point in out if point not in s2]

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break
    return out


def Zhang_Suen_thining_img(img):
    # get shape
    H, W = img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int32)
    out[img > 0] = 1

    # inverse
    out = 1 - out

    while True:
        s1 = []
        s2 = []

        # step 1 ( rasta scan )
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                # condition 1
                if out[y, x] > 0:
                    continue
                # condition 2
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1:
                    f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1:
                    f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                # condition 3
                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                # condition 4
                # x2 x4 x6
                if (out[y - 1, x] + out[y, x + 1] + out[y + 1, x]) < 1:
                    continue

                # condition 5
                # x4 x6 x8
                if (out[y, x + 1] + out[y + 1, x] + out[y, x - 1]) < 1:
                    continue

                s1.append([y, x])

        for v in s1:
            out[v[0], v[1]] = 1

        # step 2 ( rasta scan )
        for y in range(1, H - 1):
            for x in range(1, W - 1):

                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1:
                    f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1:
                    f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                # condition 3
                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                # condition 4
                # x2 x4 x8
                if (out[y - 1, x] + out[y, x + 1] + out[y, x - 1]) < 1:
                    continue

                # condition 5
                # x2 x6 x8
                if (out[y - 1, x] + out[y + 1, x] + out[y, x - 1]) < 1:
                    continue

                s2.append([y, x])

        for v in s2:
            out[v[0], v[1]] = 1

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break

    out = 1 - out
    out = out.astype(np.uint8) * 255

    return out


if __name__ == "__main__":
    print("Start")
    # debug_slim_points()
    # debug_slim_points2()
    debug_slim_points3()
    print("End")