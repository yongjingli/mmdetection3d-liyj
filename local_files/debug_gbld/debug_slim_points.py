import numpy as np
import matplotlib.pyplot as plt
import copy


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

if __name__ == "__main__":
    print("Start")
    debug_slim_points()
    print("End")