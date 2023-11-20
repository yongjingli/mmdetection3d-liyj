import cv2
import numpy as np
import matplotlib.pyplot as plt


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]

def compute_point_distance(point_0, point_1):
    distance = np.sqrt((point_0[0] - point_1[0]) ** 2 + (point_0[1] - point_1[1]) ** 2)
    return distance


def split_piecewise_lines(piecewise_lines, split_dist=12):
    split_piecewise_lines = []
    for i, raw_line in enumerate(piecewise_lines):

        start_idx = 0
        end_idx = 0
        debug_split_piecewise_lines = []

        pre_point = raw_line[0]
        for j, cur_point in enumerate(raw_line[1:]):
            point_dist = compute_point_distance(pre_point, cur_point)
            if point_dist > split_dist:
                print("point_dist", point_dist,
                      pre_point[0], pre_point[1], cur_point[0], cur_point[1])
                split_piecewise_line = raw_line[start_idx:end_idx+1]
                if len(split_piecewise_line) > 1:
                    # split_piecewise_lines.append(split_piecewise_line)
                    debug_split_piecewise_lines.append(split_piecewise_line)

                start_idx = j + 1

            pre_point = cur_point
            end_idx = j + 1

            if j == len(raw_line) - 2:
                split_piecewise_line = raw_line[start_idx:end_idx+1]
                if len(split_piecewise_line) > 1:
                    # split_piecewise_lines.append(split_piecewise_line)
                    debug_split_piecewise_lines.append(split_piecewise_line)

        # if len(debug_split_piecewise_lines) > 0:
        #     debug_split_piecewise_lines = sorted(debug_split_piecewise_lines,
        #                                          key=lambda x: len(debug_split_piecewise_lines))
        #     split_piecewise_lines.append(debug_split_piecewise_lines[0])
        split_piecewise_lines = split_piecewise_lines + debug_split_piecewise_lines

        print("fff", i, len(debug_split_piecewise_lines))
    return split_piecewise_lines


def debug_pieces_line():
    # data_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/local_files/debug_gbld/data/debug_pieces_points_20231116.npz"
    data_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/local_files/debug_gbld/data/debug_pieces_points_20231116_2.npz"
    raw_data = np.load(data_path, allow_pickle=True)
    piecewise_lines = []
    for i, file_name in enumerate(raw_data):
        piecewise_line = raw_data[file_name]
        piecewise_lines.append(piecewise_line)

    grid_size = 4
    img_h, img_w = 608//4, 960//4
    img_piecewise_lines = np.ones((img_h * grid_size, img_w * grid_size, 3), dtype=np.uint8) * 255
    img_piecewise_points = np.ones((img_h * grid_size, img_w * grid_size, 3), dtype=np.uint8) * 255

    for i, raw_line in enumerate(piecewise_lines):
        if i > len(color_list) - 1:
            color = [np.random.randint(0, 255) for i in range(3)]
        else:
            color = color_list[i]

        for cur_point in raw_line:
            cv2.circle(img_piecewise_points, (int(cur_point[0]), int(cur_point[1])), 1, color)

    piecewise_lines = split_piecewise_lines(piecewise_lines, split_dist=12)

    for i, raw_line in enumerate(piecewise_lines):
        if i > len(color_list) - 1:
            color = [np.random.randint(0, 255) for i in range(3)]
        else:
            color = color_list[i]
        pre_point = raw_line[0]

        for cur_point in raw_line[1:]:
            # x, y = cur_point[:2]
            # img_piecewise_lines[y, x] = color
            x1, y1 = int(pre_point[0]), int(pre_point[1])
            x2, y2 = int(cur_point[0]), int(cur_point[1])

            cv2.line(img_piecewise_lines, (x1, y1), (x2, y2), color, 1, 8)
            pre_point = cur_point

        start_point = raw_line[0]
        end_point = raw_line[-1]
        cv2.circle(img_piecewise_lines, (int(start_point[0]), int(start_point[1])), 5, color)
        cv2.circle(img_piecewise_lines, (int(end_point[0]), int(end_point[1])), 5, color)
        # time.sleep(0.1)

    # cv2.circle(img_piecewise_lines, (211, 398), 2, (0, 0, 0), -1)
    # cv2.circle(img_piecewise_lines, (214, 386), 2, (0, 0, 0), -1)
    #
    # cv2.circle(img_piecewise_lines, (226, 354), 10, (0, 0, 0))
    # cv2.circle(img_piecewise_lines, (238, 323), 10, (0, 0, 0))

    # plt.subplot(2, 1, 1)
    # plt.imshow(confidence_map > 0.2)
    # plt.imshow(confidence_map)
    # plt.subplot(2, 1, 2)
    # plt.imshow(img_piecewise_points)


    plt.imshow(img_piecewise_lines)
    plt.title("debug_piece_line")
    # plt.show()
    plt.show(block=True)
    plt.close('all')


if __name__ == "__main__":
    print("Start")
    debug_pieces_line()
    print("End")
