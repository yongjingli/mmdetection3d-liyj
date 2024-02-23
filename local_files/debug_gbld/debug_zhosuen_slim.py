import matplotlib.pyplot as plt
import numpy as np
import copy
from debug_utils import color_list
from debug_slim_points import get_point_neighbor, zhang_suen_thining_condiction2


def debug_zhangsuen_slim():
    data_path = "/home/liyongjing/Egolee/hdd-data/test_data/tmp/images/raw_lines.npy"
    raw_lines = np.load(data_path, allow_pickle=True)

    img_h, img_w = 608, 960
    map_h, map_w = img_h//4, img_w//4
    seg_map = np.zeros((map_h, map_w))
    emb_map = np.zeros((map_h, map_w, 3))
    debug_map = np.zeros((map_h, map_w, 3))

    for i, raw_line in enumerate(raw_lines):
        raw_line = np.array(raw_line, dtype=np.int32)
        seg_map[raw_line[:, 1], raw_line[:, 0]] = 1
        emb_map[raw_line[:, 1], raw_line[:, 0], :] = color_list[i]

        if i == 0:
            raw_line, outs = Zhang_Suen_thining_points(raw_line)
            raw_line = np.array(raw_line)

            debug_map[raw_line[:, 1], raw_line[:, 0]] = 1

            for i, out in enumerate(outs):
                debug_map_show = np.zeros((map_h, map_w, 3))
                out = np.array(out)
                debug_map_show[out[:, 1], out[:, 0]] = 1
                plt.imshow(debug_map_show)
                plt.show()


    # plt.imshow(seg_map)
    # plt.imshow(emb_map)
    plt.imshow(debug_map)
    plt.show()


def Zhang_Suen_thining_points(line):
    # line = line.tolist()
    outs = []
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

        outs.append(out)

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break
    return out, outs



if __name__ == "__main__":
    print("STart")
    debug_zhangsuen_slim()
    print("End")