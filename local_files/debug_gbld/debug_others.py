import torch
import os
import copy
import mmdet.models.necks.fpn as FPN
import mmengine
from mmengine.fileio import join_path, list_from_file, load
import numpy as np
import cv2
import json
import shutil
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from debug_utils import color_list



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
palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 215, 255), (0, 255, 255),
            (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238)]

TYPE_DICT = {
    "路面边界线": "road_boundary_line",
    "灌木丛边界线": "bushes_boundary_line",
    "围栏边界线": "fence_boundary_line",
    "石头边界线": "stone_boundary_line",
    "墙体边界线": "wall_boundary_line",
    "水面边界线": "water_boundary_line",
    "雪地边界线": "snow_boundary_line",
    "井盖边界线": "manhole_boundary_line",
    # "悬空物体边界线": "hanging_object_boundary_line",
    "其他线": "others_boundary_line"
}

TYPE_DICT_INV = {value: key for key, value in TYPE_DICT.items()}


def debug_write_data_infos():
    metainfo = dict()
    metainfo['dataset'] = 'nuscenes'
    metainfo['info_version'] = '1.1'
    converted_list = []
    for i in range(5):
        temp_data_info = str(i)
        converted_list.append(temp_data_info)
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)
    out_path = "./tmp/22.pkl"
    mmengine.dump(converted_data_info, out_path, 'pkl')

    annotations = load(out_path)

    metainfo = annotations['metainfo']
    raw_data_list = annotations['data_list']

    print("metainfo:", metainfo)
    print("raw_data_list:", raw_data_list)


def test_transform_resize():
    print("fff")


def test_dtype():
    gt_confidence = np.zeros((1, 100, 100), dtype=np.float32)
    t_gt_confidence = torch.from_numpy(gt_confidence)
    print("ffff")

def debug_sum():
    a = torch.tensor(0.2)
    b = torch.tensor(0.1)
    print(sum([a, b]))
    from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop


    from mmengine.runner.loops import ValLoop

def debug_padd():
    import mmcv

    img_path = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230728_mmdet3d/test/images/1689848740939986547.jpg"
    img = cv2.imread(img_path)
    results = {}
    results['img'] = img
    size = (img.shape[0] + 20, img.shape[1] + 20)
    padded_img = mmcv.impad(
        results['img'],
        shape=size,
        pad_val=(0, 0, 0),
        padding_mode='constant')

    plt.imshow(padded_img[:, :, ::-1])
    plt.show()


def debug_parse_label_json():
    root = "/media/dell/Egolee1/liyj/data/label_data/gbld_from_label_system/gbld_20231012_v0.2/all"
    dst_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231013_mmdet3d/debug"
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    img_root = os.path.join(root, "images")
    label_root = os.path.join(root, "json")

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", '.png']]
    for img_name in tqdm(img_names, desc="img_names"):
        # img_name = "dff47067-ede3-4ac0-b903-ef04df89a291_front_camera_821.jpg"  # 17
        # img_name = "dff47067-ede3-4ac0-b903-ef04df89a291_front_camera_461.jpg"  # 17
        # img_name = "1696991212.768321.jpg"
        # img_name = "1689848703511301286.jpg"
        # img_name = "1689848743135381664.jpg"
        # img_name = "1689848751402792956.jpg"
        # img_name = "1689848753489529712.jpg"
        # img_name = "1689848776280394106.jpg"
        # img_name = "1689848807256677791.jpg"
        # img_name = "1689848859777111978.jpg"
        # img_name = "1689848861843888811.jpg"
        # img_name = "1695030137658086822.jpg"
        # img_name = "1695030153329805898.jpg"
        # img_name = "1695030190107402825.jpg"
        # img_name = "1695030191324421119.jpg"
        # img_name = "1695030193582731768.jpg"
        # img_name = "1695030356811090637.jpg"
        # img_name = "1695030898755650041.jpg"
        # img_name = "1695031238613012474.jpg"
        img_name = "1695031389933174128.jpg"
        # print(img_name)

        img_path = os.path.join(img_root, img_name)
        json_path = os.path.join(label_root, img_name[:-4] + ".json")

        img = cv2.imread(img_path)

        with open(json_path, "r") as fp:
            labels = json.load(fp)

        lines = []
        all_points = []
        all_attr_type = []
        all_attr_visible = []
        all_attr_hanging = []
        lines_intersect_indexs = []
        for label in labels:
            intersect_index = np.array(label["intersect_index"])

            points = np.array(label["points"])
            attr_type = np.array(label["type"]).reshape(-1, 1)
            attr_visible = np.array(label["visible"]).reshape(-1, 1)
            attr_hanging = np.array(label["hanging"]).reshape(-1, 1)
            # print(points.shape, attr_type.shape, attr_visible.shape, attr_hanging.shape)

            line = np.concatenate([points, attr_type, attr_visible, attr_hanging], axis=1)

            lines.append(line)
            all_points.append(points)
            all_attr_type.append(attr_type)
            all_attr_visible.append(attr_visible)
            all_attr_hanging.append(attr_hanging)
            lines_intersect_indexs.append(intersect_index)

        # img_line = np.ones_like(img) * 255
        img_line = copy.deepcopy(img)
        img_cls = np.ones_like(img) * 255
        img_vis = np.ones_like(img) * 255
        img_hang = np.ones_like(img) * 255

        for points, attr_type, attr_visible, attr_hanging, lines_intersect_index in \
                zip(all_points, all_attr_type, all_attr_visible, all_attr_hanging,  lines_intersect_indexs):
            # 在lines_intersect_index的首尾添加0, -1的index
            # lines_intersect_index = lines_intersect_index.tolist()
            # lines_intersect_index.insert(0, 0)
            # lines_intersect_index.append(-1)

            pre_point = points[0]
            color = (255, 0, 0)
            for i, cur_point in enumerate(points[1:]):
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                cv2.circle(img, (x1, y1), 1, color, 1)
                cv2.line(img, (x1, y1), (x2, y2), color, 3)
                pre_point = cur_point

            pre_point = points[0]
            pre_point_type = attr_type[0]
            pre_point_vis = attr_visible[0]
            pre_point_hang = attr_hanging[0]

            for cur_point, point_type, point_vis, point_hang in zip(points[1:],
                                                                attr_type[1:],
                                                                attr_visible[1:],
                                                                attr_hanging[1:]):
                if point_type not in classes:
                    print("skip point type:", point_type)
                    continue

                # img_line
                color = (255, 0, 0)
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                cv2.circle(img_line, (x1, y1), 1, color, 1)
                cv2.line(img_line, (x1, y1), (x2, y2), color, 3)
                pre_point = cur_point

                # img_cls
                # A ---- B -----C
                # A点的属性代表AB线的属性
                # B点的属性代表BC线的属性
                cls_index = classes.index(pre_point_type)
                color = palette[cls_index]
                cv2.circle(img_cls, (x1, y1), 1, color, 1)
                cv2.line(img_cls, (x1, y1), (x2, y2), color, 3)
                pre_point_type = point_type

                # img_vis
                # point_vis为true的情况下为可见
                color = (0, 255, 0) if pre_point_vis else (255, 0, 0)
                cv2.circle(img_vis, (x1, y1), 1, color, 1)
                cv2.line(img_vis, (x1, y1), (x2, y2), color, 3)
                pre_point_vis = point_vis

                # img_vis
                # point_hang为true的情况为不悬空
                color = (0, 255, 0) if pre_point_hang else (255, 0, 0)
                cv2.circle(img_hang, (x1, y1), 1, color, 1)
                cv2.line(img_hang, (x1, y1), (x2, y2), color, 3)
                pre_point_hang = point_hang

        img_h, img_w, _ = img_line.shape
        img_line = cv2.resize(img_line, (img_w//4, img_h//4))
        s_img_path = os.path.join(dst_root, img_name)
        cv2.imwrite(s_img_path, img_line)

        # plt.imshow(img[:, :, ::-1])
        # plt.show()

        plt.subplot(2, 2, 1)
        plt.imshow(img_line[:, :, ::-1])

        plt.subplot(2, 2, 2)
        plt.imshow(img_cls[:, :, ::-1])

        plt.subplot(2, 2, 3)
        plt.imshow(img_vis[:, :, ::-1])

        plt.subplot(2, 2, 4)
        plt.imshow(img_hang[:, :, ::-1])
        plt.show()
        exit(1)

def debug_signal_filter():
    def moving_average(x, window_size):
        kernel = np.ones(window_size) / window_size
        # kernel = np.array([1, 1, 1])
        result = np.correlate(x, kernel, mode='same')


        # 将边界效应里的设置为原来的数值
        valid_sie = window_size//2
        result[:valid_sie] = x[:valid_sie]
        result[-valid_sie:] = x[-valid_sie:]
        return result

    # 生成一维信号
    # x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x = np.array([1, 3, 5,  7,  9, 10])

    # 应用均值滤波器
    filtered_signal = moving_average(x, window_size=3)

    print(filtered_signal)


def format_kpi_output():
    def get_row_text(row_list):
        text = ""
        for row_t in row_list:
            text = text + row_t
        return text

    def split_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    # kpi_dict = {'all': [0.75, 0.77, 0.7593688362919133], 'road_boundary_line': [0.83, 0.84, 0.8344703770197487], 'bushes_boundary_line': [0.66, 0.72, 0.6881969587255613], 'fence_boundary_line': [-1, -1, -1], 'stone_boundary_line': [0.8, 0.63, 0.7044025157232704], 'wall_boundary_line': [-1, -1, -1], 'water_boundary_line': [-1, -1, -1], 'snow_boundary_line': [-1, -1, -1], 'manhole_boundary_line': [0.5, 0.51, 0.5044510385756678], 'others_boundary_line': [-1, 0.0, -1]}
    # kpi_dict = {'all': [0.88, 0.87, 0.8744717304397487], 'road_boundary_line': [0.88, 0.89, 0.884472049689441], 'bushes_boundary_line': [0.8, 0.81, 0.8044692737430169], 'fence_boundary_line': [-1, -1, -1], 'stone_boundary_line': [0.89, 0.71, 0.7893816364772018], 'wall_boundary_line': [-1, -1, -1], 'water_boundary_line': [-1, -1, -1], 'snow_boundary_line': [-1, -1, -1], 'manhole_boundary_line': [0.71, 0.54, 0.6129496402877699], 'others_boundary_line': [-1, 0.0, -1]}
    # kpi_dict = {'all': [0.9, 0.77, 0.8294434470377021], 'road_boundary_line': [0.94, 0.83, 0.8810841332580462], 'bushes_boundary_line': [0.85, 0.72, 0.779121578612349], 'fence_boundary_line': [-1, -1, -1], 'stone_boundary_line': [0.94, 0.49, 0.6437456324248777], 'wall_boundary_line': [-1, -1, -1], 'water_boundary_line': [-1, -1, -1], 'snow_boundary_line': [-1, -1, -1], 'manhole_boundary_line': [0.81, 0.4, 0.5350949628406277], 'others_boundary_line': [-1, 0.0, -1]}
    # 修复短线的过滤和最大线数量的限制的bug
    # kpi_dict =  {'all': [0.89, 0.79, 0.8365258774538965], 'road_boundary_line': [0.94, 0.84, 0.8866928691746211], 'bushes_boundary_line': [0.82, 0.74, 0.7774503523382448], 'fence_boundary_line': [-1, -1, -1], 'stone_boundary_line': [0.92, 0.57, 0.7034205231388331], 'wall_boundary_line': [-1, -1, -1], 'water_boundary_line': [-1, -1, -1], 'snow_boundary_line': [-1, -1, -1], 'manhole_boundary_line': [0.73, 0.35, 0.4727104532839963], 'others_boundary_line': [0.5, 0.38, 0.43132803632236094]}
    kpi_dict = {'all': [0.87, 0.86, 0.864471403812825], 'road_boundary_line': [0.91, 0.86, 0.8837944664031621], 'bushes_boundary_line': [0.8, 0.79, 0.7944688874921435], 'fence_boundary_line': [-1, -1, -1], 'stone_boundary_line': [0.89, 0.73, 0.8016039481801357], 'wall_boundary_line': [-1, -1, -1], 'water_boundary_line': [-1, -1, -1], 'snow_boundary_line': [-1, -1, -1], 'manhole_boundary_line': [0.63, 0.35, 0.4495412844036697], 'others_boundary_line': [0.7, 0.62, 0.6570779712339139]}


    grid_n = 20       # 控制格子的长度
    row_max_num = 5  # 控制一行里格子的最大数量

    names = list(kpi_dict.keys())
    split_names = split_list(names, row_max_num)
    for names in split_names:
        headers = ["|", "Statics/Category".center(grid_n)]
        row_0 = ["|", "Acc".center(grid_n)]
        row_1 = ["|", "Recall".center(grid_n)]
        row_2 = ["|", "F".center(grid_n)]

        for name in names:
            show_name = name.replace("_line", "")

            kpi = kpi_dict[name]
            acc, recall, F = kpi

            show_name = show_name.center(grid_n)
            acc = str(round(acc, 2)).center(grid_n)
            recall = str(round(recall, 2)).center(grid_n)
            F = str(round(F, 2)).center(grid_n)

            headers.append("|" + show_name)
            row_0.append("|" + acc)
            row_1.append("|" + recall)
            row_2.append("|" + F)

            # 最后格子的右边也加入"|"
            if name == names[-1]:
                headers.append("|")
                row_0.append("|")
                row_1.append("|")
                row_2.append("|")

        text_head = get_row_text(headers)
        text_row_0 = get_row_text(row_0)
        text_row_1 = get_row_text(row_1)
        text_row_2 = get_row_text(row_2)
        line_str = "- " * (len(text_head)//2)

        print(line_str)
        print(text_head)
        print(line_str)
        print(text_row_0)
        print(text_row_1)
        print(text_row_2)

    line_str = "_ " * (len(text_head) // 2)
    print(line_str)

def get_line_splits():
    # line_types = ["1", "1", "1", "2", "2", "3", "3"]
    line_types = ["1", "1", "1", "1", "1", "1", "1"]
    line_points = list(range(len(line_types)))
    pre_index = 0
    # pre_type = line_types[0]
    line_indexs = []  # [(0, 3), (3, 5), (5,7)]
    for cur_index in range(1, len(line_types)):
        if line_types[cur_index] != line_types[pre_index]:
            line_indexs.append((pre_index, cur_index))
            pre_index = cur_index

        if cur_index == len(line_types) - 1:
            line_indexs.append((pre_index, cur_index))
            pre_index = cur_index

    print(line_indexs)
    for line_index in line_indexs:
        id_0, id_1 = line_index
        line_type = line_types[id_0:id_1]

        # 对于属性是添加当前点的属性
        line_type = line_type + [line_types[id_1-1]]

        # 对于点坐标是添加下个点的位置
        line_point = line_points[id_0:id_1]
        line_point = line_point + [line_points[id_1]]
        print(line_type)
        print(line_point)


def debug_line_dist():
    img = np.ones((1860, 2880, 3), dtype=np.uint8) * 255
    thickness = 60
    line_point_0 = (0, 1860//2)
    line_point_1 = (2880, 1860//2)
    cv2.line(img, line_point_0, line_point_1, color=(0, 0, 0), thickness=1)
    cv2.line(img, line_point_0, line_point_1, color=(0, 0, 0), thickness=thickness)

    line_point_0 = (0, 1860//2 + 60)
    line_point_1 = (2880, 1860//2 + 60)
    cv2.line(img, line_point_0, line_point_1, color=(0, 0, 0), thickness=1)
    cv2.line(img, line_point_0, line_point_1, color=(0, 0, 0), thickness=thickness)

    plt.imshow(img)
    plt.show()


def show_line_heatmap():
    # line_heat_map_path = "/home/liyongjing/Downloads/20231027/1695030883240994950_img_heatmamp_line.npy"
    # line_heat_map_path = "/home/liyongjing/Downloads/20231027/1695031238613012474_img_heatmamp_line.npy"
    line_heat_map_path = "/home/liyongjing/Downloads/20231027/1695031405813262853_img_heatmamp_line.npy"
    # line_heat_map_path = "/home/liyongjing/Downloads/20231027/c0bfbf61-2adf-40d3-85b1-ed9332a5fedb_front_camera_9201_img_heatmamp_line.npy"
    line_heat_map = np.load(line_heat_map_path)
    plt.subplot(2, 1, 1)
    plt.imshow(line_heat_map )
    plt.subplot(2, 1, 2)
    plt.imshow(line_heat_map > 0.2)
    plt.show()


def show_img():
    import matplotlib
    # matplotlib.use('Qt5Agg')
    # plt.switch_backend('agg')
    # plt.switch_backend('TkAgg')
    print(matplotlib.pyplot.get_backend())
    img = np.ones((100, 100))
    print(__file__)
    # 172.18.60.136:10.0
    print(os.chdir(os.path.split(__file__)[0]))
    print(os.environ['DISPLAY'])

    plt.imshow(img)
    plt.show()


def debug_dist_seg_loss():
    in_tensor = torch.ones((2, 2, 5, 10))

    # dist_weight = torch.ones_like(in_tensor)
    N, C, H, W = in_tensor.shape

    # min_w = 1
    # max_w = 5  # 实际的数值为max_w - (max_w - min_w)/H
    # dist_weight_col = np.arange(min_w, max_w, (max_w - min_w)/H)
    # dist_weight = np.repeat(dist_weight_col, W).reshape(H, W)

    # torch
    min_w = 1
    max_w = 5  # 实际的数值为max_w - (max_w - min_w)/H
    dist_weight_col = torch.arange(min_w, max_w, (max_w - min_w)/H)
    dist_weight = dist_weight_col.repeat(1, W).reshape(W, H).transpose(1, 0)

    print(torch.is_tensor(in_tensor))
    print(torch.is_tensor(dist_weight))

    in_tensor.requires_grad = True
    print(in_tensor.requires_grad)
    print(dist_weight.requires_grad)

    in_tensor = in_tensor * dist_weight

    in_tensor_np = in_tensor.detach().numpy()
    plt.imshow(in_tensor_np[0][1])
    plt.show()


def debug_show_img_quick():
    import time
    import matplotlib.pyplot as plt

    num_plots = 100
    for i in range(num_plots):
        print(i)
        # 绘制图像
        x = [0, 1]
        y = [0, 1]
        plt.plot(x, y)

        # 显示图像
        plt.show()
        time.sleep(1)
        plt.close('all')

def debug_sort_points():
    def sort_point_by_x_y(image_xs, image_ys):
        # 按照 x 坐标进行从小到大排序，在 x 相同时按照 y 坐标从大到小进行排序
        sorted_indices = np.lexsort((image_ys, image_xs))
        return sorted_indices

    def sort_point_by_x_inverse_y(image_xs, image_ys):
        # 按照 x 坐标进行从小到大排序，在 x 相同时按照 y 坐标从大到小进行排序
        sorted_indices = np.lexsort((-image_ys, image_xs))
        return sorted_indices

    def sort_point_by_x_y(image_xs, image_ys):
        indexs = image_xs.argsort()
        image_xs_sort = image_xs[indexs]
        image_ys_sort = image_ys[indexs]

        # 将第一个点加入
        x_same = image_xs_sort[0]              # 判断是否为同一个x
        y_pre = image_ys_sort[0]               # 上一个不是同一个相同x的点的y

        indexs_with_same_x = [indexs[0]]       # 记录相同的x的index
        ys_with_same_x = [image_ys_sort[0]]    # 记录具有相同x的ys

        new_indexs = []
        for i, (idx, x_s, y_s) in enumerate(zip(indexs[1:], image_xs_sort[1:], image_ys_sort[1:])):
            if x_s == x_same:
                indexs_with_same_x.append(idx)
                ys_with_same_x.append(y_s)
            else:
                # 如果当前xs与前面的不一样，将前面的进行截断统计分析
                # 对y进行排序， 需要判断y是从大到小还是从小到大排序, 需要跟上一个x对应的y来判断，距离近的排在前面
                # 首先按照从小到大排序
                index_y_with_same_x = np.array(ys_with_same_x).argsort()

                # 判断是否需要倒转过来排序
                if len(index_y_with_same_x) > 1:
                    if abs(index_y_with_same_x[-1] - y_pre) < abs(index_y_with_same_x[0] - y_pre):
                        index_y_with_same_x = index_y_with_same_x[::-1]

                new_indexs = new_indexs + np.array(indexs_with_same_x)[index_y_with_same_x].tolist()

                # 为下次的判断作准备
                y_pre = ys_with_same_x[index_y_with_same_x[-1]]
                x_same = x_s
                indexs_with_same_x = [idx]
                ys_with_same_x = [y_s]

            if i == len(image_xs) - 2:    # 判断是否为最后一个点
                index_y_with_same_x = np.array(ys_with_same_x).argsort()
                if len(index_y_with_same_x) > 1:
                    if abs(index_y_with_same_x[-1] - y_pre) < abs(index_y_with_same_x[0] - y_pre):
                        index_y_with_same_x = index_y_with_same_x[::-1]
                new_indexs = new_indexs + np.array(indexs_with_same_x)[index_y_with_same_x].tolist()
        return new_indexs

    def sort_point_by_x_y2(image_xs, image_ys):
        indexs = image_xs.argsort()
        # image_xs_sort = image_xs[indexs]
        image_ys_sort = image_ys[indexs]

        # 判断线的方向主体方向
        y_direct = image_ys_sort[0] - image_ys_sort[-1]
        if y_direct < 0:
            # 按照 x 坐标进行从小到大排序，在 x 相同时按照 y 坐标从大到小进行排序
            sorted_indices = np.lexsort((image_ys, image_xs))
        else:
            # 按照 x 坐标进行从小到大排序，在 x 相同时按照 y 坐标从大到小进行排序
            sorted_indices = np.lexsort((-image_ys, image_xs))
        return sorted_indices


    x = np.array([2, 2, 2, 4, 4,  1])
    y = np.array([3, 2, 1, 3, 4, 1])
    indices2 = sort_point_by_x_inverse_y(x, y)
    print("indices2", indices2)

    new_indexs = sort_point_by_x_y(x, y)
    print(new_indexs)
    print(x[new_indexs])
    print(y[new_indexs])


def debug_discrimate():
    # embedding_i = embedding_b * seg_mask_i
    embedding_b = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)   # 2是维度，3是个数
    seg_mask_i = np.array([1, 0, 1])
    embedding_i = embedding_b * seg_mask_i
    mean_i = np.sum(embedding_i, axis=1) / np.sum(seg_mask_i)
    print(embedding_i)
    print(mean_i)

    a = torch.randn(4, 4)
    print(torch.sum(a, 1))
    print(torch.sum(a))

    b = a > 0.2
    print(torch.sum(b, 1))
    print(torch.sum(b))

import torch.nn.functional as F
def debug_emb_dist():
    embedding_i = torch.tensor([[1, 2], [1, 3], [2, 4], [0, 0]], )
    mean_i = torch.tensor([1, 2])
    embed_dim = 2
    print(embedding_i.shape)
    print(mean_i.shape)

    embedding_i = torch.transpose(embedding_i, 1, 0)

    embedding_diff = embedding_i - mean_i.reshape(embed_dim, 1)
    # [0, 0], [0, 1], [1, 2], [-1, -2], shape为[2, 4] 前面为特征维度，后面为特征个数
    embedding_diff = embedding_diff.float()
    emb_norm = torch.norm(embedding_diff, dim=0)   # 默认范数为2 # (1 * 1 + 2 * 2)，然后开平方

    emb_norm_np = np.linalg.norm(embedding_diff.numpy(), ord=2, axis=0)

    a = np.array([1, 2, -4, 6])
    b = np.linalg.norm(a, ord=2)

    delta_var = 0.5
    # [0, 1, 2.2361, 2.2361] - 0.5 = -0.5, 0.5, 1.736, 1.736
    # relu 后 0， 0.5， 1.736，1.736, 距离小于0.5个都loss为0
    # 平方后0， 0.25， 3.0139， 3.0139，距离越大的放大
    # 求平均后为1.5695
    loss_var = torch.mean(F.relu(emb_norm - delta_var) ** 2)


def debug_split_line_in_random_crop():
    mask = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    mask = mask > 0
    line_indexs = []
    start_indx = None
    for i, _mask in enumerate(mask):
        if _mask and start_indx is None:
            start_indx = i
        elif (not _mask) and start_indx is not None:
            end_indx = i - 1
            line_indexs.append((start_indx, end_indx))
            start_indx = None

        if i == len(mask) - 1:
            end_indx = i
            if start_indx is not None:
                line_indexs.append((start_indx, end_indx))

    print(line_indexs)

def debug_sigmoid_and_sigmoid_inverse():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_inverse(p):
        return np.log(p / (1 - p))

    a = np.array([2, 3, 4])
    b = sigmoid_inverse(sigmoid(a))
    print(a, b)


def filter_lines():
    line_points = np.array([[0, 2], [0, 4]])
    points_0 = line_points[:-1]
    points_1 = line_points[1:]

    points_dist = np.sqrt((points_0[:, 0] - points_1[:, 0]) ** 2
                          + (points_0[:, 1] - points_1[:, 1]) ** 2)
    if points_dist < 10:
        print("Filter")
    print(points_dist)


def find_certain_class_data():
    # root = "/home/liyongjing/Egolee/hdd-data/data/label_datas/gbld_20231202_v0.3"
    root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20240102_mmdet3d_spline_by_cls/train"
    img_root = os.path.join(root, "images")
    json_root = os.path.join(root, "jsons")

    s_root = "/home/liyongjing/Downloads/debug_1"
    for dir in ['images', 'jsons']:
        dir_path = os.path.join(s_root, dir)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)

    json_names = [name for name in os.listdir(json_root) if name.endswith('.json')]
    for json_name in tqdm(json_names, desc="json_names"):
        img_path = os.path.join(img_root, json_name[:-5] + ".jpg")
        json_path = os.path.join(json_root, json_name)
        with open(json_path, "r") as fp:
            labels = json.load(fp)

        has_certain_class = False
        for label in labels['shapes']:
            # intersect_index = np.array(label["intersect_index"])
            points = np.array(label["points"])
            attr_type = np.array(label["points_type"]).reshape(-1, 1)
            attr_visible = np.array(label["points_visible"]).reshape(-1, 1)
            attr_hanging = np.array(label["points_hanging"]).reshape(-1, 1)
            attr_covered = np.array(label["points_covered"]).reshape(-1, 1)

            if "others_boundary_line" in attr_type:
                has_certain_class = True
                break

        if has_certain_class:
            dst_img_path = os.path.join(s_root, "images", json_name[:-5] + ".jpg")
            dst_json_path = os.path.join(s_root, "jsons", json_name)

            shutil.copy(img_path, dst_img_path)
            shutil.copy(json_path, dst_json_path)


def debug_parse_dataset_json():
    root = os.path.join("/home/liyongjing/Downloads/debug_1")

    img_root = os.path.join(root, "images")
    label_root = os.path.join(root, "jsons")

    dst_root = img_root + "_debug_vis"
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    down_scale = 1

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", '.png']]
    for img_name in tqdm(img_names, desc="img_names"):
        # img_name = "1696991193.66982.jpg"
        # img_name = "1689848745238965269.jpg"
        # img_name = "1700108695.418044.jpg"
        # print("img_name:", img_name)
        img_path = os.path.join(img_root, img_name)
        json_path = os.path.join(label_root, img_name[:-4] + ".json")

        img = cv2.imread(img_path)
        # img = np.ones_like(img) * 255

        with open(json_path, "r") as fp:
            labels = json.load(fp)

        lines = []
        all_points = []
        all_points_type = []
        all_points_visible = []
        all_points_hanging = []
        all_points_covered = []
        lines_intersect_indexs = []
        for label in labels["shapes"]:
            # intersect_index = np.array(label["intersect_index"])

            points = np.array(label["points"])
            points_type = np.array(label["points_type"]).reshape(-1, 1)
            points_visible = np.array(label["points_visible"]).reshape(-1, 1)
            # print(points_visible)
            points_hanging = np.array(label["points_hanging"]).reshape(-1, 1)
            points_covered = np.array(label["points_covered"]).reshape(-1, 1)
            # print(points.shape, attr_type.shape, attr_visible.shape, attr_hanging.shape)

            line = np.concatenate([points, points_type, points_visible, points_hanging, points_covered], axis=1)

            lines.append(line)
            all_points.append(points)
            all_points_type.append(points_type)
            all_points_visible.append(points_visible)
            all_points_hanging.append(points_hanging)
            all_points_covered.append(points_covered)

        # img_line = np.ones_like(img) * 255
        img_h, img_w, img_c = img.shape
        img_h, img_w = img_h//down_scale, img_w//down_scale
        img = cv2.resize(img, (img_w, img_h))

        img_line = np.ones((img_h, img_w, img_c), dtype=np.uint8) * 255
        # img_line = copy.deepcopy(img)

        img_cls = np.ones_like(img_line) * 255
        img_vis = np.ones_like(img_line) * 255
        img_hang = np.ones_like(img_line) * 255
        img_covered = np.ones_like(img_line) * 255

        line_count = 0

        for points, points_type, points_visible, points_hanging, points_covered in \
                zip(all_points, all_points_type, all_points_visible, all_points_hanging, all_points_covered):
            # 在lines_intersect_index的首尾添加0, -1的index
            # lines_intersect_index = lines_intersect_index.tolist()
            # lines_intersect_index.insert(0, 0)
            # lines_intersect_index.append(-1)

            if "others_boundary_line" not in points_type:
                continue

            pre_point = points[0]
            # color = (255, 0, 0)
            color = (0, 0, 255)
            for i, cur_point in enumerate(points[1:]):
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                x1, y1 = x1//down_scale, y1//down_scale
                x2, y2 = x2//down_scale, y2//down_scale

                cv2.circle(img, (x1, y1), 1, color, 1)
                cv2.line(img, (x1, y1), (x2, y2), color, 3)
                pre_point = cur_point

            pre_point = points[0]
            pre_point_type = points_type[0]
            pre_point_vis = points_visible[0]
            pre_point_hang = points_hanging[0]
            pre_point_covered = points_covered[0]

            First_Point = True
            for cur_point, point_type, point_vis, point_hang, point_covered in zip(points[1:],
                                                                points_type[1:],
                                                                points_visible[1:],
                                                                points_hanging[1:],
                                                                points_covered[1:]):
                if point_type not in classes:
                    print("skip point type:", point_type)
                    continue

                # img_line
                # color = (255, 0, 0)
                color = color_list[line_count % len(color_list)]
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                x1, y1 = x1 // down_scale, y1 // down_scale
                x2, y2 = x2 // down_scale, y2 // down_scale

                cv2.circle(img_line, (x1, y1), 1, color, 1)
                cv2.line(img_line, (x1, y1), (x2, y2), color, 3)
                if First_Point:
                    center_point = points[len(points)//2]
                    cv2.putText(img_line, str(line_count), (int(center_point[0]), int(center_point[1])),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 12)
                    First_Point = False

                pre_point = cur_point
                # 点的属性全都是字符类型
                # img_cls
                # A ---- B -----C
                # A点的属性代表AB线的属性
                # B点的属性代表BC线的属性
                cls_index = classes.index(pre_point_type[0])
                color = color_list[cls_index]
                cv2.circle(img_cls, (x1, y1), 1, color, 1)
                cv2.line(img_cls, (x1, y1), (x2, y2), color, 3)
                pre_point_type = point_type

                # img_vis
                # 绿色为true, 蓝色为false
                # point_vis为true的情况下为可见
                color = (0, 255, 0) if pre_point_vis[0] == "true" else (255, 0, 0)
                cv2.circle(img_vis, (x1, y1), 1, color, 1)
                cv2.line(img_vis, (x1, y1), (x2, y2), color, 3)
                pre_point_vis = point_vis

                # img_hang
                # point_hang为true的情况为悬空
                color = (0, 255, 0) if pre_point_hang[0] == "true" else (255, 0, 0)
                cv2.circle(img_hang, (x1, y1), 1, color, 1)
                cv2.line(img_hang, (x1, y1), (x2, y2), color, 3)
                pre_point_hang = point_hang

                # img_covered
                # point_covered为true的情况为被草遮挡
                color = (0, 255, 0) if pre_point_covered[0] == "true" else (255, 0, 0)
                cv2.circle(img_covered, (x1, y1), 1, color, 1)
                cv2.line(img_covered, (x1, y1), (x2, y2), color, 3)
                pre_point_covered = point_covered

            line_count = line_count + 1

        img_h, img_w, _ = img_line.shape
        # img_line = cv2.resize(img_line, (img_w//4, img_h//4))
        s_img_path = os.path.join(dst_root, img_name)
        # cv2.imwrite(s_img_path, img_line)
        cv2.imwrite(s_img_path, img)
        # exit(1)


def get_label_url():
    img_root = "/home/liyongjing/Downloads/debug_1/images_debug_vis"
    json_root = "/home/liyongjing/Egolee/hdd-data/data/label_datas/gbld_20231202_v0.3/json"

    s_root = img_root + "_url"

    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)

    img_names = [name for name in os.listdir(img_root) if name.endswith('.jpg')]
    for img_name in tqdm(img_names, desc="json_names"):
        json_path = os.path.join(json_root, img_name[:-4] + ".json")
        img_path = os.path.join(img_root, img_name)

        with open(json_path, "r") as fp:
            labels = json.load(fp)
        url = labels['url']
        s_url_path = os.path.join(s_root, img_name[:-4] + ".txt" )
        with open(s_url_path, 'w') as fp:
            fp.write(url)
        exit(1)


if __name__ == "__main__":
    print("Start")
    # test_mmdet_fpn()
    # debug_write_data_infos()
    # test_dtype()
    # debug_sum()
    # debug_padd()

    # 调试标注系统解析的json文件
    # debug_parse_label_json()

    # 对一维信号进行平滑处理
    #debug_signal_filter()

    # 对kpi的输出进行格式的转换
    # format_kpi_output()

    # 根据类别将线段分段
    # get_line_splits()

    # debug_line_dist()

    # show_line_heatmap()

    # show_img()

    # 加强近处的seg losss
    # debug_dist_seg_loss()

    # 在循环中快速显示图像
    # debug_show_img_quick()

    # debug_sort_points()

    # 调试discrimate聚类loss
    # debug_discrimate()
    # debug_emb_dist()

    # 调试crop数据增强中的分段mask
    # debug_split_line_in_random_crop()

    # 调试sigmoid和sigmoid的逆操作
    # debug_sigmoid_and_sigmoid_inverse()

    # 调试曲线过滤
    # filter_lines()

    # 筛选包含特定信息的标注文件以及可视化
    # find_certain_class_data()
    # debug_parse_dataset_json()
    get_label_url()

    print("End")