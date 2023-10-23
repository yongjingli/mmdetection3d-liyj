import os
import shutil
import cv2
import json
import copy
from tqdm import tqdm
import numpy as np
import mmengine
import matplotlib.pyplot as plt
from mmengine.fileio import join_path, list_from_file, load
from dense_line_points import dense_line_points, dense_line_points_by_interp, dense_line_points_by_interp_with_attr


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (221, 160, 221),  (128, 0, 128), (203, 192, 255), (238, 130, 238), (0, 69, 255),
              (130, 0, 75), (255, 255, 0), (250, 51, 153), (214, 112, 218), (255, 165, 0),
              (169, 169, 169), (18, 74, 115),
              (240, 32, 160), (192, 192, 192), (112, 128, 105), (105, 128, 128),
              ]


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

classes = list(TYPE_DICT.values())


def draw_curce_line_on_img(img, points, cls_name, color=(0, 255, 0)):
    pre_point = points[0]
    for i, cur_point in enumerate(points[1:]):
        x1, y1 = int(pre_point[0]), int(pre_point[1])
        x2, y2 = int(cur_point[0]), int(cur_point[1])

        # cv2.circle(img, (x1, y1), 1, color, 1)
        cv2.line(img, (x1, y1), (x2, y2), color, 3)
        pre_point = cur_point
        # show order
        # if i % 100 == 0:
        #     img = cv2.putText(img, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    txt_i = len(points) // 2
    txt_x = int(points[txt_i][0])
    txt_y = int(points[txt_i][1])
    # img = cv2.putText(img, cls_name, (txt_y, txt_x), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)

    return img


def parse_v02_data_2_lld_data(base_infos):
    # cvat_xml_path = "/media/dell/Elements SE/liyj/data/collect_data_20230720/rosbag2_2023_07_20-18_24_24/glass_edge_overfit_20230721/annotations.xml"
    # cvat_xml_path = base_infos["cvat_xml_path"]

    src_root = base_infos["src_root"]
    dst_root = base_infos["dst_root"]
    dst_w = base_infos["dst_w"]
    dst_h = base_infos["dst_h"]
    max_num = base_infos["max_num"]
    dense_points = base_infos["dense_points"]

    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    for tmp in ['train', 'test']:
        tmp_path = os.path.join(dst_root, tmp)
        os.mkdir(tmp_path)

    for tmp in ['images', 'jsons']:
        os.mkdir(os.path.join(dst_root, 'train', tmp))
        os.mkdir(os.path.join(dst_root, 'test', tmp))

    img_root = os.path.join(src_root, "images")
    label_root = os.path.join(src_root, "json")
    image_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", '.png']]

    # from xml.dom.minidom import parse
    # import xml.dom.minidom
    #
    # # 使用minidom解析器打开 XML 文档
    # DOMTree = xml.dom.minidom.parse(cvat_xml_path)
    # collection = DOMTree.documentElement
    #
    # images = collection.getElementsByTagName("image")
    if max_num != -1:
        image_names = image_names[:max_num]

    count = 0
    for image_name in tqdm(image_names):
        test = True if count % 10 == 0 else False
        # test = False
        root_name = "test" if test else "train"
        s_root = os.path.join(dst_root, root_name)
        count = count + 1

        # image_name = image.getAttribute("name")
        # # # TODO
        # if image_name != "1689848680876640844.jpg":
        #     continue
        # print(image_name)
        # exit(1)
        json_path = os.path.join(label_root, image_name[:-4] + ".json")
        with open(json_path, "r") as fp:
            labels = json.load(fp)

        project_id = labels['project_id']
        project_name = labels['project_name']
        task_id = labels['task_id']
        task_name = labels['task_name']
        job_id = labels['job_id']
        url = labels['url']
        frame_index = labels['frame_index']
        data = labels['data']

        poly_lines = []
        # poly_lines_type = []        # 点的类别
        # poly_lines_visible = []     # 点的可见性
        # poly_lines_hanging = []     # 点的悬空属性
        # poly_lines_covered = []     # 点的是否被草挡住的属性
        for label in labels['data']:
            intersect_index = np.array(label["intersect_index"])
            points = np.array(label["points"])
            attr_type = np.array(label["type"]).reshape(-1, 1)
            attr_visible = np.array(label["visible"]).reshape(-1, 1)
            attr_hanging = np.array(label["hanging"]).reshape(-1, 1)
            attr_covered = np.array(label["covered_by_grass"]).reshape(-1, 1)

            # 投票得到线的整体类别
            attr_type_list = attr_type.reshape(-1).tolist()
            poly_line_name = max(attr_type_list, key=attr_type_list.count)
            poly_line = {"points": points,
                         "label": poly_line_name,          # 线的类别
                         "points_type": attr_type,         # 点的类别
                         "points_visible": attr_visible,   # 点的可见性
                         "points_hanging": attr_hanging,   # 点的悬空属性
                         "points_covered": attr_covered,   # 点的是否被草挡住的属性
                         }

            poly_lines.append(poly_line)
            # print(points.shape, attr_type.shape, attr_visible.shape, attr_hanging.shape)
            # line = np.concatenate([points, attr_type, attr_visible, attr_hanging], axis=1)
            #
            # lines.append(line)
            # all_points.append(points)
            # all_attr_type.append(attr_type)
            # all_attr_visible.append(attr_visible)
            # all_attr_hanging.append(attr_hanging)
            # lines_intersect_indexs.append(intersect_index)



        # poly_lines = image.getElementsByTagName('polyline')

        # 解析标注数据,得到curve_lines
        # poly_line_pts, curve_id分别代表当前曲线的点, 当前曲线是否为同一曲线
        curve_lines = []
        poly_line_names = {}
        curve_count = 1
        for i, poly_line in enumerate(poly_lines):
            # poly_line_name = poly_line.getAttribute("label")
            poly_line_name = poly_line["label"]

            # 进行名称的转换
            if poly_line_name not in TYPE_DICT.keys() and poly_line_name not in TYPE_DICT.values():
                continue

            if poly_line_name in TYPE_DICT.keys():
                poly_line_name = TYPE_DICT[poly_line_name]


            # poly_line_pts = poly_line.getAttribute("points")
            # poly_line_pts = poly_line.getAttribute("points")
            poly_line_pts = poly_line["points"]

            # curve_id = poly_line.getAttribute("id")
            # attr_name = poly_line.getElementsByTagName("attribute")[0].getAttribute("name")
            # curve_id = int(poly_line.getElementsByTagName("attribute")[0].childNodes[0].nodeValue)
            # print(attr_name, curve_id)
            curve_id = i

            points_type = poly_line["points_type"]          # 点的类别
            points_visible = poly_line["points_visible"]   # 点的可见性
            points_hanging = poly_line["points_hanging"]   # 点的悬空属性
            points_covered = poly_line["points_covered"]   # 点的是否被草挡住的属性

            curve_lines.append([poly_line_pts, curve_id, poly_line_name,
                                points_type, points_visible, points_hanging, points_covered])

        # 读取img的信息, 并缩放为指定尺寸
        img_path = os.path.join(img_root, image_name)
        # print(img_path)
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape

        if dst_w == -1 or dst_h == -1:
            dst_w = img_w
            dst_h = img_h

        img_res = cv2.resize(img, (dst_w, dst_h))
        sx, sy = dst_w / img_w, dst_h / img_h

        # 保存标注文件的信息内容
        new_json = {
            'version': '4.5.6',
            'flags': {},
            'shapes': [],
            'imagePath': image_name,   #image_name
            'imageData': None,
            'imageHeight': dst_w,
            'imageWidth': dst_h,
        }

        # 对曲线进行scale
        for curve_line in curve_lines:
            points, cls, cls_name, points_type, points_visible, points_hanging, points_covered = curve_line
            # points， 每个标注为单独的曲线聚类
            # cls， 用来聚类是否为同一曲线，如同一条曲线被遮挡分段
            # cls_name， 包括三种类别"glass_edge", "glass_edge_up_plant", "glass_edge_up_build"

            # scale points
            points[:, 0] = np.clip(points[:, 0] * sx, 0, dst_w - 1e-4)
            points[:, 1] = np.clip(points[:, 1] * sy, 0, dst_h - 1e-4)

            # 将点进行稠密化
            if dense_points:
                # points = dense_line_points(img_res, points)
                # points = dense_line_points_by_interp(img_res, points)
                points, points_type, points_visible, \
                points_hanging, points_covered = dense_line_points_by_interp_with_attr(img_res, points,
                                                               points_type, points_visible,
                                                               points_hanging, points_covered)

            item = {
                "label": cls_name,
                "points": points.tolist(),
                "points_type": points_type.tolist(),
                "points_visible": points_visible.tolist(),
                "points_hanging": points_hanging.tolist(),
                "points_covered": points_covered.tolist(),
                "group_id": None,
                "shape_type": "line",
                "flag": {},
                "id": cls,
            }

            new_json['shapes'].append(item)

        dst_json_path = os.path.join(s_root, "jsons", image_name[:-4] + ".json")
        with open(dst_json_path, 'w') as sf:
            json.dump(new_json, sf)

        dst_png_path = os.path.join(s_root, "images", image_name[:-4] + ".jpg")
        cv2.imwrite(dst_png_path, img_res)
        # print(dst_png_path)

        # # debug
        # for i, curve_line in enumerate(curve_lines):
        #     points, cls, cls_name, points_type, points_visible, points_hanging, points_covered = curve_line
        #     color = color_list[cls]
            # cls_name = cls_names[cls]
            # cls_name = "glass_edge"
            # img_show = draw_curce_line_on_img(img_res, points, cls_name, color)
            # print(img_show.shape)

        # cv2.namedWindow("img_show", 0)
        # cv2.namedWindow("img_show", cv2.WINDOW_NORMAL)
        # cv2.imshow("img_show", img_show)
        # cv2.waitKey(0)
        # exit(1)


def generate_mmdet3d_trainval_infos(generate_dataset_infos):
    metainfo = generate_dataset_infos["metainfo"]
    dst_root = generate_dataset_infos["dst_root"]

    for subset in ["train", "test"]:
        subset_metainfo = copy.deepcopy(metainfo)
        subset_metainfo['subset'] = subset
        imgs_root = os.path.join(dst_root, subset, "images")
        img_names = [name for name in os.listdir(imgs_root) if name[-4:] in [".jpg", ".png"]]
        img_paths = []
        json_paths = []
        data_list = []
        for img_name in img_names:
            # debug TODO
            # if subset == "train":
            #     if img_name != "1689848680876640844.jpg":
            #         continue

            b_path_exit_check = True

            img_path = "/".join([subset, "images", img_name])
            if not os.path.exists(os.path.join(dst_root, img_path)):
                print("img path not exit:{}".format(img_path))
                b_path_exit_check = False

            json_name = img_name[:-4] + ".json"
            json_path = "/".join([subset, "jsons", json_name])
            if not os.path.exists(os.path.join(dst_root, json_path)):
                print("json path not exit:{}".format(json_path))
                b_path_exit_check = False

            if not b_path_exit_check:
                assert b_path_exit_check, "path check fail"

            sample = {"img": img_name, "ann": json_name}

            data_list.append(sample)

        converted_data_info = dict(metainfo=metainfo, data_list=data_list)

        out_path = os.path.join(dst_root, "gbld_infos_{}.pkl".format(subset))
        mmengine.dump(converted_data_info, out_path, 'pkl')

        # debug
        annotations = load(out_path)
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        print("metainfo:", metainfo)
        print("raw_data_list:", raw_data_list)


def combine_images_jsons():
    root = "/media/dell/Egolee1/liyj/data/label_data/gbld_from_label_system/gbld_20231012_v0.2"
    # 16  17  22  23  24  25  26
    sub_dirs = ["16",  "17",  "22",  "23",  "24",  "25",  "26"]

    dst_dir = "all"
    dst_root = os.path.join(root, dst_dir)
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    names = ["images", "json"]
    for name in names:
        os.mkdir(os.path.join(dst_root, name))

    for sub_dir in tqdm(sub_dirs, desc=sub_dirs):
        src_root = os.path.join(root, sub_dir)
        for name in names:
            src_root2 = os.path.join(src_root, name)
            dst_root2 = os.path.join(dst_root, name)

            for file_name in os.listdir(src_root2):
                shutil.copy(os.path.join(src_root2, file_name),
                            os.path.join(dst_root2, file_name))


def debug_parse_dataset_json(generate_dataset_infos):
    root = os.path.join(generate_dataset_infos["dst_root"], 'train')


    img_root = os.path.join(root, "images")
    label_root = os.path.join(root, "jsons")

    dst_root = img_root + "_debug_vis"
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", '.png']]
    for img_name in tqdm(img_names, desc="img_names"):
        # img_name = "1696991348.057879.jpg"
        print("img_name:", img_name)
        img_path = os.path.join(img_root, img_name)
        json_path = os.path.join(label_root, img_name[:-4] + ".json")

        img = cv2.imread(img_path)

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
        img_line = copy.deepcopy(img)
        img_cls = np.ones_like(img) * 255
        img_vis = np.ones_like(img) * 255
        img_hang = np.ones_like(img) * 255
        img_covered = np.ones_like(img) * 255

        for points, points_type, points_visible, points_hanging, points_covered in \
                zip(all_points, all_points_type, all_points_visible, all_points_hanging, all_points_covered):
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
            pre_point_type = points_type[0]
            pre_point_vis = points_visible[0]
            pre_point_hang = points_hanging[0]
            pre_point_covered = points_covered[0]

            for cur_point, point_type, point_vis, point_hang, point_covered in zip(points[1:],
                                                                points_type[1:],
                                                                points_visible[1:],
                                                                points_hanging[1:],
                                                                points_covered[1:]):
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
        # plt.imshow(img_hang[:, :, ::-1])
        plt.imshow(img_covered[:, :, ::-1])
        plt.show()
        # exit(1)


if __name__ == "__main__":
    # 在v0.2的版本中,采用分段标注的方式和分配属性的方式
    # 属性包括可见属性、被草遮挡和悬空属性等

    # 用来生成数据集和相应的infos
    generate_dataset_infos = {
        "type": 'image',
        # "dst_w": 1920,
        # "dst_h": 1080,
        "dst_w": -1,     # 不对图像进行resize等操作
        "dst_h": -1,
        "max_num": -1,  # default -1, 全部的数据

        "metainfo": {
            'dataset': 'GbldMono2dDataset',
            'info_version': '1.0',
            'data': "2023-10-17",
            },

        "dense_points": True,

        # "src_root": "/media/dell/Egolee1/liyj/data/ros_bags/collect_data_20230720/rosbag2_2023_07_20-18_24_24/glass_edge_overfit_20230721",
        # "dst_root": "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230926_mmdet3d",
        # "cvat_xml_path": "/home/dell/下载/2/annotations.xml"

        #  20231013
        # "src_root": "/media/dell/Egolee1/liyj/data/label_data/gbld_from_label_system/gbld_20231012_v0.2/28",
        # "dst_root": "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231017_mmdet3d_debug",
        # "cvat_xml_path": "/media/dell/Egolee1/liyj/label_data/20230926/dataset_29/annotations.xml",

        #  20231013
        "src_root": "/media/dell/Egolee1/liyj/data/label_data/gbld_from_label_system/gbld_20231020_v0.2",
        "dst_root": "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231020_mmdet3d",

        # 服务器
        #"src_root": "/data-hdd/liyj/data/label_data/20230926/dataset_27/images",
        #"dst_root": "/data-hdd/liyj/data/dataset/glass_edge_overfit_20230927_mmdet3d",
        #"cvat_xml_path": "/data-hdd/liyj/data/label_data/20230926/dataset_27/annotations.xml"
        }

    # 将文件整理到一个文件夹中
    # combine_images_jsons()

    # 转成dataset数据格式
    # parse_v02_data_2_lld_data(generate_dataset_infos)

    # 可视化生成json文件
    debug_parse_dataset_json(generate_dataset_infos)

    # 生成mmdet3d数据加载的相关信息
    # generate_mmdet3d_trainval_infos(generate_dataset_infos)

