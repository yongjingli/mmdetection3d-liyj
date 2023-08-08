import os
import shutil
import cv2
import json
import copy
from tqdm import tqdm
import numpy as np
import mmengine
from mmengine.fileio import join_path, list_from_file, load


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (221, 160, 221),  (128, 0, 128), (203, 192, 255), (238, 130, 238), (0, 69, 255),
              (130, 0, 75), (255, 255, 0), (250, 51, 153), (214, 112, 218), (255, 165, 0),
              (169, 169, 169), (18, 74, 115),
              (240, 32, 160), (192, 192, 192), (112, 128, 105), (105, 128, 128),
              ]


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


def parse_cvat_xml_2_lld_data(base_infos):
    cvat_xml_path = "/media/dell/Elements SE/liyj/data/collect_data_20230720/rosbag2_2023_07_20-18_24_24/glass_edge_overfit_20230721/annotations.xml"

    src_root = base_infos["src_root"]
    dst_root = base_infos["dst_root"]
    dst_w = base_infos["dst_w"]
    dst_h = base_infos["dst_h"]
    max_num = base_infos["max_num"]

    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    for tmp in ['train', 'test']:
        tmp_path = os.path.join(dst_root, tmp)
        os.mkdir(tmp_path)

    for tmp in ['images', 'jsons']:
        os.mkdir(os.path.join(dst_root, 'train', tmp))
        os.mkdir(os.path.join(dst_root, 'test', tmp))

    from xml.dom.minidom import parse
    import xml.dom.minidom

    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(cvat_xml_path)
    collection = DOMTree.documentElement

    images = collection.getElementsByTagName("image")
    if max_num != -1:
        images = images[:max_num]

    count = 0
    for image in tqdm(images):
        test = True if count % 10 == 0 else False
        # test = False
        root_name = "test" if test else "train"
        s_root = os.path.join(dst_root, root_name)
        count = count + 1

        image_name = image.getAttribute("name")

        poly_lines = image.getElementsByTagName('polyline')

        # 解析标注数据,得到curve_lines
        # poly_line_pts, curve_id分别代表当前曲线的点, 当前曲线是否为同一曲线
        curve_lines = []
        poly_line_names = {}
        curve_count = 1
        for poly_line in poly_lines:
            poly_line_name = poly_line.getAttribute("label")
            # poly_line_pts = poly_line.getAttribute("points")
            poly_line_pts = poly_line.getAttribute("points")

            # curve_id = poly_line.getAttribute("id")
            attr_name = poly_line.getElementsByTagName("attribute")[0].getAttribute("name")
            curve_id = int(poly_line.getElementsByTagName("attribute")[0].childNodes[0].nodeValue)
            print(attr_name, curve_id)

            # 数据解析为numpy的形式
            poly_line_pts_np = []
            for poly_line_pt in poly_line_pts.split(";"):
                pts = [float(d) for d in poly_line_pt.split(",")]
                poly_line_pts_np.append(pts)
            poly_line_pts_np = np.array(poly_line_pts_np)
            poly_line_pts = poly_line_pts_np

            # if poly_line_name not in poly_line_names.keys():
            #     curve_id = curve_count
            #     poly_line_names[poly_line_name] = curve_count
            #
            #     curve_count += 1
            # else:
            #     curve_id = poly_line_names[poly_line_name]

            print(poly_line_name, curve_id)

            curve_lines.append([poly_line_pts, curve_id, poly_line_name])

        # 读取img的信息, 并缩放为指定尺寸
        img_path = os.path.join(src_root, image_name)
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
            points, cls, cls_name = curve_line
            # points， 每个标注为单独的曲线聚类
            # cls， 用来聚类是否为同一曲线，如同一条曲线被遮挡分段
            # cls_name， 包括三种类别"glass_edge", "glass_edge_up_plant", "glass_edge_up_build"

            # scale points
            points[:, 0] = np.clip(points[:, 0] * sx, 0, dst_w - 1e-4)
            points[:, 1] = np.clip(points[:, 1] * sy, 0, dst_h - 1e-4)

            item = {
                "label": cls_name,
                "points": points.tolist(),
                "group_id": None,
                "shape_type": "line",
                "flag": {},
                "id": cls,
            }

            new_json['shapes'].append(item)

        dst_json_path = os.path.join(s_root, "jsons", image_name.split(".")[0] + ".json")
        with open(dst_json_path, 'w') as sf:
            json.dump(new_json, sf)

        dst_png_path = os.path.join(s_root, "images", image_name.split(".")[0] + ".jpg")
        cv2.imwrite(dst_png_path, img_res)

        # # debug
        for i, curve_line in enumerate(curve_lines):
            points, cls, cls_name = curve_line
            color = color_list[cls]
            # cls_name = cls_names[cls]
            # cls_name = "glass_edge"
            img_show = draw_curce_line_on_img(img_res, points, cls_name, color)
            print(img_show.shape)

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
            #     if img_name != "1689848678804013195.jpg":
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


if __name__ == "__main__":
    # 用来生成数据集和相应的infos
    generate_dataset_infos = {
        "type": 'image',
        # "dst_w": 1920,
        # "dst_h": 1080,
        "dst_w": -1,     # 不对图像进行resize等操作
        "dst_h": -1,
        "max_num": -1,  # default -1

        "metainfo": {
            'dataset': 'GbldMono2dDataset',
            'info_version': '1.0',
            'data': "2023-07-28",
            },

        "src_root": "/media/dell/Elements SE/liyj/data/collect_data_20230720/rosbag2_2023_07_20-18_24_24/glass_edge_overfit_20230721",
        "dst_root": "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230728_mmdet3d",
    }

    parse_cvat_xml_2_lld_data(generate_dataset_infos)

    # 生成mmdet3d数据加载的相关信息
    generate_mmdet3d_trainval_infos(generate_dataset_infos)