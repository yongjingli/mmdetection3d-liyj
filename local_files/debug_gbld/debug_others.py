import torch
import mmdet.models.necks.fpn as FPN
import mmengine
from mmengine.fileio import join_path, list_from_file, load
import numpy as np
import cv2
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    print("Start")
    # test_mmdet_fpn()
    # debug_write_data_infos()
    # test_dtype()
    # debug_sum()
    debug_padd()
    print("End")