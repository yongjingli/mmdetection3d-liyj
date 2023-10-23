# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
from mmengine.dataset import Compose, pseudo_collate
from copy import deepcopy
from mmdet3d.apis import inference_mono_3d_detector, init_model
from mmdet3d.registry import VISUALIZERS
import numpy as np
import mmdet
import cv2
import os
import torch
import shutil
from types import MethodType
from tqdm import tqdm
import matplotlib.pyplot as plt
from mmengine.dataset import Compose, pseudo_collate


color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238), (130, 0, 75),
              (169, 169, 169), (0, 69, 255)]  # [纯红、纯绿、纯蓝、金色、纯黄、天蓝、紫色、粉色、紫罗兰、藏青色、深灰色、橙红色]


def forward_dummy(self, batch_inputs):
    x = self.extract_feat(batch_inputs)
    results = self.bbox_head.forward(x)
    return results

#
# def forward(self,
#             inputs: torch.Tensor,
#             data_samples: OptSampleList = None,
#             mode: str = 'tensor') -> ForwardResults:
#     if mode == 'loss':
#         return self.loss(inputs, data_samples)
#     elif mode == 'predict':
#         return self.predict(inputs, data_samples)
#     elif mode == 'tensor':
#         return self._forward(inputs, data_samples)
#     else:
#         raise RuntimeError(f'Invalid mode "{mode}". '
#                            'Only supports loss, predict and tensor mode')


def onnx_export(self, batch_inputs):
    x = self.backbone(batch_inputs)
    if self.with_neck:
        x = self.neck(x)
    results = self.bbox_head.forward(x)

    #dir_seg_pred, dir_offset_pred, dir_seg_emb_pred, dir_connect_emb_pred, dir_cls_pred
    concat_out = torch.concat(results, dim=1)
    return concat_out


def main():
    # save_root = "/home/dell/liyongjing/test_data/debug"
    save_root = "/home/dell/下载/debug"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)

    # build the model from a config file and a checkpoint file
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/configs/gbld_debug_config.py"
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug/20230804_164059/vis_data/config.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug/20230804_164059/epoch_250.pth"

    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug_no_dcn/20230812_182852/vis_data/config.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug_no_dcn/epoch_250.pth"

    # gbld_20230907.onnx
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230907_v0.2_fit_line_crop/gbld_debug_config_no_dcn_v0.2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230907_v0.2_fit_line_crop/epoch_250.pth"

    # gbld_20230927.onnx
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230927_v0.2_fit_line_crop/gbld_debug_config_no_dcn_v0.2.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/gbld_overfit_20230927_v0.2_fit_line_crop/epoch_250.pth"

    # debug visible hanging covered
    config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered/gbld_debug_config_no_dcn_datasetv2.py"
    checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/work_dirs/debug_visible_hanging_covered/epoch_100.pth"

    # debug
    device = 'cuda:0'
    model = init_model(config_path, checkpoint_path, device=device)

    # origin_forward = model.forward
    model.forward = MethodType(onnx_export, model)
    # model.forward = MethodType(forward_dummy, model)
    # model.forward = model.forward_dummy
    # (960, 608)  # (img_w, img_h)
    img_h = 608
    img_w = 960
    x = torch.randn(1, 3, img_h, img_w, requires_grad=True)
    x = x.to(device)

    input = x.detach().cpu().numpy()
    save_path = save_root + "/" + "gbld_pt_input_" + str(0) + ".npy"
    np.save(save_path, input)
    save_path = save_root + "/" + "gbld_pt_input_" + str(0) + ".bin"
    input.tofile(save_path)

    input_len = 1
    for s in input.shape:
        input_len = input_len * s
    print("input_len:", input_len)

    model.eval()
    # with torch.no_grad():
    #     outputs = model(x)
    #     output = outputs.detach().cpu().numpy()
    #     save_path = save_root + "/" + "gpld_pt_output_" + str(0) + ".npy"
    #     np.save(save_path, output)
    #
    #     input_len = 1
    #     for s in output.shape:
    #         input_len = input_len * s
    #     print("output_len {}:{}".format(0, input_len))

    out_path = save_root + "/" + "gbld_20230927.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=x,
            f=out_path,
            input_names=['input'],
            output_names=['output'],
            # operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
            # opset_version=11,
            # enable_onnx_checker=False,
        )

    print("export_onnx_centernet Done")



    #
    # # 采用官方的方式进行推理
    # cfg = model.cfg
    # # build the data pipeline
    # test_pipeline = deepcopy(cfg.val_dataloader.dataset.pipeline)
    # test_pipeline = Compose(test_pipeline)
    #
    # test_root = "/home/dell/liyongjing/test_data/rosbag2_2023_07_20-18_24_24_0_image"
    # # test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230728_mmdet3d/train/images"
    # img_names = [name for name in os.listdir(test_root) if name[-4:] in ['.jpg', '.png']]
    # for img_name in tqdm(img_names):
    #     # img_name = "1689848710237850651.jpg"
    #     path_0 = os.path.join(test_root, img_name)
    #     path_0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.jpg"
    #     # path_0 = "/media/dell/Elements SE/liyj/data/collect_data_20230720/rosbag2_2023_07_20-18_24_24/glass_edge_overfit_20230721/1689848678804013195.jpg"
    #     # path_0 = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230728_mmdet3d/test/images/1689848678804013195.jpg"
    #     # path_0 = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230728_mmdet3d/train/images/1689848680876640844.jpg"
    #
    #     img_paths = [path_0]
    #     batch_data = []
    #     for img_path in img_paths:    # 这里可以是列表,然后推理采用batch的形式
    #         data_info = {}
    #         data_info['img_path'] = img_path
    #         data_input = test_pipeline(data_info)
    #         batch_data.append(data_input)
    #
    #     # 这里设置为单个数据, 可以是batch的形式
    #     collate_data = pseudo_collate(batch_data)
    #
    #     # forward the model
    #     with torch.no_grad():
    #         results = model.test_step(collate_data)
    #         # data_batch = data_preprocessor(data_batch)
    #         # results = model(collate_data["inputs"], collate_data["data_samples"], mode="predict")
    #
    #         for batch_id, result in enumerate(results):
    #             img_origin = cv2.imread(img_paths[batch_id])
    #
    #             img = collate_data["inputs"]["img"][batch_id]
    #             img = img.cpu().detach().numpy()
    #             img = np.transpose(img, (1, 2, 0))
    #             mean = np.array([103.53, 116.28, 123.675, ])
    #             std = np.array([1.0, 1.0, 1.0, ])
    #
    #             # 模型的内部进行归一化的处理data_preprocessor,从dataset得到的数据实际是未处理前的
    #             # 如果是从result里拿到的img,则需要进行这样的还原
    #             # img = img * std + mean
    #             img = img.astype(np.uint8)
    #
    #             stages_result = result.pred_instances.stages_result[0]
    #             # meta_info = result.metainfo
    #             # batch_input_shape = meta_info["batch_input_shape"]
    #             # pred_line_map = np.zeros(batch_input_shape, dtype=np.uint8)
    #
    #             single_stage_result = stages_result[0]
    #             for curve_line in single_stage_result:
    #                 curve_line = np.array(curve_line)
    #                 pre_point = curve_line[0]
    #
    #                 line_cls = pre_point[4]
    #                 color = color_list[int(line_cls)]
    #                 for cur_point in curve_line[1:]:
    #                     x1, y1 = int(pre_point[0]), int(pre_point[1])
    #                     x2, y2 = int(cur_point[0]), int(cur_point[1])
    #
    #                     thickness = 3
    #                     cv2.line(img_origin, (x1, y1), (x2, y2), color, thickness, 8)
    #                     pre_point = cur_point
    #
    #             s_img_path = os.path.join(save_root, img_name)
    #             cv2.imwrite(s_img_path, img_origin)
    #             # plt.imshow(img_origin[:, :, ::-1])
    #             # plt.show()
    #             exit(1)

        # print("ff")


if __name__ == "__main__":
    # 进行模型预测, 给出图片的路径, 即可进行模型预测
    print("Start")
    main()
    print("End")
