from argparse import ArgumentParser
import os
import cv2
import time
import shutil
import torch
import mmcv
import mmengine
import open3d as o3d
from tqdm import tqdm
from mmdet3d.apis import inference_multi_modality_detector, init_model
from mmdet3d.registry import VISUALIZERS
from copy import deepcopy
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmengine.dataset import Compose, pseudo_collate

# 在加载官方权重的时候,出现权重shape不匹配的情况
# size mismatch for pts_middle_encoder.conv_input.0.weight: copying a param with shape
def convert_checkpoints(path, s_path):
    # path = '../checkpoints/bevfuse/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'
    # s_path = "../checkpoints/bevfuse/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af_convert.pth"
    model = torch.load(path)

    for key in model['state_dict'].keys():
        if (key == 'pts_middle_encoder.conv_input.0.weight') or (key == 'pts_middle_encoder.conv_out.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer3.2.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer2.2.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer1.2.0.weight') or (('pts_middle_encoder.encoder_layers.encoder_layer' in key) and ('conv' in key)):
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 0, 1)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 1, 2)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 2, 3)
            model['state_dict'][key] = torch.transpose(model['state_dict'][key], 3, 4)

    torch.save(model, s_path)


def bevfuse_infer(cfg_path, checkpoint_path, data_root):
    # 保存结果的位置
    s_root = os.path.join("./", "bevfuse_result")
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)

    # 加载模型
    # bevfuse
    # cfg_path = "../projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
    # checkpoint_path = "../checkpoints/bevfuse/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af_convert.pth"

    device = 'cuda:0'
    model = init_model(cfg_path, checkpoint_path, device=device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    # 读取数据的位置
    # data_root = "../data/nuscenes"
    ann_file = os.path.join(data_root, "nuscenes_infos_val.pkl")
    data_list = mmengine.load(ann_file)['data_list']

    # 单个数据运行
    is_batch = False
    all_consume_time = []
    for index in tqdm(range(len(data_list)), desc="processing bevfuse..."):
    # for index in tqdm(range(10), desc="processing bevfuse..."):
        data = []

        # get data info containing calib
        data_info = data_list[index]
        pcd_name = data_info["lidar_points"]["lidar_path"]
        pcd = os.path.join(data_root, "samples", "LIDAR_TOP", pcd_name)

        for cam_type, img_info in data_info['images'].items():
            img_info['img_path'] = os.path.join(data_root, "samples", cam_type,  img_info['img_path'])
            assert os.path.isfile(img_info['img_path']), f'{img_info["img_path"]} does not exist.'

        data_ = dict(
            lidar_points=dict(lidar_path=pcd),
            images=data_info['images'],
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)

        data_['timestamp'] = data_info['timestamp']

        # 构建batch的形式
        data_ = test_pipeline(data_)
        data.append(data_)

        collate_data = pseudo_collate(data)

        # 运行结果
        torch.cuda.synchronize()
        start = time.time()
        # forward the model
        with torch.no_grad():
            results = model.test_step(collate_data)
        torch.cuda.synchronize()
        end = time.time()
        single_time = round((end - start) * 1000, 2)   # ms
        # print("forward time:", end - start)
        all_consume_time.append(single_time)

        if not is_batch:
            result, data = results[0], data[0]
        else:
            result, data = results, data

        # 可视化运行结果
        points = data['inputs']['points']
        if isinstance(result.img_path, list):
            img = []
            for img_path in result.img_path:
                single_img = mmcv.imread(img_path)
                single_img = mmcv.imconvert(single_img, 'bgr', 'rgb')
                img.append(single_img)
        else:
            img = mmcv.imread(result.img_path)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
        data_input = dict(points=points, img=img)

        s_img_name = pcd_name.split(".")[0] + ".jpg"
        s_img_path = os.path.join(s_root, s_img_name)

        # show the results
        visualizer.add_datasample(
            'result',
            data_input,
            data_sample=result,
            draw_gt=False,
            show=False,  #args.show
            wait_time=-1,
            out_file=s_img_path,  # args.out_dir
            pred_score_thr=0.2,  #args.score_thr
            vis_task='multi-modality_det')

    # 统计运行时间
    all_consume_time = all_consume_time[3:]    # 不统计前面的几次运行,一开始的运行时间会很慢,运行稳定后统计
    model_time = sum(all_consume_time)/len(all_consume_time)
    print("average single time:", model_time)

    with open(os.path.join(s_root, "time_forward.txt"), 'w') as fp:
        fp.write("average single model infer time: {} ms".format(model_time))

    # 保存为视频
    s_video(s_root)


def s_video(video_root):
    video_root = os.path.abspath(video_root)
    img_names = [name for name in os.listdir(video_root) if name.endswith(".jpg")]
    video_names = [name.split("-")[0] for name in os.listdir(video_root) if name.endswith(".jpg")]
    video_names = list(set(video_names))

    for video_name in tqdm(video_names):
        video_path = os.path.join(video_root, "bevfuse_" + video_name + '_output.avi')
        print(video_path)
        v_img_names = [name for name in img_names if name.startswith(video_name)]

        img = cv2.imread(os.path.join(video_root, v_img_names[0]))
        img_height, img_width, _ = img.shape

        v_img_names = sorted(v_img_names, key=lambda x: float(x.split(".")[0].split("__")[-1]))
        video_path = os.path.join(video_root, video_path)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 5, (img_width, img_height))

        for img_name in tqdm(v_img_names, desc="saving video:{}".format(video_name)):
            frame = cv2.imread(os.path.join(video_root, img_name))
            frame = cv2.resize(frame, (img_width, img_height))

            img_h, img_w, _ = frame.shape
            out.write(frame)
        out.release()


if __name__ == "__main__":
    print("Start")
    # 在加载官方权重的时候,出现权重shape不匹配的情况
    # size mismatch for pts_middle_encoder.conv_input.0.weight: copying a param with shape
    # 另外保存权重
    origin_checkpoint_path = '../checkpoints/bevfuse/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'
    convert_origin_checkpoint_path = "../checkpoints/bevfuse/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af_convert.pth"
    convert_checkpoints(origin_checkpoint_path, convert_origin_checkpoint_path)

    # 运行模型
    config_path = "../projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
    data_root = "../data/nuscenes"
    bevfuse_infer(config_path, convert_origin_checkpoint_path, data_root)
    print("End")
