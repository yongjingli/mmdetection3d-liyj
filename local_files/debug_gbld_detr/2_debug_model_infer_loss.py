# 用来调试生成模型

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import cv2
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
import torch

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
import numpy as np
from mmdet3d.utils import replace_ceph_backend


#----------------------------------------------调试模型构建----------------------------------------------
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Sequence, Union
import copy
import shutil
from functools import partial
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.dist import (broadcast, get_dist_info, get_rank, init_dist,
                           is_distributed, master_only)
from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import TORCH_VERSION

from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from mmengine.dataset import Compose, pseudo_collate
# from debug_utils import (decode_gt_lines, cal_points_orient, draw_orient,
#                          filter_near_same_points, parse_ann_info, parse_ann_infov2,
#                          discriminative_cluster_postprocess)

from vis_utils import color_list, draw_line, draw_bbox


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main(config_path, checkpoint_path):
    args = parse_args()

    args.config = config_path
    args.checkpoint = checkpoint_path

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    debug_model_infer(cfg, args)


def build_dataloader(dataloader: Union[DataLoader, Dict],
                     seed: Optional[int] = None,
                     diff_rank_seed: bool = False) -> DataLoader:
    """Build dataloader.

    The method builds three components:

    - Dataset
    - Sampler
    - Dataloader

    An example of ``dataloader``::

        dataloader = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=1,
            num_workers=9
        )

    Args:
        dataloader (DataLoader or dict): A Dataloader object or a dict to
            build Dataloader object. If ``dataloader`` is a Dataloader
            object, just returns itself.
        seed (int, optional): Random seed. Defaults to None.
        diff_rank_seed (bool): Whether or not set different seeds to
            different ranks. If True, the seed passed to sampler is set
            to None, in order to synchronize the seeds used in samplers
            across different ranks.


    Returns:
        Dataloader: DataLoader build from ``dataloader_cfg``.
    """
    if isinstance(dataloader, DataLoader):
        return dataloader

    dataloader_cfg = copy.deepcopy(dataloader)

    # build dataset
    dataset_cfg = dataloader_cfg.pop('dataset')
    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
    else:
        # fallback to raise error in dataloader
        # if `dataset_cfg` is not a valid type
        dataset = dataset_cfg

    # build sampler
    sampler_cfg = dataloader_cfg.pop('sampler')
    if isinstance(sampler_cfg, dict):
        sampler_seed = None if diff_rank_seed else seed
        sampler = DATA_SAMPLERS.build(
            sampler_cfg,
            default_args=dict(dataset=dataset, seed=sampler_seed))
    else:
        # fallback to raise error in dataloader
        # if `sampler_cfg` is not a valid type
        sampler = sampler_cfg

    # build batch sampler
    batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
    if batch_sampler_cfg is None:
        batch_sampler = None
    elif isinstance(batch_sampler_cfg, dict):
        batch_sampler = DATA_SAMPLERS.build(
            batch_sampler_cfg,
            default_args=dict(
                sampler=sampler,
                batch_size=dataloader_cfg.pop('batch_size')))
    else:
        # fallback to raise error in dataloader
        # if `batch_sampler_cfg` is not a valid type
        batch_sampler = batch_sampler_cfg

    # build dataloader
    init_fn: Optional[partial]

    if 'worker_init_fn' in dataloader_cfg:
        worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
        worker_init_fn_type = worker_init_fn_cfg.pop('type')
        if isinstance(worker_init_fn_type, str):
            worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
        elif callable(worker_init_fn_type):
            worker_init_fn = worker_init_fn_type
        else:
            raise TypeError(
                'type of worker_init_fn should be string or callable '
                f'object, but got {type(worker_init_fn_type)}')
        assert callable(worker_init_fn)
        init_fn = partial(worker_init_fn,
                          **worker_init_fn_cfg)  # type: ignore
    else:
        if seed is not None:
            disable_subprocess_warning = dataloader_cfg.pop(
                'disable_subprocess_warning', False)
            assert isinstance(disable_subprocess_warning, bool), (
                'disable_subprocess_warning should be a bool, but got '
                f'{type(disable_subprocess_warning)}')
            init_fn = partial(
                default_worker_init_fn,
                num_workers=dataloader_cfg.get('num_workers'),
                rank=get_rank(),
                seed=seed,
                disable_subprocess_warning=disable_subprocess_warning)
        else:
            init_fn = None

    # `persistent_workers` requires pytorch version >= 1.7
    if ('persistent_workers' in dataloader_cfg
            and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
        print_log(
            '`persistent_workers` is only available when '
            'pytorch version >= 1.7',
            logger='current',
            level=logging.WARNING)
        dataloader_cfg.pop('persistent_workers')

    # The default behavior of `collat_fn` in dataloader is to
    # merge a list of samples to form a mini-batch of Tensor(s).
    # However, in mmengine, if `collate_fn` is not defined in
    # dataloader_cfg, `pseudo_collate` will only convert the list of
    # samples into a dict without stacking the batch tensor.
    collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                        dict(type='pseudo_collate'))
    if isinstance(collate_fn_cfg, dict):
        collate_fn_type = collate_fn_cfg.pop('type')
        if isinstance(collate_fn_type, str):
            collate_fn = FUNCTIONS.get(collate_fn_type)
        else:
            collate_fn = collate_fn_type
        collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    elif callable(collate_fn_cfg):
        collate_fn = collate_fn_cfg
    else:
        raise TypeError(
            'collate_fn should be a dict or callable object, but got '
            f'{collate_fn_cfg}')
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler if batch_sampler is None else None,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
        **dataloader_cfg)
    return data_loader


from mmdet3d.apis import init_model
def debug_model_infer(cfg, args):
    default_scope = 'mmdet3d'
    _experiment_name = "123"

    from mmengine.registry import DefaultScope
    default_scope = DefaultScope.get_instance(  # type: ignore
        _experiment_name,
        scope_name=default_scope)

    from mmengine.registry import MODELS
    # model = cfg['model']
    # model = MODELS.build(model)

    device = 'cuda:0'

    model = cfg.get('model')
    data_preprocessor = model.data_preprocessor
    data_preprocessor = MODELS.build(data_preprocessor)

    model = init_model(cfg, args.checkpoint, device=device)

    train_dataloader = cfg.get('train_dataloader')
    # train_dataloader = cfg.get('test_dataloader')
    data_loader = build_dataloader(train_dataloader)

    for idx, data_batch in enumerate(data_loader):
        # 调用model的data_preprocessor进行处理后
        # 合并为["imgs"], Det3DDataPreprocessor在将所有图像进行堆叠的时候,会根据一个size的倍数进行paddng,同时会根据最大的img-size进行stack
        # 在右下角进行padding的同时也不会影响det的label坐标
        data_batch = data_preprocessor(data_batch)

        data_batch["inputs"]["imgs"] = data_batch["inputs"]["imgs"].to(device)

        # print(data_batch["inputs"]["imgs"].shape)

        # ["img"]为list
        # for img in data_batch["inputs"]["img"]:
        #     print("img：", img.shape)
        # "gt_instances".bboxes

        # out = debug_model(batch_inputs_dict, data_samples=batch_img_metas, mode="loss")

        # loss
        # loss = model(data_batch["inputs"], data_batch["data_samples"], mode="loss")


        # predict
        results = model(data_batch["inputs"], data_batch["data_samples"], mode="predict")

        for batch_id, result in enumerate(results):

            img = data_batch["inputs"]["imgs"][batch_id]
            img = img.cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            mean = np.array([103.53, 116.28, 123.675, ])
            std = np.array([1.0, 1.0, 1.0, ])

            img = img * std + mean
            img = img.astype(np.uint8).copy()

            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.detach().cpu().numpy()
            labels = pred_instances.labels.detach().cpu().numpy()
            scores = pred_instances.scores.detach().cpu().numpy()
            pts = pred_instances.pts.detach().cpu().numpy()

            seg = pred_instances.seg[0].detach().cpu().numpy()[0]
            plt.imshow(seg)
            plt.show()
            # print(labels)

            for bbox, label, score, pt in zip(bboxes, labels, scores, pts):
                if score > 0.4:
                    # print(label)
                    x1, y1, x2, y2 = [int(d) for d in bbox[:4]]
                    color = color_list[int(label)]
                    print(label)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

                    img = draw_line(img, pt, color_list[int(label)])


                # cv2.rectangle(img, (y1, x1), (y2, x2), (255, 0, 0), 1)

            # meta_info = result.metainfo
            # batch_input_shape = meta_info["batch_input_shape"]
            #
            # gt_line_map = np.zeros(batch_input_shape, dtype=np.uint8)
            #
            # gt_lines = meta_info["gt_lines"]
            # # gt_lines = meta_info["eval_gt_lines"]
            # for gt_line in gt_lines:
            #     gt_label = gt_line["label"]
            #     points = gt_line["points"]
            #     category_id = gt_line["category_id"]
            #
            #     pre_point = points[0]
            #     for cur_point in points[1:]:
            #         x1, y1 = int(pre_point[0]), int(pre_point[1])
            #         x2, y2 = int(cur_point[0]), int(cur_point[1])
            #
            #         thickness = 3
            #         cv2.line(gt_line_map, (x1, y1), (x2, y2), (1), thickness, 8)
            #         pre_point = cur_point
            #
            # pred_line_map = np.zeros(batch_input_shape, dtype=np.uint8)
            # single_stage_result = stages_result[0]
            # for curve_line in single_stage_result:
            #     curve_line = np.array(curve_line)
            #     pre_point = curve_line[0]
            #     for cur_point in curve_line[1:]:
            #         x1, y1 = int(pre_point[0]), int(pre_point[1])
            #         x2, y2 = int(cur_point[0]), int(cur_point[1])
            #
            #         thickness = 3
            #         cv2.line(pred_line_map, (x1, y1), (x2, y2), (1), thickness, 8)
            #         pre_point = cur_point
            #
            # plt.subplot(3, 1, 1)
            # plt.imshow(gt_line_map)
            # plt.subplot(3, 1, 2)
            # plt.imshow(pred_line_map)
            #
            # plt.subplot(3, 1, 3)
            # plt.imshow(img[:, :, ::-1])
            plt.imshow(img[:, :, ::-1])
            plt.show()
            exit(1)


def debug_model_infer_heatmap(config_path, checkpoint_path):
    # test_root = "/data-hdd/liyj/data/label_data/20230926/dataset_28/images"
    # test_root = "/media/dell/Egolee1/liyj/label_data/20230926/dataset_28/images"
    # test_root = "/media/dell/Egolee1/liyj/data/ros_bags/20231012/rosbag2_2023_10_11-10_28_36/parse_data/image"
    # test_root = "/media/dell/Egolee1/liyj/data/ros_bags/20231012/rosbag2_2023_10_11-10_28_36/parse_data/image"
    test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231013_mmdet3d/train/images"
    print(test_root)

    model = init_model(config_path, checkpoint_path, device='cuda:0')
    cfg = model.cfg

    test_pipeline = deepcopy(cfg.val_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)

    save_root = test_root + "_debug"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)
    img_names = [name for name in os.listdir(test_root) if name[-4:] in ['.jpg', '.png']]
    jump = 5
    count = 10

    for img_name in tqdm(img_names):
        # img_name = "1695031543693612696.jpg"
        # img_name = "258733034.13391113.jpg"
        img_name = "1695031238613012474.jpg"
        count = count + 1
        if count % jump != 0:
            continue

        # img_name = "1695031543693612696.jpg"
        path_0 = os.path.join(test_root, img_name)
        # path_0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.jpg"

        img = cv2.imread(path_0)

        img_paths = [path_0]
        batch_data = []
        for img_path in img_paths:    # 这里可以是列表,然后推理采用batch的形式
            data_info = {}
            data_info['img_path'] = img_path
            data_input = test_pipeline(data_info)
            batch_data.append(data_input)

        # 这里设置为单个数据, 可以是batch的形式
        collate_data = pseudo_collate(batch_data)

        # forward the model
        with torch.no_grad():
            # results = model.test_step(collate_data)
            collate_data = model.data_preprocessor(collate_data, False)

            # 与gt进行计算losss
            # results = model.loss(batch_inputs=collate_data["inputs"],
            #                      batch_data_samples=collate_data["data_samples"])


            # 得到预测解析后的结果
            # results = model.predict(batch_inputs=collate_data["inputs"],
            #                      batch_data_samples=collate_data["data_samples"])

            # 得到head的heatmap的结果,用于分析
            # dir_seg_pred, dir_offset_pred, dir_seg_emb_pred, dir_connect_emb_pred, dir_cls_pred, dir_orient_pred
            results = model._forward(batch_inputs=collate_data["inputs"],
                                 batch_data_samples=collate_data["data_samples"])

            # 为多分辨率多尺度的预测结果,取最大的分辨率
            # dir_seg_pred, dir_offset_pred, dir_seg_emb_pred,
            bath_size = 0
            stages = 0
            heatmap_map_seg = torch.sigmoid(results[0][stages].cpu().detach()).numpy()[bath_size][0] > 0.2
            heatmap_map_emb = results[2][stages].cpu().detach().numpy()[bath_size][0]

            plt.subplot(2, 2, 1)
            plt.imshow(img[:, :, ::-1])
            plt.subplot(2, 2, 2)
            plt.imshow(heatmap_map_seg)
            plt.subplot(2, 2, 3)
            plt.imshow(heatmap_map_emb)
            plt.show()

            exit(1)


def debug_model_infer_loss(config_path, checkpoint_path):
    # test_root = "/data-hdd/liyj/data/label_data/20230926/dataset_28/images"
    # test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230927_mmdet3d/train"
    # test_root = "/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231013_mmdet3d/train"
    # test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231108_mmdet3d_spline_by_cls/train"
    test_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231116_mmdet3d_spline_by_cls_overfit/train"

    image_root = os.path.join(test_root, "images")
    json_root = os.path.join(test_root, "jsons")


    model = init_model(config_path, checkpoint_path, device='cuda:0')
    cfg = model.cfg

    # test_pipeline = deepcopy(cfg.val_dataloader.dataset.pipeline)
    # test_pipeline = Compose(test_pipeline)

    train_pipeline = deepcopy(cfg.train_dataloader.dataset.pipeline)
    train_pipeline_no_aug = []
    for _pipeline in train_pipeline:
        print(_pipeline["type"])
        if _pipeline["type"] in ["mmdet.LoadImageFromFile", "GgldResize", "GgldLoadLines", "Pad",
                                 "GgldLineMapsGenerate",  "GgldLineMapsGenerateV2", "PackGbldMono2dInputs"]:
            train_pipeline_no_aug.append(_pipeline)

    # train_pipeline = Compose(train_pipeline)
    train_pipeline = Compose(train_pipeline_no_aug)
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)

    save_root = image_root + "_loss"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)
    img_names = [name for name in os.listdir(image_root) if name[-4:] in ['.jpg', '.png']]
    jump = 1
    # count = 10
    count = len(img_names)

    for img_name in tqdm(img_names):
        # img_name = "1695031543693612696.jpg"
        # img_name = "dff47067-ede3-4ac0-b903-ef04df89a291_front_camera_3531.jpg"
        # img_name = "1696991348.057879.jpg"
        # img_name = "1696991348.45783.jpg"
        # img_name = "1696991118.675607.jpg"
        print(img_name)
        count = count + 1
        if count % jump != 0:
            continue

        # img_name = "1695031543693612696.jpg"
        img_path_0 = os.path.join(image_root, img_name)
        json_path_0 = os.path.join(json_root, img_name[:-4] + ".json")
        # path_0 = "/home/dell/liyongjing/programs/mmdetection3d-liyj/local_files/debug_gbld/data/1689848680876640844.jpg"

        img_paths = [img_path_0]
        json_paths = [json_path_0]
        batch_data = []
        for idx, (img_path, json_path) in enumerate(zip(img_paths, json_paths)):    # 这里可以是列表,然后推理采用batch的形式
            data_info = {}
            data_info["img_name"] = img_name
            data_info["ann_name"] = img_name[:-4] + ".json"
            data_info['img_path'] = img_path
            data_info['ann_path'] = json_path
            data_info['sample_idx'] = idx

            # data_info['ann_info'] = parse_ann_info(data_info)
            data_info['ann_info'] = parse_ann_infov2(data_info)

            data_input = train_pipeline(data_info)
            # data_input = test_pipeline(data_info)
            batch_data.append(data_input)

        # 这里设置为单个数据, 可以是batch的形式
        collate_data = pseudo_collate(batch_data)


        # forward the model
        with torch.no_grad():
            # results = model.test_step(collate_data)
            collate_data = model.data_preprocessor(collate_data, False)

            # 与gt进行计算losss
            loss = model.loss(batch_inputs=collate_data["inputs"],
                                 batch_data_samples=collate_data["data_samples"])


            # 得到head的heatmap的结果,用于分析
            # dir_seg_pred, dir_offset_pred, dir_seg_emb_pred, dir_connect_emb_pred, dir_cls_pred, dir_orient_pred
            heatmap = model._forward(batch_inputs=collate_data["inputs"],
                                 batch_data_samples=collate_data["data_samples"])

            # 得到预测解析后的结果, rescale=False为输入大小的图像上
            results = model.predict(batch_inputs=collate_data["inputs"],
                                 batch_data_samples=collate_data["data_samples"], rescale=False)
            results = results[0]

            # 为多分辨率多尺度的预测结果,取最大的分辨率
            # dir_seg_pred, dir_offset_pred, dir_seg_emb_pred, dir_connect_emb_pred, \
            # dir_cls_pred, dir_orient_pred, dir_visible_pred, dir_hanging_pred, dir_covered_pred
            bath_size = 0
            stages = 0

            heatmap_map_seg = torch.sigmoid(heatmap[0][stages].cpu().detach()).numpy()[bath_size][0]
            heatmap_map_emb = heatmap[2][stages].cpu().detach().numpy()[bath_size]
            heatmap_map_cls = torch.sigmoid(heatmap[4][stages].cpu().detach()).numpy()[bath_size]
            heatmap_map_cls = np.argmax(heatmap_map_cls, axis=0)
            # heatmap_map_cls = torch.sigmoid(heatmap_map_cls[3]) > 0.2  # 选择不同类别的heatmap
            heatmap_map_covered = torch.sigmoid(heatmap[8][stages].cpu().detach()).numpy()[bath_size][0]

            # 调试discrimate聚类loss和效果
            if 0:
                heatmap_map_discrimate = heatmap[9][stages].cpu().detach().numpy()[bath_size]
                discriminative_cluster_postprocess(heatmap_map_seg > 0.2, heatmap_map_discrimate)
                # discriminative_cluster_postprocess(heatmap_map_seg > 0.2, heatmap_map_emb)

            # show pred result
            img = collate_data["inputs"]["imgs"][bath_size]
            img = img.cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            mean = np.array([103.53, 116.28, 123.675, ])
            std = np.array([1.0, 1.0, 1.0, ])

            img = img * std + mean
            img = img.astype(np.uint8)

            img_show = copy.deepcopy(img.copy())

            stages_result = results.pred_instances.stages_result[0]
            meta_info = results.metainfo
            batch_input_shape = meta_info["batch_input_shape"]

            gt_line_map = np.zeros(batch_input_shape, dtype=np.uint8)

            gt_lines = meta_info["gt_lines"]
            # gt_lines = meta_info["eval_gt_lines"]
            for gt_line in gt_lines:
                gt_label = gt_line["label"]
                points = gt_line["points"]
                category_id = gt_line["category_id"]

                pre_point = points[0]
                for cur_point in points[1:]:
                    x1, y1 = int(pre_point[0]), int(pre_point[1])
                    x2, y2 = int(cur_point[0]), int(cur_point[1])

                    thickness = 3
                    cv2.line(gt_line_map, (x1, y1), (x2, y2), (1), thickness, 8)
                    pre_point = cur_point

            pred_line_map = np.zeros(batch_input_shape, dtype=np.uint8)
            single_stage_result = stages_result[0]
            for curve_line in single_stage_result:
                curve_line = np.array(curve_line)
                pre_point = curve_line[0]

                line_cls = pre_point[4]
                color = color_list[int(line_cls)]
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                color = [0, 0, 255]

                point_num = len(curve_line)

                for i, cur_point in enumerate(curve_line[1:]):
                    x1, y1 = int(pre_point[0]), int(pre_point[1])
                    x2, y2 = int(cur_point[0]), int(cur_point[1])

                    thickness = 3
                    cv2.line(pred_line_map, (x1, y1), (x2, y2), (1), thickness, 8)

                    cv2.line(img_show, (x1, y1), (x2, y2), color, thickness, 8)
                    line_orient = cal_points_orient(pre_point, cur_point)

                    if i % 5 == 0:
                        orient = pre_point[5]
                        if orient != -1:
                            reverse = False  # 代表反向是否反了
                            orient_diff = abs(line_orient - orient)
                            if orient_diff > 180:
                                orient_diff = 360 - orient_diff

                            if orient_diff > 90:
                                reverse = True

                            # color = (0, 255, 0)
                            # if reverse:
                            #     color = (0, 0, 255)

                            # 绘制预测的方向
                            # 转个90度,指向草地
                            orient = orient + 90
                            if orient > 360:
                                orient = orient - 360

                            # img_origin = draw_orient(img_origin, pre_point, orient, arrow_len=30, color=color)

                    if i == point_num // 2:
                        line_orient = line_orient + 90
                        if line_orient > 360:
                            line_orient = line_orient - 360
                        img_show = draw_orient(img_show, pre_point, line_orient, arrow_len=50, color=color)


                    pre_point = cur_point

            # show heatmap
            # plt.imshow(heatmap_map_seg)
            # plt.imshow(heatmap_map_emb)
            # plt.show()
            all_loss = 0
            for _loss in loss.keys():
                print(_loss, loss[_loss].detach().cpu().numpy())
                all_loss = all_loss + loss[_loss].detach().cpu().numpy()

            # show result
            plt.subplot(2, 2, 1)
            plt.imshow(img_show[:, :, ::-1])

            plt.subplot(2, 2, 2)
            plt.imshow(gt_line_map)

            plt.subplot(2, 2, 3)
            # plt.imshow(pred_line_map)
            plt.imshow(heatmap_map_seg)

            plt.subplot(2, 2, 4)
            # plt.imshow(heatmap_map_emb)
            # plt.imshow(heatmap_map_cls)
            # plt.imshow(heatmap_map_covered)
            plt.imshow(heatmap_map_emb[0])

            s_name = "loss_" + str(round(all_loss, 2)) + "_" + img_name
            s_path = os.path.join(save_root, s_name)
            plt.savefig(s_path, dpi=300)

            plt.show()
            exit(1)


if __name__ == '__main__':
    # 作为辅助任务
    config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D_DETR/work_dirs/gbld_overfit_detr_20231227_0/gbld_deter_config_overfit.py"
    checkpoint_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D_DETR/work_dirs/gbld_overfit_detr_20231227_0/epoch_10000.pth"

    print("config_path:", config_path)
    # 对比gt和pred的区别
    main(config_path, checkpoint_path)

    # 用来可视化预测的结果, heatmap, 输入形式为图片
    # debug_model_infer_heatmap(config_path, checkpoint_path)

    # 用来查看sample的loss情况, 输入形式包括图片和json
    # debug_model_infer_loss(config_path, checkpoint_path)
    print("end")