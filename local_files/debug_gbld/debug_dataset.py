# 用于数据集调试

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import torch
import cv2
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend
from debug_utils import cal_points_orient, draw_orient, color_list
from gbld_mono2d_decode_numpy import GlasslandBoundaryLine2DDecodeNumpy
from debug_utils import sigmoid, sigmoid_inverse, draw_pred_result_numpy

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--config', help='train config file path')
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


def main(config_path):
    args = parse_args()

    args.config = config_path

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

    debug_dataset(cfg)


#----------------------------------------------调试模型构建----------------------------------------------
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Sequence, Union
import copy
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


def debug_dataset(cfg):
    default_scope = 'mmdet3d'
    _experiment_name = "123"

    from mmengine.registry import DefaultScope

    default_scope = DefaultScope.get_instance(  # type: ignore
        _experiment_name,
        scope_name=default_scope)

    from mmengine.registry import MODELS

    train_dataloader = cfg.get('train_dataloader')
    val_dataloader = cfg.get('val_dataloader')
    test_dataloader = cfg.get('test_dataloader')
    train_cfg = cfg.get('train_cfg')
    val_cfg = cfg.get('val_cfg')
    test_cfg = cfg.get('test_cfg')

    optim_wrapper = cfg.get('optim_wrapper')
    val_evaluator = cfg.get('val_evaluator')
    test_evaluator = cfg.get('test_evaluator')

    # train_dataloader主要是定义数据加载,包括train_pipeline等
    # train_cfg控制训练loop
    # optim_wrapper训练优化器材
    training_related = [train_dataloader, train_cfg, optim_wrapper]
    val_related = [val_dataloader, val_cfg, val_evaluator]
    test_related = [test_dataloader, test_cfg, test_evaluator]


    data_loader = build_dataloader(train_dataloader)
    dataset = data_loader.dataset

    # # 官方的EpochBasedTrainLoop定义
    # from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop

    # 这个data_preprocessor包含在模型的定义里面, 在进行forard前会调用这个模块
    model = cfg.get('model')
    data_preprocessor = model.data_preprocessor
    data_preprocessor = MODELS.build(data_preprocessor)

    grid_size = 4/model.bbox_head.up_scale
    glass_land_boundary_line_2d_decode_numpy = GlasslandBoundaryLine2DDecodeNumpy(confident_t=0.2, grid_size=grid_size)
    glass_land_boundary_line_2d_decode_numpy.debub_emb = 0
    glass_land_boundary_line_2d_decode_numpy.debug_existing_points = 0
    glass_land_boundary_line_2d_decode_numpy.debug_piece_line = 0
    glass_land_boundary_line_2d_decode_numpy.debug_exact_line = 0

    if 0:
        # 进行data_loader的debug,不好调试,需要加载的数据太多,debug的数据半天不出来
        for idx, data_batch in enumerate(data_loader):
            # ["img"]为list
            for img in data_batch["inputs"]["img"]:
                print("img：", img.shape)
            # "gt_instances".bboxes

            print(isinstance(data_batch, list))
            # 调用model的data_preprocessor进行处理后,构成batch的形式
            # 合并为["imgs"], Det3DDataPreprocessor在将所有图像进行堆叠的时候,会根据一个size的倍数进行paddng,同时会根据最大的img-size进行stack
            # 在右下角进行padding的同时也不会影响det的label坐标
            data_batch = data_preprocessor(data_batch)
            print(data_batch["inputs"]["imgs"].shape)

            # 对于每个case的label, label的数量都是不定的,所以采用list[]的形式输入程序
            print(len(data_batch["data_samples"]))
            # print(data_batch["data_samples"][0].gt_instances.bboxes.shape)
            # print(data_batch["data_samples"][1].gt_instances.bboxes.shape)

            # print(data_batch["data_samples"][1].gt_instances.gt_line_maps_stages.shape)

            # batch形式的label-gt不在dataloader里构建的,在head里构建
            print(len(data_batch["data_samples"][1].gt_instances.gt_line_maps_stages))
            print(data_batch["data_samples"][1].gt_instances.gt_line_maps_stages[0]["orient_map_mask"].shape)

            exit(1)

    print("metainfo:", dataset.metainfo)
    for idx, data in enumerate(dataset):
        print("one data", idx)
        print(data["inputs"]["img"].shape)
        # data: "data_samples", "inputs"
        # data_samples: gt_instances, get_instances_3d等等
        # inputs: img

        # 这是直接从dataset得到的, 是不带batch维度的
        # data["inputs"]["img"].shape, torch.Size([3, 900, 1600])

        # 如果是从dataloader的话
        # 那么是data["inputs"]["imgs"]

        # 测试数据集的transform
        img = data["inputs"]["img"]
        img = img.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        img_show = img.copy()


        gt_lines = data['data_samples'].gt_lines
        for line_id, gt_line in enumerate(gt_lines):
            gt_label = gt_line["label"]
            points = gt_line["points"]
            category_id = gt_line["category_id"]

            color = color_list[line_id]

            pre_point = points[0]
            for i, cur_point in enumerate(points[1:]):
                x1, y1 = int(pre_point[0]), int(pre_point[1])
                x2, y2 = int(cur_point[0]), int(cur_point[1])

                # thickness = 1
                thickness = 3
                # cv2.line是具有一定宽度的直线,如何获取密集的orients?应该也构建自身的mask?
                cv2.line(img_show, (x1, y1), (x2, y2), (255, 0, 0), thickness, 8)

                # 求每个点的方向
                if i % 100 == 0:
                    orient = cal_points_orient(pre_point, cur_point)

                    # 转个90度,指向草地
                    # orient = orient + 90
                    # if orient > 360:
                    #     orient = orient - 360

                    img_show = draw_orient(img_show, pre_point, orient, arrow_len=15, color=color)

                pre_point = cur_point
                # print(orient)
        # plt.imshow(img_show[:, :, ::-1])
        # plt.show()

        # debug_train_gt
        if 1:

            gt_line_maps_stages = data["data_samples"].gt_instances.gt_line_maps_stages
            # 可视化不同stage的结果
            for i, gt_line_maps in enumerate(gt_line_maps_stages):
                gt_confidence = gt_line_maps["gt_confidence"]
                gt_offset_x = gt_line_maps["gt_offset_x"]
                gt_offset_y = gt_line_maps["gt_offset_y"]
                gt_line_index = gt_line_maps["gt_line_index"]
                ignore_mask = gt_line_maps["ignore_mask"]
                foreground_mask = gt_line_maps["foreground_mask"]
                gt_line_id = gt_line_maps["gt_line_id"]
                gt_line_cls = gt_line_maps["gt_line_cls"]
                foreground_expand_mask = gt_line_maps["foreground_expand_mask"]
                orient_map_mask = gt_line_maps["orient_map_mask"]
                orient_map_sin = gt_line_maps["orient_map_sin"]
                orient_map_cos = gt_line_maps["orient_map_cos"]

                if "gt_confidence_visible" in gt_line_maps:
                    gt_confidence_visible = gt_line_maps["gt_confidence_visible"]
                else:
                    gt_confidence_visible = None

                if "gt_confidence_hanging" in gt_line_maps:
                    gt_confidence_hanging = gt_line_maps["gt_confidence_hanging"]
                else:
                    gt_confidence_hanging = None

                if "gt_confidence_covered" in gt_line_maps:
                    gt_confidence_covered = gt_line_maps["gt_confidence_covered"]
                else:
                    gt_confidence_covered = None

                if "point_map_mask" in gt_line_maps:
                    point_map_mask = gt_line_maps["point_map_mask"]
                else:
                    point_map_mask = None

                if "point_map_confidence" in gt_line_maps:
                    point_map_confidence = gt_line_maps["point_map_confidence"]
                else:
                    point_map_confidence = None

                if "point_map_index" in gt_line_maps:
                    point_map_index = gt_line_maps["point_map_index"]
                else:
                    point_map_index = None

                gt_confidence = gt_confidence.cpu().detach().numpy()
                gt_offset_x = gt_offset_x.cpu().detach().numpy()
                gt_offset_y = gt_offset_y.cpu().detach().numpy()
                gt_line_index = gt_line_index.cpu().detach().numpy()
                ignore_mask = ignore_mask.cpu().detach().numpy()
                foreground_mask = foreground_mask.cpu().detach().numpy()
                gt_line_id = gt_line_id.cpu().detach().numpy()
                gt_line_cls = gt_line_cls.cpu().detach().numpy()
                foreground_expand_mask = foreground_expand_mask.cpu().detach().numpy()

                orient_map_mask = orient_map_mask.cpu().detach().numpy()
                orient_map_sin = orient_map_sin.cpu().detach().numpy()
                orient_map_cos = orient_map_cos.cpu().detach().numpy()

                if gt_confidence_visible is not None:
                    gt_confidence_visible = gt_confidence_visible.cpu().detach().numpy()

                if gt_confidence_hanging is not None:
                    gt_confidence_hanging = gt_confidence_hanging.cpu().detach().numpy()

                if gt_confidence_covered is not None:
                    gt_confidence_covered = gt_confidence_covered.cpu().detach().numpy()

                if point_map_mask is not None:
                    point_map_mask = point_map_mask.cpu().detach().numpy()

                if point_map_confidence is not None:
                    point_map_confidence = point_map_confidence.cpu().detach().numpy()

                if point_map_index is not None:
                    point_map_index = point_map_index.cpu().detach().numpy()

                if 0:
                    # 可视化线段的端点预测
                    plt.subplot(3, 1, 1)
                    plt.title("point_map_mask")
                    plt.imshow(point_map_mask[0])

                    plt.subplot(3, 1, 2)
                    plt.title("point_map_confidence")
                    plt.imshow(point_map_confidence[0])

                    plt.subplot(3, 1, 3)
                    plt.title("point_map_index")
                    plt.imshow(point_map_index[0])
                    plt.show()

                # 将gt-map解析为预测结果
                if 1:
                    # if i != 2:
                    #     continue
                    img_show2 = img.copy()

                    heatmap_map_seg = gt_confidence
                    heatmap_map_offset = np.concatenate([gt_offset_x, gt_offset_y], axis=0)
                    heatmap_map_seg_emb = gt_line_index
                    heatmap_map_connect_emb = gt_line_id
                    heatmap_map_cls = gt_line_cls
                    heatmap_map_orient = np.concatenate([orient_map_sin, orient_map_cos], axis=0)
                    heatmap_map_visible = gt_confidence_visible
                    heatmap_map_hanging = gt_confidence_hanging
                    heatmap_map_covered = gt_confidence_covered

                    # 需要进行sigmoid操作的特征图
                    heatmap_map_seg = sigmoid_inverse(heatmap_map_seg)
                    heatmap_map_offset = sigmoid_inverse(heatmap_map_offset)
                    heatmap_map_cls = sigmoid_inverse(heatmap_map_cls)

                    # gt_confidence_visible，gt_confidence_hanging，gt_confidence_covered
                    if heatmap_map_visible is not None:
                        heatmap_map_visible = sigmoid_inverse(heatmap_map_visible)

                    if heatmap_map_hanging is not None:
                        heatmap_map_hanging = sigmoid_inverse(heatmap_map_hanging)

                    if heatmap_map_covered is not None:
                        heatmap_map_covered = sigmoid_inverse(heatmap_map_covered)

                    # 对于orient则需要sigmoid(orient_pred) * 2 - 1, 其逆操作如下
                    heatmap_map_orient = 1 + sigmoid_inverse(heatmap_map_orient)/2

                    glass_land_boundary_line_2d_decode_numpy.grid_size = model.bbox_head.strides[i]/model.bbox_head.up_scale
                    print(model.bbox_head.strides[i])
                    curse_lines_with_cls = glass_land_boundary_line_2d_decode_numpy.forward(heatmap_map_seg,
                                                                                            heatmap_map_offset,
                                                                                            heatmap_map_seg_emb,
                                                                                            heatmap_map_connect_emb,
                                                                                            heatmap_map_cls,
                                                                                            orient_pred=heatmap_map_orient,
                                                                                            visible_pred=heatmap_map_visible,
                                                                                            hanging_pred=heatmap_map_hanging,
                                                                                            covered_pred=heatmap_map_covered,)


                    # 对其进行可视化
                    img_show2 = draw_pred_result_numpy(img_show2, curse_lines_with_cls[0])

                # orient_mask = np.bitwise_and(orient_map_sin != 0, orient_map_cos != 0)
                # orient_mask = orient_mask[0]

                # plt.imshow(orient_mask)
                # plt.title("orient_mask")
                # plt.show()

                # 将orient转回到角度表示
                # orients = np.arctan2(orient_map_sin, orient_map_cos) / np.pi * 180
                # 转回到MEBOW的表示
                # if 180 >= orient >= 0:
                #     orient = 90 + (180 - orient)
                # elif orient >= -90:
                #     orient = 270 + abs(orient)
                # elif orient >= -180:
                #     orient = abs(orient) - 90
                # else:
                #     orient = None

                plt.imshow(img_show[:, :, ::-1])
                # plt.imshow(img_show2[:, :, ::-1])
                plt.show()
                #
                # plt.subplot(2, 2, 1)
                # plt.title("gt_confidence")
                # plt.imshow(gt_confidence[0])
                #
                # plt.subplot(2, 2, 2)
                # plt.title("gt_line_index")
                # plt.imshow(gt_line_index[0])
                #
                # plt.subplot(2, 2, 3)
                # plt.title("orient_map_mask")
                # plt.imshow(orient_map_mask[0])
                #
                # plt.subplot(2, 2, 4)
                # plt.title("img_show")
                # plt.imshow(img_show[:, :, ::-1])
                # plt.show()
                exit(1)

        exit(1)


if __name__ == '__main__':
    # config_path = "./projects/GrasslandBoundaryLine2D/configs/gbld_debug_config_no_dcn.py"
    # config_path = "./projects/GrasslandBoundaryLine2D/configs/gbld_debug_config_no_dcn_v0.2.py"
    # config_path = "./projects/GrasslandBoundaryLine2D/configs/gbld_debug_config_no_dcn_v0.2.py"
    config_path = "/home/liyongjing/Egolee/programs/mmdetection3d-liyj/projects/GrasslandBoundaryLine2D/configs/gbld_config_v0.4_overfit.py"

    # config_path = "./projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-seg.py"

    # 用于加载数据的测试调试,可用于验证数据加载和数据增强的正确性
    # debug的时候发现padding的像素在左边,这是由于padding后的flip造成的
    print("config_path:", config_path)
    main(config_path)