# 用来调试模型的测评阶段

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import cv2
import os.path as osp
import matplotlib.pyplot as plt
import torch
import shutil
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
import numpy as np
from mmdet3d.utils import replace_ceph_backend
from mmengine.evaluator import Evaluator


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


def build_evaluator(evaluator: Union[Dict, List, Evaluator]) -> Evaluator:
    if isinstance(evaluator, Evaluator):
        return evaluator
    elif isinstance(evaluator, dict):
        # if `metrics` in dict keys, it means to build customized evalutor
        if 'metrics' in evaluator:
            evaluator.setdefault('type', 'Evaluator')
            return EVALUATOR.build(evaluator)
        # otherwise, default evalutor will be built
        else:
            return Evaluator(evaluator)  # type: ignore
    elif isinstance(evaluator, list):
        # use the default `Evaluator`
        return Evaluator(evaluator)  # type: ignore
    else:
        raise TypeError(
            'evaluator should be one of dict, list of dict, and Evaluator'
            f', but got {evaluator}')


from mmdet3d.apis import init_model
def debug_model_infer(cfg, args):
    save_root = "/home/dell/liyongjing/test_data/debug"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)


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
    # model = init_model(args.config, args.checkpoint, device=device)

    train_dataloader = cfg.get('train_dataloader')
    train_data_loader = build_dataloader(train_dataloader)

    val_dataloader = cfg.get('val_dataloader')
    val_data_loader = build_dataloader(val_dataloader)

    val_evaluator = cfg.get('val_evaluator')
    val_evaluator = build_evaluator(val_evaluator)

    # for idx, data_batch in enumerate(train_data_loader):
    for idx, data_batch in enumerate(val_data_loader):
        # 调用model的data_preprocessor进行处理后
        # 合并为["imgs"], Det3DDataPreprocessor在将所有图像进行堆叠的时候,会根据一个size的倍数进行paddng,同时会根据最大的img-size进行stack
        # 在右下角进行padding的同时也不会影响det的label坐标

        data_batch = data_preprocessor(data_batch)
        data_batch["inputs"]["imgs"] = data_batch["inputs"]["imgs"].to(device)

        # loss
        # loss = model(data_batch["inputs"], data_batch["data_samples"], mode="loss")

        # predict
        results = model(data_batch["inputs"], data_batch["data_samples"], mode="predict")
        # print(results[0].pred_instances.stages_result[0])
        # print(results[0].pred_instances.stages_result[0][0])
        # for line in results[0].pred_instances.stages_result[0][0]:
        #     for point in line:
        #         print(point)
        #
        # exit(1)

        # print(results[0]["pred_instances"]["stages_result"][0])
        # exit(1)

        # with autocast(enabled=self.fp16):
        #     outputs = self.runner.model.val_step(data_batch)

        # 采用evaluator进行测评
        val_evaluator.process(data_samples=results, data_batch=data_batch)

        if idx == 2:
            break

        if 1:
            # 根据预测的结果进行可视化
            for batch_id, result in enumerate(results):
                img = data_batch["inputs"]["imgs"][batch_id]

                img = img.cpu().detach().numpy()
                img = np.transpose(img, (1, 2, 0))
                mean = np.array([103.53, 116.28, 123.675, ])
                std = np.array([1.0, 1.0, 1.0, ])

                img = img * std + mean
                img = img.astype(np.uint8)

                stages_result = result.pred_instances.stages_result[0]

                meta_info = result.metainfo

                batch_input_shape = meta_info["batch_input_shape"]
                ori_shape = meta_info["ori_shape"]

                img_show = copy.deepcopy(img)
                # 这里的img包含padding的结果,直接resize是有问题的,可以考虑padding去除后再进行resize
                # 这里的可视化仅仅作为参考,绘制的结果不能完全匹配上
                img_show = cv2.resize(img_show, (ori_shape[1], ori_shape[0]))

                # gt_line_map = np.zeros(batch_input_shape, dtype=np.uint8)
                gt_line_map = np.zeros(ori_shape, dtype=np.uint8)
                # print(meta_info.keys())

                # eval_gt_lines = meta_info["eval_gt_lines"]
                eval_gt_lines = meta_info["ori_eval_gt_lines"]
                # eval_gt_lines = meta_info["gt_lines"]    # 训练集中为gt_lines
                for gt_line in eval_gt_lines:
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

                print("ori_shape", ori_shape)
                pred_line_map = np.zeros(ori_shape, dtype=np.uint8)
                single_stage_result = stages_result[0]
                for curve_line in single_stage_result:
                    curve_line = np.array(curve_line)
                    pre_point = curve_line[0]
                    for cur_point in curve_line[1:]:
                        x1, y1 = int(pre_point[0]), int(pre_point[1])
                        x2, y2 = int(cur_point[0]), int(cur_point[1])

                        thickness = 3
                        cv2.line(pred_line_map, (x1, y1), (x2, y2), (1), thickness, 8)
                        cv2.line(img_show, (x1, y1), (x2, y2), (0, 255, 0), thickness, 8)
                        pre_point = cur_point

                plt.subplot(4, 1, 1)
                plt.imshow(gt_line_map)

                plt.subplot(4, 1, 2)
                plt.imshow(pred_line_map)

                plt.subplot(4, 1, 3)
                plt.imshow(img[:, :, ::-1])

                plt.subplot(4, 1, 4)
                plt.imshow(img_show[:, :, ::-1])
                plt.show()

                cv2.imwrite(os.path.join(save_root, "debug_model_test.jpg"), img_show)
                exit(1)

    # 测评的结果
    metrics = val_evaluator.evaluate(len(val_data_loader.dataset))
    print(metrics)



if __name__ == '__main__':
    # config_path = "./projects/GlasslandBoundaryLine2D/configs/gbld_debug_config.py"
    # config_path = "./projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-seg.py"
    #

    # 用来调试kpi的测评情况
    # config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug/20230803_150518/vis_data/config.py"
    config_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/projects/GlasslandBoundaryLine2D/configs/gbld_debug_config.py"
    checkpoint_path = "/home/dell/liyongjing/programs/mmdetection3d-liyj/work_dirs/gbld_debug/epoch_200.pth"

    print("config_path:", config_path)
    main(config_path, checkpoint_path)