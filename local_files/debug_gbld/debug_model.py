# 用来调试生成模型

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import torch

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


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


    # 用来调试模型的构建
    debug_model(cfg)


#----------------------------------------------调试模型构建----------------------------------------------
def debug_model(cfg):
    default_scope = 'mmdet3d'
    _experiment_name = "123"

    from mmengine.registry import DefaultScope


    default_scope = DefaultScope.get_instance(  # type: ignore
        _experiment_name,
        scope_name=default_scope)

    from mmengine.registry import MODELS
    model = cfg['model']
    debug_model = MODELS.build(model)

    batch_inputs_dict = {}
    batch_inputs_dict['imgs'] = torch.randn((1, 3, 512, 512))

    # tensor
    # outs = debug_model(batch_inputs_dict, mode="tensor")
    # for i, out in enumerate(outs):
    #     for j, _out in enumerate(out):
    #         print(i, j, _out.shape)

    # predict
    # fake infos
    from mmdet3d.structures import Det3DDataSample
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes
    data_sample = Det3DDataSample()
    meta_info = dict(
    img_shape=(800, 1196, 3),
    pad_shape=(800, 1216, 3))
    gt_instances_3d = InstanceData(metainfo=meta_info)
    gt_instances_3d.bboxes_3d = BaseInstance3DBoxes(torch.rand((5, 7)))
    gt_instances_3d.labels_3d = torch.randint(0, 3, (5,))
    data_sample.gt_instances_3d = gt_instances_3d

    # fake data
    gt_instances = InstanceData(metainfo=meta_info)
    gt_instances.gt_seg = [torch.randn((1, 1, 128, 128)),
                           torch.randn((1, 1, 64, 64)),
                           torch.randn((1, 1, 32, 32)),
                           torch.randn((1, 1, 16, 16))]

    gt_instances.gt_offset = [torch.randn((1, 2, 128, 128)),
                           torch.randn((1, 2, 64, 64)),
                           torch.randn((1, 2, 32, 32)),
                           torch.randn((1, 2, 16, 16))]
    gt_instances.gt_line_index = [torch.randn((1, 1, 128, 128)),
                           torch.randn((1, 1, 64, 64)),
                           torch.randn((1, 1, 32, 32)),
                           torch.randn((1, 1, 16, 16))]
    gt_instances.gt_ignore_mask = [torch.randn((1, 1, 128, 128)) > 0 ,
                           torch.randn((1, 1, 64, 64)) > 0,
                           torch.randn((1, 1, 32, 32)) > 0 ,
                           torch.randn((1, 1, 16, 16)) > 0]
    gt_instances.gt_foreground_mask = [torch.randn((1, 1, 128, 128)) > 0,
                           torch.randn((1, 1, 64, 64)) > 0,
                           torch.randn((1, 1, 32, 32)) > 0,
                           torch.randn((1, 1, 16, 16)) > 0]
    gt_instances.gt_line_id = [torch.randn((1, 1, 128, 128)),
                           torch.randn((1, 1, 64, 64)),
                           torch.randn((1, 1, 32, 32)),
                           torch.randn((1, 1, 16, 16))]
    gt_instances.gt_line_cls = [torch.randn((1, 10, 128, 128)),
                           torch.randn((1, 10, 64, 64)),
                           torch.randn((1, 10, 32, 32)),
                           torch.randn((1, 10, 16, 16))]

    data_sample.gt_instances = gt_instances

    batch_img_metas = [data_sample]
    #
    # out = debug_model(batch_inputs_dict, data_samples=batch_img_metas, mode="predict")
    # # out[0]._pred_instances.single_result    # TODO 结果的正确性需要调试


    out = debug_model(batch_inputs_dict, data_samples=batch_img_metas, mode="loss")
    print(out)


if __name__ == '__main__':
    config_path = "./projects/GlasslandBoundaryLine2D/configs/gbld_debug_config.py"
    # config_path = "./projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-seg.py"

    # 设置不同的模式,对model进行调试,是程序的入口,需要在定义的位置进行打断点调试
    print("config_path:", config_path)
    main(config_path)