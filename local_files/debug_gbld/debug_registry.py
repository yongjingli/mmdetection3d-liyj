def test_model_registry():
    import torch
    import torch.nn as nn
    from mmengine import Registry, MODELS as MMENGINE_MODELS

    # 通过locations的方式进行注册, 采用导入模块的方式使用
    # 会调用models同级下的__init__.py,从这里导入该目录下需要的模块
    # mmdet3d采用pip install -v -e .的方式进行安装,可以在其他各个目录种进行调用
    # -e采用的可编辑的按照模型，直接修改源码就不用重复安装, pip list命令可以看到按照的位置
    # "-v" means verbose, or more output
    # "-e" means installing a project in edtiable mode,
    # thus any local modifications made to the code will take effect without reinstallation.

    # mmdet3d和mmalpha的同时导入, 采用这种方式导入mmdet3d.models好像不行
    MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmalpha', locations=['mmalpha.models', "mmdet3d.models"])

    # 另外一种实现方式, 从mmdet3d直接导入已经registry的models
    # from mmdet3d.registry import MODELS
    # MODELS = Registry('model', parent=MODELS, scope='mmalpha', locations=['mmalpha.models'])

    # 可以查看module_dict和_locations, 其中_locations实在build的时候才会加载 later load, 所以在module_dict是看不到_locations里的模块信息
    # MODELS.import_from_location()
    print("MODELS:", MODELS.module_dict)
    print("MODELS:", MODELS._locations)

    # 在本地register, 在module_dict可以看到模块信息
    @MODELS.register_module()
    class LogSoftmaxModule(nn.Module):
        def __init__(self, dim=None):
            super().__init__()

        def forward(self, x):
            print('call LogSoftmaxModule.forward')
            return x
    print("MODELS:", MODELS.module_dict)

    model = MODELS.build(dict(type="LogSoftmax"))
    input = torch.randn(2)
    output = model(input)
    print(output)

    model = MODELS.build(dict(type="LogSoftmaxModule"))
    input = torch.randn(2)
    output = model(input)
    print(output)
    print("locations1:", MODELS._locations, MODELS._imported, MODELS.module_dict)


    # from mmengine import Registry, MODELS
    from mmengine.registry import MODELS

    model = MODELS.build(dict(type="LogSoftmax"))
    print("locations2:", MODELS._locations, MODELS._imported, MODELS.module_dict)
    input = torch.randn(2)
    output = model(input)
    print(output)


def test_model_registry2():
    # 解决导入mmdet3d的问题
    # (1) 采用下面的这种方还是导入，可以将registry下的都注册进来，但是使用不了那种嵌套的registry
    # 例如registry-A种调用registry-B, 那么就找不到registry-B的定义
    # from mmdet3d.registry import MODELS as MODELS

    # 在mmdetection3d中,在runner中会进行DefaultScope.get_instance的实例化
    # 会将from mmengine.registry import MODELS里的scope进行实例化
    # 这样就可以调用mmdet3d中的定义

    default_scope = 'mmdet3d'
    _experiment_name = "4554"

    from mmengine.registry import DefaultScope

    default_scope = DefaultScope.get_instance(  # type: ignore
        _experiment_name,
        scope_name=default_scope)

    from mmengine.registry import MODELS
    model = dict({'type': 'FCOSMono3D',
                  'data_preprocessor': {'type': 'Det3DDataPreprocessor',
                                        'mean': [103.53, 116.28, 123.675],
                                        'std': [1.0, 1.0, 1.0],
                                        'bgr_to_rgb': False, 'pad_size_divisor': 32},
                  'backbone': {'type': 'mmdet.ResNet', 'depth': 101, 'num_stages': 4,
                               'out_indices': (0, 1, 2, 3), 'frozen_stages': 1,
                               'norm_cfg': {'type': 'BN', 'requires_grad': False},
                               'norm_eval': True, 'style': 'caffe',
                               'init_cfg': {'type': 'Pretrained', 'checkpoint': 'open-mmlab://detectron2/resnet101_caffe'},
                               'dcn': {'type': 'DCNv2', 'deform_groups': 1, 'fallback_on_stride': False},
                               'stage_with_dcn': (False, False, True, True)},
                  'neck': {'type': 'mmdet.FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256,
                           'start_level': 1, 'add_extra_convs': 'on_output', 'num_outs': 5, 'relu_before_extra_convs': True},
                  'bbox_head': {'type': 'FCOSMono3DHead', 'num_classes': 10, 'in_channels': 256, 'stacked_convs': 2,
                                'feat_channels': 256, 'use_direction_classifier': True, 'diff_rad_by_sin': True,
                                'pred_attrs': True, 'pred_velo': True, 'dir_offset': 0.7854, 'dir_limit_offset': 0,
                                'strides': [8, 16, 32, 64, 128], 'group_reg_dims': (2, 1, 3, 1, 2), 'cls_branch': (256,),
                                'reg_branch': ((256,), (256,), (256,), (256,), ()), 'dir_branch': (256,),
                                'attr_branch': (256,), 'loss_cls': {'type': 'mmdet.FocalLoss', 'use_sigmoid': True,
                                                                    'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 1.0},
                                'loss_bbox': {'type': 'mmdet.SmoothL1Loss', 'beta': 0.1111111111111111, 'loss_weight': 1.0},
                                'loss_dir': {'type': 'mmdet.CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0},
                                'loss_attr': {'type': 'mmdet.CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0},
                                'loss_centerness': {'type': 'mmdet.CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0},
                                'bbox_coder': {'type': 'FCOS3DBBoxCoder', 'code_size': 9},
                                'norm_on_bbox': True, 'centerness_on_reg': True, 'center_sampling': True, 'conv_bias': True,
                                'dcn_on_last_conv': True},
                  'train_cfg': {'allowed_border': 0, 'code_weight': [1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
                                'pos_weight': -1, 'debug': False}, 'test_cfg': {'use_rotate_nms': True, 'nms_across_levels': False,
                                                                                'nms_pre': 1000, 'nms_thr': 0.8, 'score_thr': 0.05,
                                                                                'min_bbox_size': 0, 'max_per_img': 200}})
    MODEL = MODELS.build(model)
    print("Done")


if __name__ == "__main__":
    print("Start")
    # test_model_registry()
    test_model_registry2()
    print("End")