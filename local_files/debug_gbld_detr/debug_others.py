


def debug_model_registry():
    default_scope = 'mmdet3d'
    _experiment_name = "123"

    from mmengine.registry import DefaultScope


    default_scope = DefaultScope.get_instance(  # type: ignore
        _experiment_name,
        scope_name=default_scope)

    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmdet.registry import MODELS as MMDET_MODELS
    from mmdet3d.registry import MODELS as MMMDET3D_MODELS

    # MMENGINE_MODELS包含为mmdet和mmdet3d的模块，为其父节点，调用子节点的模块时需要按照mmdet.xx或者mmdet3d.xx的方式进行调用
    bbox_head = {'type': 'mmdet.DETRHead', 'num_classes': 80, 'embed_dims': 256,
     'loss_cls': {'type': 'mmdet.CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0,
                  'class_weight': 1.0 },
     'loss_bbox': {'type': 'mmdet.L1Loss', 'loss_weight': 5.0},
     'loss_iou': {'type': 'mmdet.GIoULoss', 'loss_weight': 2.0}, 'train_cfg': None, 'test_cfg': None}



    bbox_head = MMENGINE_MODELS.build(bbox_head)
    a = "1"


if __name__ == "__main__":
    print("Start")
    debug_model_registry()
    print("End")
