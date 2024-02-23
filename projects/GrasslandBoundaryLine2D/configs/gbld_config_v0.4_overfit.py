custom_imports = dict(
    imports=['projects.GrasslandBoundaryLine2D.gbld2d'], allow_failed_imports=False)

experiment_name = "debug_gbld_datasetv2"
work_dir = './work_dirs/' + experiment_name

# data_root = '/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230728_mmdet3d'
# data_root = '/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20230927_mmdet3d'
# data_root = '/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231013_mmdet3d'
# data_root = '/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231013_mmdet3d'
# data_root = '/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231017_mmdet3d_debug'
# data_root = '/home/dell/liyongjing/dataset/glass_lane/glass_edge_overfit_20231020_mmdet3d'
# data_root = '/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231023_mmdet3d'
# data_root = '/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231023_mmdet3d_2'
# data_root = '/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231031_mmdet3d_spline'
# data_root = '/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231101_mmdet3d_spline'
# data_root = '/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231110_mmdet3d_spline_by_cls'
# overfit
# data_root = '/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231116_mmdet3d_spline_by_cls_overfit'
# data_root = '/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/gbld_overfit_20231125_mmdet3d_spline_by_cls'

# 过拟合调试
# data_root = "/home/liyongjing/Egolee/hdd-data/data/dataset/glass_lane/debug_overfit"

# 服务器数据
# data_root = '/data-hdd/liyj/data/dataset/glass_edge_overfit_20231013_mmdet3d'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/glass_edge_overfit_20231020_mmdet3d'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/glass_edge_overfit_20231023_mmdet3d'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/glass_edge_overfit_20231023_mmdet3d_split_line'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231026_mmdet3d_spline'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231030_mmdet3d_spline'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231031_mmdet3d_spline'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231101_mmdet3d_spline'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231102_mmdet3d_spline'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231102_mmdet3d_spline_by_cls_by_visible'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231108_mmdet3d_spline_by_cls'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231110_mmdet3d_spline_by_cls'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231116_mmdet3d_spline_by_cls'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231123_mmdet3d_spline_by_cls'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231202_mmdet3d_spline_by_cls'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231208_mmdet3d_spline_by_cls'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231212_mmdet3d_spline_by_cls'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231213_mmdet3d_spline_by_cls'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231216_mmdet3d_spline_by_cls'
# data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20231218_mmdet3d_spline_by_cls'

data_root = '/data-ssd2/liyj/dataset/gbld_2d/gbld_overfit_20240102_mmdet3d_spline_by_cls'

# 拟合调试
# data_root = "/data-ssd2/liyj/dataset/gbld_2d/debug_overfit"


classes = [
            'road_boundary_line',
            'bushes_boundary_line',
            'fence_boundary_line',
            'stone_boundary_line',
            'wall_boundary_line',
            'water_boundary_line',
            'snow_boundary_line',
            'manhole_boundary_line',
            'others_boundary_line',
        ]
palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 215, 255), (0, 255, 255),
            (230, 216, 173), (128, 0, 128), (203, 192, 255), (238, 130, 238)]

num_classes = len(classes)
test_line_thinkness = 60      # 测试时绘制线的宽度,在原图大小上进行测试, 这个宽度需要根据输入的size进行调整
test_t_iou = 0.5              # 测试时单个实例的ap阈值
# input_size = (960, 544)     # (img_w, img_h)    # 对应输入为[1920, 1080]  需要padding,在右下角
input_size = (960, 608)       # (img_w, img_h)      # 对应输入为[2880, 1860]  需要padding,在右下角

# work_dir = './work_dirs/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_v1.0-min'
batch_size = 8
# batch_size = 2  # 调试

# 模型部分的定义
model = dict(
    type='GBLDMono2Detector',              # 新建一个glass_edge_line_detector的类
    data_preprocessor=dict(                # 可以对图像或者点云进行预处理
        type='Det3DDataPreprocessor',
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            1.0,
            1.0,
            1.0,
        ],
        bgr_to_rgb=False,
        pad_size_divisor=32),

    backbone=dict(
        type='mmdet.ResNet',
        # depth=101,
        depth=50,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe'),          # 根据选择的模型设置预训练模型
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),   # 设置是否使用DCN
        dcn=None,
        stage_with_dcn=(
            False,
            False,
            True,
            True,
        )),

    neck=dict(
        type='mmdet.FPN',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        out_channels=256,
        start_level=0,  #start_level=1, 结合start_level和num_outs设计输出的分辨率和数量
        # add_extra_convs='on_output',
        num_outs=4,   # num_outs=5
        # relu_before_extra_convs=True
    ),

    bbox_head=dict(
        type='GBLDMono2DHead',            # 设计head
        num_classes=num_classes,          # 代表类别的数量, 需要同步修改GgldLineMapsGenerate, loss_cls
        in_channels=256,                  # backbone输入的channel数
        up_scale=1,                       # 是否对从FPN得到的特征进行上采样，大于1为需要，会对neck出来的strides进行调整
                                          # 对GgldLineMapsGenerateV2进行相应的修改

        with_orient=True,                  # 是否预测点的方向
        with_visible=True,                 # 是否预测点的可见性
        with_hanging=True,                 # 是否预测点的悬空属性
        with_covered=True,                 # 是否预测点的草遮挡属性
        with_discriminative=False,         # 是否采用discriminative的聚类方法
        with_point_emb=False,               # 是否预测曲线端点的heatmap和emb

        # 选择在哪些分辨率的stages上进行预测
        # 对neck的feat进行选择,可选择在不同的分辨率上进行head的预测
        # [0]: 代表选择第一个特征,在单层进行预测,可添加为多个
        # 需同步修改strides, stage_loss_weight的数量和大小
        # 需同步修改数据集生成中GgldLineMapsGenerate的gt_down_scales, 同时根据up_scale的数值来调整

        in_feat_index=[0,
                       1,
                       2,
                       # 3,
                       ],

        strides=[                          # 输入feat对应的下采样倍数,
            4,                             # 可以调整backbone的num_outs来调整个数,或者start_level来调整下采样倍数
            8,
            16,
            # 32,
        ],

        stage_loss_weight=[2.0,           # 设置不同stage的loss权重
                           1.5,
                           1.0,
                           # 0.5
                           ],

        gbld_decode=dict(
            type='GlasslandBoundaryLine2DDecode',
            grid_size=4,             # 解析时候的下采样倍数,decode包含在head里面，对于不同的stage动态输入strides
            confident_t=0.2,),       # 超参数,seg-map的confident的阈值

        seg_branch=(256, ),          # 用于预测seg的heatmap
        offset_branch=(256,),        # 用于预测x和y的offset
        seg_emb_branch=(256, ),      # 用于判断是否为同一曲线的emb(可以为分段曲线)
        connect_emb_branch=(256, ),  # 用于判断不同的分段曲线是否可以连接的emb(暂时实际解析没用)
        cls_branch=(256, ),          # 用于预测seg的类别
        orient_branch=(256, ),       # 用于预测点的朝向
        visible_branch=(256,),        # 用于预测点的可见属性
        hanging_branch=(256,),        # 用于预测点的悬空属性
        covered_branch=(256,),        # 用于预测点的草遮挡属性
        discriminative_branch=(256,),  # 采用discriminative的方式进行聚类

        seg_point_branch=(256, ),          # 用于预测曲线端点seg的heatmap
        seg_point_emb_branch=(256, ),      # 用于预测曲线端点的emb

        num_seg=1,                   # 预测的channel数量
        num_offset=2,
        num_seg_emb=1,
        num_connect_emb=1,
        num_orient=2,                # 采用sin和cos的方式进行预测
        num_visible=1,               # 预测可见属性
        num_hanging=1,               # 预测悬空属性
        num_covered=1,               # 预测被草遮挡属性
        num_discriminative_emb=16,  # discriminative聚类特征的维度

        num_seg_point=1,             # 预测曲线端点heatmap的channel数量
        num_seg_point_emb=1,         # 预测曲线端点emb数量

        # loss
        loss_seg=dict(
            type='GbldSegLoss',
            focal_loss_gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            use_dist_weight=False,  # 是否使用距离加权, y越大权重越大
            max_dist_weight=3.0),   # 最大y的权重

        loss_offset=dict(
            type='GbldOffsetLoss',
            loss_weight=1.0),

        loss_seg_emb=dict(
            type='GbldEmbLoss',
                pull_margin=0.5,
                push_margin=1.0,
                loss_weight=10.0),     # 10

        loss_connect_emb=dict(
            type='GbldEmbLoss',
            pull_margin=0.5,
            push_margin=1.0,
            loss_weight=1.0),

        loss_cls=dict(
            type='GbldClsLoss',
            num_classes=num_classes,
            loss_weight=5.0),

        loss_orient=dict(
            type='GbldOrientLoss',
            loss_weight=2.0),

        loss_visible=dict(
            type='GbldClsLoss',
            num_classes=1,
            loss_weight=5.0),

        loss_hanging=dict(
            type='GbldClsLoss',
            num_classes=1,
            loss_weight=5.0),

        loss_covered=dict(
            type='GbldClsLoss',
            num_classes=1,
            loss_weight=5.0),

        loss_discriminative=dict(
            type='GbldDiscriminativeLoss',
            delta_var=0.5,
            delta_dist=3.0,
            norm=2,
            alpha=1.0,
            beta=1.0,
            gamma=0.001,
            loss_weight=5.0),

        loss_seg_point=dict(
            type='GbldSegLoss',
            focal_loss_gamma=2.0,
            alpha=0.25,
            loss_weight=0.5,
            use_dist_weight=False,  # 是否使用距离加权, y越大权重越大
            max_dist_weight=3.0),  # 最大y的权重

        loss_seg_point_emb=dict(
            type='GbldEmbLoss',
            pull_margin=0.5,
            push_margin=1.0,
            loss_weight=5.0),  # 10

        conv_bias=True,
        dcn_on_last_conv=True),



    # 待处理,目的还不明确,先保留
    train_cfg=dict(
        allowed_border=0,
        code_weight=[
            1.0,
            1.0,
            0.2,
            1.0,
            1.0,
            1.0,
            1.0,
            0.05,
            0.05,
        ],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=200))

# 数据集加载部分
input_modality = dict(use_lidar=False, use_camera=True)
backend_args = None

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='GbldMono2dDatasetV2',
        data_root=data_root,
        data_prefix=dict(
            img='train/images',
            ann='train/jsons'),
        ann_file='gbld_infos_train.pkl',

        # data_prefix=dict(
        #     img='test/images',
        #     ann='test/jsons'),
        # ann_file='gbld_infos_test.pkl',

        load_type='mv_image_based',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='GgldLoadLines', name="load_gt_lines"),

            # dict(type='GgldRandomRotate', prob=0.3, angle_range=12),
            # dict(type='GgldRandomRotate', prob=1.0, angle_range=12),      # debug

            # 该方法实现还有问题, TODO
            # dict(type='GgldRandomCrop', prob=0.8, max_margin_scale=0.4, keep_ratio=True),
            dict(type='GgldRandomCrop', prob=0.5, max_margin_scale=0.4, keep_ratio=True),
            # dict(type='GgldRandomCrop', prob=1.0, max_margin_scale=0.4, keep_ratio=True),   # debug

            dict(type='GgldResize', scale=input_size,
            # keep_ratio=False,
            keep_ratio=True,
            ),

            dict(type='GgldColor', prob=0.5),

            # 当输入的img-size为固定的时候, 对应上面的keep_ratio=false, 设置size_divisor保证为32的倍数
            # 当输入的img-seize不固定的时候, 对应上面的keep_ratio=true,设置size为固定的, 保证整个batch的输入size是固定的
            # 需要设置size=(w, h), 设置pad后的size也需要尽量保证为32的倍数
            # dict(type='Pad', size_divisor=32),
            dict(type='Pad', size=input_size),

            # direction=['horizontal', 'vertical', 'diagonal']
            dict(type='GgldRandomFlip', prob=0.5, direction=['horizontal']),

            # gt_down_scales用来产生不同下采样倍数下的gt,需要跟head输出的分辨率保持一致和stage保持一致
            # dict(type='GgldLineMapsGenerateV2', gt_down_scales=[4, 8, 16, 32], num_classes=num_classes),
            dict(type='GgldLineMapsGenerateV2',
                 gt_down_scales=[4, 8, 16],   # up_scale = 2: [2, 4, 8] # up_scale = 4: [1, 2, 4]
                 num_classes=num_classes,
                 filter_small_line=True,      # 是否过滤断线
                 filter_length=32,            # 过滤曲线的长度
                 ),
            # dict(type='GgldLineMapsGenerateV2', gt_down_scales=[4, 8], num_classes=num_classes),
            # dict(type='GgldLineMapsGenerateV2', gt_down_scales=[4], num_classes=num_classes),

            # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PackGbldMono2dInputs',
                keys=[
                    'img',
                    'gt_line_maps_stages',
                ],
                meta_keys=['gt_lines', 'sample_idx', 'scale_factor', 'ori_shape']
            ),
        ],
        # 标注名称不在列表的将会被过滤
        metainfo=dict(classes=classes,
                      version='v1.0-mini'),

        test_mode=False,
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='GbldMono2dDatasetV2',
        data_root=data_root,
        data_prefix=dict(
            img='test/images',
            ann='test/jsons'),
        ann_file='gbld_infos_test.pkl',

        # overfit train
        # data_prefix=dict(
        #     img='train/images',
        #     ann='train/jsons'),
        # ann_file='gbld_infos_train.pkl',

        load_type='mv_image_based',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile',
                 imdecode_backend='cv2',
                 backend_args=None),
            dict(type='GgldLoadLines', name="load_gt_lines"),

            dict(type='GgldResize', scale=input_size,
            # keep_ratio=False,
            keep_ratio=True
            ),

            # 当输入的img-size为固定的时候,设置size_divisor保证为32的倍数
            # 当输入的img-seize不固定的时候,设置size为固定的, 保证整个batch的输入size是固定的
            # 设置pad后的size也需要尽量保证为32的倍数
            # dict(type='Pad', size_divisor=32),
            dict(type='Pad', size=input_size),

            # direction=['horizontal', 'vertical', 'diagonal']
            # dict(type='GgldRandomFlip', prob=1.0, direction=['horizontal']),

            # gt_down_scales用来产生不同下采样倍数下的gt
            # dict(type='GgldLineMapsGenerate', gt_down_scales=[4, 8, 16, 32], num_classes=3),

            # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PackGbldMono2dInputs',
                keys=[
                    'img',
                    'gt_line_maps_stages',
                ],
                meta_keys=['eval_gt_lines', 'sample_idx', 'scale_factor', 'ori_shape',
                           'ori_gt_lines', 'ori_eval_gt_lines']
            ),
        ],
        metainfo=dict(classes=classes,
                      version='v1.0-mini'),
        test_mode=True,
        backend_args=None))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='GbldMono2dDataset',
        data_root=data_root,
        data_prefix=dict(
            img='test/images',
            ann='test/jsons'),
        ann_file='gbld_infos_test.pkl',
        load_type='mv_image_based',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='GgldLoadLines', name="load_gt_lines"),

            dict(type='GgldResize', scale=input_size,
            keep_ratio=True,
            # keep_ratio=False
            ),
            # dict(type='Pad', size_divisor=32),
            dict(type='Pad', size=input_size),

            dict(type='PackGbldMono2dInputs',
                 keys=['img', ],
                    meta_keys=['eval_gt_lines', 'sample_idx', 'scale_factor', 'ori_shape',
                               'ori_gt_lines', 'ori_eval_gt_lines']),
        ],
        metainfo=dict(classes=classes,
                      version='v1.0-mini'),
        test_mode=True,
        backend_args=None))

val_evaluator = dict(
    type='GbldMetric',
    data_root=data_root,
    ann_file='gbld_infos_test.pkl',
    dataset_meta=dict(classes=classes,
                      version='v1.0-mini',
                      palette=palette),
    test_stage=[0],        # 模型在不同的分辨率上都有预测的结果,选择测评的位置
    metric='line_instance',
    line_thinkness=test_line_thinkness,     # 绘制线的宽度
    t_iou=test_t_iou,                              # ap实例的iou阈值
    rescale=True,          # 在原图分辨率上进行测评,模型预测默认是rescale=True的结果
    backend_args=None)

test_evaluator = dict(
    type='GbldMetric',
    data_root=data_root,
    ann_file='gbld_infos_test.pkl',
    dataset_meta=dict(classes=classes,
                      version='v1.0-mini',
                      palette=palette),
    test_stage=[0],
    metric='line_instance',
    line_thinkness=test_line_thinkness,
    t_iou=test_t_iou,  # ap实例的iou阈值
    rescale=True,  # 在原图分辨率上进行测评
    backend_args=None)

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='GbldVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer',
    line_color=palette,
    text_color=palette,
    line_width=3,
    font_size=18,
    with_text=True,
    classes=classes)   # None或者classes, None 为显示数据类别, classes为显示具体的类别

# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_begin=10, val_interval=1)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=250, val_begin=50, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=35, norm_type=2))

# 根据base_batchsize和实际的batchsize自动调整学习率
# 然后线性的下降学习率,https://arxiv.org/abs/1706.02677, enable为false则不采用
auto_scale_lr = dict(enable=False, base_batch_size=16)

# 这里有两个schedule
param_scheduler = [
    # begin: Step at which to start updating the learning rate.
    # end: Step at which to stop updating the learning rate. Defaults to INF.
    # 可以理解为warmup, 在optimizer中有个基础的lr
    # start_factor的意思为将lr*start_factor作为一开始的lr
    # 然后通过begin-end的迭代,即将lr逐渐提升为在optimizer中设置的lr,也就是[lr*start_factor, lr]的提升过程
    dict(
        type='LinearLR',
        start_factor=0.3333333333333333,
        by_epoch=False,
        begin=0,
        end=500
    ),

    # MultiStepLR生效的范围为[begin, end]
    # 每到一个milestones的数值,就会下降gamma
    dict(
        type='MultiStepLR',
        begin=0,
        # end=12,
        end=250,
        by_epoch=True,
        # milestones=[
        #     8,
        #     11,
        # ],
        milestones=[
            20,      # debug
            166,     # 大概在[end - begin] * 8/12
            229,     # 大概在[end - begin] * 11/12
        ],
        gamma=0.1),
]

default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # checkpoint=dict(type='CheckpointHook', interval=-1),
    checkpoint=dict(type='CheckpointHook', interval=50),    # 设置保存checkpoint的间隔
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook',
                       draw_gt=True, draw_pred=True,     # 控制可视化的内容
                       show=False, wait_time=1),         # 控制是否显示,以及显示的间距
    # gbld2d_transform_hook=dict(type='Gbld2DTransformHook',
    #                            name="Gbld2DTransformHook",
    #                            set_first_epoch=True)
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
launcher = 'none'


# 解决分布式训练报错
# RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by making sure all `forward` function outputs participate in calculating loss.
find_unused_parameters = True