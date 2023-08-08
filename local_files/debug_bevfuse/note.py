# 1.编译安装
# https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion
# 需要先编译运行,要不然导入时会提示找不到一些库
# python projects/BEVFusion/setup.py develop

# 2.加载模型
# 在加载官方权重的时候,出现权重shape不匹配的情况
# size mismatch for pts_middle_encoder.conv_input.0.weight: copying a param with shape
# https://github.com/open-mmlab/mmdetection3d/issues/2584
# 查看debug_others.py的def convert_checkpoints()实现



