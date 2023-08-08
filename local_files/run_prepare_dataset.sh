cd ../
#https://github.com/open-mmlab/mmdetection3d/tree/main/docs/en/advanced_guides/datasets
# 1. nuscences
#python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
# 2.nuscences mini
python tools/create_data.py nuscenes --root-path ./data/nuscenes --version v1.0-mini --out-dir ./data/nuscenes --extra-tag nuscenes
