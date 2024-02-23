## single-gpu testing
#python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--cfg-options test_evaluator.pklfile_prefix=${RESULT_FILE}]  [--show] [--show-dir ${SHOW_DIR}]
#
## CPU: disable GPUs and run single-gpu testing script (experimental)
#export CUDA_VISIBLE_DEVICES=-1
#python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--cfg-options test_evaluator.pklfile_prefix=${RESULT_FILE}]  [--show] [--show-dir ${SHOW_DIR}]
#
## multi-gpu testing
#./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--cfg-options test_evaluator.pklfile_prefix=${RESULT_FILE}]  [--show] [--show-dir ${SHOW_DIR}]

cd ../
# tpvformer
CHECKPOINT_PATH="./checkpoints/tpvformer/tpvformer_8xb1-2x_nus-seg_20230411_150639-bd3844e2.pth"
python tools/test.py ./projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-seg.py  $CHECKPOINT_PATH

# multi-gpu testing
#bash tools/dist_test.sh ./projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-seg.py  $CHECKPOINT_PATH 1
