
CAMERA=${PWD##*/}
TRT_FILE=xp_model_qnx_fp16_${CAMERA}.trt
MODEL_DATE=$1  #model release date
MODEL_PATH=https://artifactory.xiaopeng.us/artifactory/perception_release/${CAMERA}/2019${MODEL_DATE}
LATEST_PATH=https://artifactory.xiaopeng.us/artifactory/perception_release/latest/

curl -u owen:chengzi0109 -T ${MODEL_DATE}/${TRT_FILE} ${MODEL_PATH}/${TRT_FILE} | tee ${MODEL_DATE}/${TRT_FILE}.log
curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/${TRT_FILE}.log ${MODEL_PATH}/${TRT_FILE}.log
curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/checkpoint_${CAMERA}.onnx ${MODEL_PATH}/checkpoint.onnx
curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/booster_${CAMERA}.onnx ${MODEL_PATH}/model/booster.onnx
curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/checkpoint_${CAMERA}.pt ${MODEL_PATH}/checkpoint.pt
curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/config_${CAMERA}.yaml ${MODEL_PATH}/config.yaml
curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/args_${CAMERA}.yaml ${MODEL_PATH}/args.yaml
curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/deploy_${CAMERA}.yaml ${MODEL_PATH}/deploy_${CAMERA}.yaml
curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/anchor_cell_${CAMERA}.bin ${MODEL_PATH}/anchor_cell_${CAMERA}.bin

#update latest release
# curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/${TRT_FILE}     ${LATEST_PATH}/${TRT_FILE}
# curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/${TRT_FILE}.log ${LATEST_PATH}/${TRT_FILE}.log
# curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/config_${CAMERA}.yaml     ${LATEST_PATH}/config_${CAMERA}.yaml
# curl -s -u owen:chengzi0109 -T ${MODEL_DATE}/deploy_${CAMERA}.yaml     ${LATEST_PATH}/deploy_${CAMERA}.yaml
