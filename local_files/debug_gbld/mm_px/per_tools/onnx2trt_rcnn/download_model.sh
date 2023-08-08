
CAMERA=${PWD##*/}
MODEL_PATH=$1   #model path on cloud
MODEL_DATE=$2   #model release date

mkdir ${MODEL_DATE}
scp inception-prod-2-8:${MODEL_PATH}/model/checkpoint.onnx ${MODEL_DATE}/checkpoint_${CAMERA}.onnx
scp inception-prod-2-8:${MODEL_PATH}/model/booster.onnx ${MODEL_DATE}/booster_${CAMERA}.onnx
scp inception-prod-2-8:${MODEL_PATH}/model/checkpoint.pt ${MODEL_DATE}/checkpoint_${CAMERA}.pt
scp inception-prod-2-8:${MODEL_PATH}/config.yaml ${MODEL_DATE}/config_${CAMERA}.yaml
scp inception-prod-2-8:${MODEL_PATH}/args.yaml ${MODEL_DATE}/args_${CAMERA}.yaml
scp inception-prod-2-8:${MODEL_PATH}/anchor_cell.bin ${MODEL_DATE}/anchor_cell_${CAMERA}.bin

#combined yaml
scp inception-prod-2-8:${MODEL_PATH}/lld_params.yaml .
scp inception-prod-2-8:${MODEL_PATH}/mod_params.yaml .
scp inception-prod-2-8:${MODEL_PATH}/ds_params.yaml . 
scp inception-prod-2-8:${MODEL_PATH}/ihb_params.yaml .
python ../combined_yaml.py ${MODEL_DATE} ${CAMERA}
rm lld_params.yaml mod_params.yaml ds_params.yaml ihb_params.yaml