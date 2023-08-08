
CAMERA=${PWD##*/}
TRT_FILE=xp_model_qnx_fp16_${CAMERA}.trt
MODEL_DATE=$1
TARGET_IP=root@10.160.66.169

sshpass -p root scp ${MODEL_DATE}/checkpoint_${CAMERA}.onnx ${TARGET_IP}:/mnt/perception/onnx_trt/
sshpass -p root ssh ${TARGET_IP} "echo convert trt; cd /mnt/perception/onnx_trt; ./onnx2trt_rcnn -d 16 -b 2 -o ${TRT_FILE} checkpoint_${CAMERA}.onnx ;"
sshpass -p root scp ${TARGET_IP}:/mnt/perception/onnx_trt/${TRT_FILE} ${MODEL_DATE}/