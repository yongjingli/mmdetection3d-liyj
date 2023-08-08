
export UPLOAD_PATH="https://artifactory.xiaopeng.us/artifactory/perception_release/PDK601/standard/libonnxtrt/20220207_1_0_5/"

cd bin
tar -zcvf onnx2trt_v1.0.5_orin_pdk601.tgz libonnxtrt_orin.so libonnxtrt_dla.so onnx2trt_orin readCalibrationTable test_multi_models test_kpi test_models profile_model

curl -s -u owen:chengzi0109 -T libonnxtrt_orin.so ${UPLOAD_PATH}/libonnxtrt.so
curl -s -u owen:chengzi0109 -T libonnxtrt_dla.so ${UPLOAD_PATH}/libonnxtrt_dla.so
curl -s -u owen:chengzi0109 -T onnx2trt_v1.0.5_orin_pdk601.tgz ${UPLOAD_PATH}

cd -

