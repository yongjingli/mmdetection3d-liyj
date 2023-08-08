Introduction

Inference Service provides a producer-consumer mode service to do engine inference. The purpose of this service is to manage inference engines more clearly, and reduce the overall memory usage as loading TensorRT releated libraries. This design document of this service can be found at:
https://xiaopeng.feishu.cn/docs/doccno4AB9lYuod58kgU3gzkUFd#

Compile Command:
check out to the folder "Inference_Service_Demon", and then type "make TARGET_ARCH=aarch64" to compile. After compilation, two apps, "Network_Client" and "Inference_Service" are in this folder. The libonntrt_client.so is located in the bin folder of onnx2trt_orin.

Test Commands:

Test Block Style Engine Inference:
1. Check out to the folder "Inference_Service_Demon".
2. use the test_multi_models under this folder.
    2-1.run test_multi_models in a process: ./test_multi_models -e ./ngp_main_v1.0.0_int8b1_lld.trt -i ./main.png -g -t 10 -s ./libclient.so -G 1 -m CudaGraph=True -T 2
        Note: you have to specify -G 1 for GPU input and -m for engine config, as well as -T 2 for blocking mode engine inference.
    2-2. launch inference server in another process: ./Inference_Service -sn 1 -n 1000 -sp /tmp/client_socket_0
        Note: -sn means "service number", that is how many engine you are going to serve. The number should be the same as -e you specify in test_multi_models
              -n means the number of iterations you are going to run. It should be at least 10 times larger than the -t option you specify in test_multi_models

Test Non-Block Style Engine Inference:
1. Check out to the folder "Inference_Service_Demon".
2. Use Network_Client and Inference_Service under this folder.
    2-1. run Network_Client: ./Network_Client -sn 1 -n 20
        Note: -sn means "service number", that is how many engine you are going to serve. The number should be the same as -sn specify in Inference_Server
              -n means the number of iterations you are going to run. It should be at least 10 times larger than the -n option you specify in Inference_Server
    2-1. run Inferece_Service:./Inference_Service -sn 1 -n 200