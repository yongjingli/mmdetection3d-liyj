# Maskrcnn Convert
## Depends: 
- pytorch1.2/maskrcnn-benchmark or Caffe2/detectron
- Tensorrt5 or 6
- onnx
- onnx-tensorrt

## Usage:
- Build third party packages( for JetsonXavier, TensorRT5.0.3 )
```
  make third_party 
```
- Build tools and so
```
  make (or make all)
```
- Convert maskrcnn model (.pth) to onnx  ( on Server/PC, depends pytorch1.2 & maskrcnn-benchmark )
```
tar -zxvf third_party/maskrcnn_export_1127.tgz
cd export_onnx/demo
python export_onnx.py --config-file 1029/e2e_mask_rcnn_R_50_FPN_1x.yaml  MODEL.WEIGHT 1029/0.9596_model_1790000.pth
```
- Convert model (.onnx) to cudaEngine (.trt) ( on Target, etc. DriveXavier )
```
  make convert_maskrcnn convert_xpmodel
```
- Test several methods of integration
```
  make test ( Depends trt files and images )
```
- Commands for X86-ubuntu, TensorRT-5.0.2.6 or TensorRT-5.0.0.10. GPU5 server has so for TensorRT5.0.2.
```
  make third_party ARCH=x86_64 TENSORRTPATH=third_party/TensorRT-5.0.2.6 TENSORRTLIB=/usr/lib/x86_64-linux-gnu/
  make ARCH=x86_64 TENSORRTPATH=third_party/TensorRT-5.0.2.6 TENSORRTLIB=/usr/lib/x86_64-linux-gnu/
```
- Commands for QNX, TensorRT-5.1.3/6.0.0. Set CUDA_INSTALL_DIR, CUDNN_INSTALL_DIR if needed.
```
  make -f Makefile_QNX third_party 
  make -f Makefile_QNX [ CUDA_INSTALL_DIR=xxxx CUDNN_INSTALL_DIR=xxxx TENSORRT_INCDIR=xxx TENSORRT_LIBDIR=xxx ]
```

## ChangeLog
- Create @ 2018.8.30:
* 1.Splited mask-rcnn with nods£ºrpn_cls_probs_fpnX/rpn_bbox_pred_fpnX, X=2~6, including some convolutions of roi_fpn.
* 2.Converted the first half of mask-rcnn to onnx format after adding code for 'upsample' and 'affinechannel'. Then used onnx-tensorrt to support plugin layer for 'upsample'.
* 2.Wrote a sample which integrated C/C++ & Python & TensorRT4 to run maskrcnn on TX2. It cost about 0.8s to process one 1280x768 image. 
- Add Plugins @ 2018.9.14:
* 1.Use the onnx operator "ATen" for operator RoIAlign, BatchPermutation, GenerateProposalsOp, CollectAndDistributeFpnRpnProposalsOp
* 2.Add Plugins code for GenerateProposalsOp(CPU,FP32), CollectAndDistributeFpnRpnProposalsOp(CPU,FP32), RoIAlign(CUDA,FP32), BatchPermutation(CUDA,FP32/FP16)
* 3.Use contract to reduce the input number of CollectAndDistributeFpnRpnProposalsOp (Reduced to 2 from 10 )
- Update Plugins and sample @ 2018.11.14:
* 1.Add the CUDA codes for CollectAndDistributeFpnRpnProposalsOp(GPU,FP32). 
* 2.Updated the Interface functions and header file for libonnxtrt.so ( .so will be easier using in other programs like ROS). Sample: src/maskrcnn.cpp
* 3.Combined the codes for Maskrcnn-retnet50 with plugin for Feng'model and run parallel.
- Update DLA support @ 2018.11.20:
* 1.Add support of DLA when using TensorRT5. Run correct before layer No.173.
* 2.Extend flag "TRT_DEBUGLEVEL=-1" for skipping Plugin layers to avoid overflow when building Engine with DLA. 
* 3.Remove configureWithFormat which is incompatible with TensorRT5
- Update DLA support @ 2018.11.30: 
* Add new sample, using openCV to read encoded images( readImageByOpenCV ) , openmp to run  mulit-models parallel
* example: 
* bin/maskrcnn_mulit_models -s bin/libonnxtrt.so -e model_final_608x960b1.trt -i output6_960x608_134544.645.png -o output_final_960x608_mulit.ppm -m model_final/masknetweight.bin -e LLD_604x960fp32_MKZ.trt -i input_960_604.jpg -o output_LLD_960x604_mulit.ppm
* MaskNet need weight "masknetweight.bin" 
- Update third_party packages @ 2018.12.19:
* create folder third_party and add packages ( onnx-tensorrt, protobuf3.5.1 )
* modified codes in plugin to support tensorRT4/5. Add builtin_mask_importers.h & builtin_mask_plugins.h
* Usage: make third_party & make
- Update third_party packages @ 2019.3.4:
* Add support of X86-ubuntu16.04. Such as docker of GPU05 server.
- Update third_party packages @ 2019.4.12:
* Merged the QNX version of onnx2trt tool with ¡°dev_perception¡° branch, removed CMake File in onnx-tensorrt and directly using the source.
* Moved sample files such as maskrcnn.cpp to sample folder.
* Added code for int8 and changed the calibrator to support reading images from list.
- Update onnx2trt tool @ 2019.8.9:
* Updated maskrcnn to support keypoint-branch.
* Added code for gemvInt8 to replace FC-layer inside TensorRT.
- Update onnx2trt tool @ 2019.11.14:
* Combined Roi layers to one Plugin. 
* Supported rpn/fpn pruned model.
* Supported onnx file for mask-branch & keypoint-branch.
* Ported to TensorRT6.
- Update onnx2trt tool @ 2019.11.22:
* Supported TensorRT6, fixed the second input size of plugin.
* Supported maskrcnn with 8 keypoint(from Chenghao).
- Update onnx2trt tool @ 2019.12.4:
* Combined Roi layers supported batch=2. 
* Supported Upsample Int8 Plugin.(Based on TensorRT6 IPluginV2IOExt)
- Update onnx2trt tool @ 2019.12.25:
* Selected DLACalibrator for DLA Int8.(Based on TensorRT6 IInt8EntropyCalibrator2)
* Supported two DLA models.

TODO:
1.Optimize mulit-models on GPU and DLA.

Tested input:
Input Image size: 1920x1008, 1920x1208
Input dimensions (Resnet50)£º1x3x704x1344, 2x3x376x512, 1x3x640x960, 1x3x608x960

Memory Usage:(1x3x704x1344)
Total Activation Memory: 1409 MB (1477519360 Bytes) ( CUDA: 647 MB)

## input & output
* input data: RGB(Float32) - Means[102.9801,115.9465,122.7717]
  Source Image -> decode to RGB -> Crop & resize to input size( such as 320x576) -> convert to Float32 -> sub  means[102.9801,115.9465,122.7717] ->  input data 
* output data: 
TensorRT-engine for Mask-rcnn( Faster-rcnn )
class_num = 6, Input = 3x640x960(CHW)
Out0:gpu_0/rois_1, shape:[300, 5, 1]
Out1:gpu_0/bbox_pred_1, shape:[300, 24, 1]
Out2:gpu_0/cls_prob_1, shape:[300, 6, 1]
Out3:gpu_0/fpn_res2_2_sum_1, shape:[256, 80, 144]
Out4:gpu_0/fpn_res3_3_sum_1, shape:[256, 40, 72]
Out5:gpu_0/fpn_res4_5_sum_1, shape:[256, 20, 36]
Out6:gpu_0/fpn_res5_2_sum_1, shape:[256, 10, 18]
After mask-process
Out0:gpu_0/rois_1, shape:[300, 5, 1]
Out1:gpu_0/bbox_pred_1, shape:[300, 24, 1]
Out2:gpu_0/cls_prob_1, shape:[300, 6, 1]
Out3:gpu_0/mask_roi, shape:[100, 5]
Out4:gpu_0/mask_roi_fpn, shape:[100, 21]
Out5:gpu_0/mask_fcn_probs, shape:[100, 28, 28, 6]

##
 bin/maskrcnn_mulit_models -s bin/libonnxtrt.so -e model_final_608x960b1.trt -i output6_960x608_134544.645.png -o output_final_960x608_mulit.ppm -m model_final/masknetweight.bin -e LLD_604x960fp32_MKZ.trt -i input_960_604.jpg -o output_LLD_960x604_mulit.ppm

## More detail(Caffe2/detectron):
### 1. Analyzing maskrcnn
* Model builder

```
model_builder.py:
def ResNet50_faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_detection_model(
        model, ResNet.add_ResNet50_conv4_body, ResNet.add_ResNet_roi_conv5_head
    )

model_builder.py : build_generic_detection_model
    # Add the conv body (called "backbone architecture" in papers)
    # E.g., ResNet-50, ResNet-50-FPN, ResNeXt-101-FPN, etc.
    blob_conv, dim_conv, spatial_scale_conv = add_conv_body_func(model)
    ...
    if not model.train:  # == inference
        # Create a net that can be used to execute the conv body on an image
        # (without also executing RPN or any other network heads)
        model.conv_body_net = model.net.Clone('conv_body_net')

ResNet.py: 
    add_ResNet50_conv4_body
    add_ResNet_roi_conv5_head
    AffineChannel renamed to  onnx BatchNormalization(onnx/defs/nn/defs.cc)
    UpsampleNearest renamed to onnx Upsample(onnx/defs/tensor/defs.cc)
 ```

* Inference

```
test.py: im_detect_bbox(model, im, target_scale, target_max_size, boxes=None):
```

### 2. Split MaskRCNN

```
model_net_input = ['gpu_0/data', 'gpu_0/im_info']
model_net_output = ['gpu_0/cls_prob', 'gpu_0/bbox_pred']

splitedNodes = ['gpu_0/fpn_res2_2_sum',  'gpu_0/fpn_res3_3_sum', 'gpu_0/fpn_res4_5_sum', 'gpu_0/fpn_res5_2_sum', 
                'gpu_0/rpn_cls_probs_fpn2', 'gpu_0/rpn_bbox_pred_fpn2', 'gpu_0/rpn_cls_probs_fpn3', 'gpu_0/rpn_bbox_pred_fpn3',
                'gpu_0/rpn_cls_probs_fpn4', 'gpu_0/rpn_bbox_pred_fpn4', 'gpu_0/rpn_cls_probs_fpn5', 'gpu_0/rpn_bbox_pred_fpn5', 
                'gpu_0/rpn_cls_probs_fpn6', 'gpu_0/rpn_bbox_pred_fpn6']
roi_input = splitedNodes + ['gpu_0/im_info']
roi_output = ['gpu_0/cls_prob', 'gpu_0/bbox_pred']
roi_net, roi_outputBlob = model.net.ClonePartial('',roi_input, roi_output)

res_net_input = ['gpu_0/data']
res_net_output = splitedNodes
res_net, res_outputBlob = model.net.ClonePartial('',res_net_input, res_net_output)
```

```
normal:
INFO infer_simple_onnx.py: 247: Processing ../test_pic/test/vlcsnap-00889.jpg -> /tmp/detectron/vlcsnap-00889.jpg.pdf
INFO infer_simple_onnx.py: 255: Inference time: 1.386s
INFO infer_simple_onnx.py: 257:  | im_detect_bbox: 1.361s
INFO infer_simple_onnx.py: 257:  | misc_mask: 0.010s
INFO infer_simple_onnx.py: 257:  | im_detect_mask: 0.009s
INFO infer_simple_onnx.py: 257:  | misc_bbox: 0.004s

detail:(res-net runed 2 times)
INFO infer_simple_onnx.py: 210: Inference time: 2.162s
INFO infer_simple_onnx.py: 212:  | part1_conv_body_net: 0.780s
INFO infer_simple_onnx.py: 212:  | im_detect_bbox: 2.117s
INFO infer_simple_onnx.py: 212:  | im_detect_mask: 0.016s
INFO infer_simple_onnx.py: 212:  | misc_mask: 0.027s
INFO infer_simple_onnx.py: 212:  | misc_bbox: 0.001s
INFO infer_simple_onnx.py: 212:  | part1_res_net: 0.661s
INFO infer_simple_onnx.py: 212:  | part2_roi_net: 0.603s
```

### 3.Run Res-net50 inside MaskRCNN

```
/usr/src/tensorrt/bin$ ./trtexec --onnx=/home/nvidia/model_1280x720.onnx 
Average over 10 runs is 287.759 ms (percentile time is 287.978).
/usr/src/tensorrt/bin$ ./trtexec --onnx=/home/nvidia/model_1280x720.onnx --fp16
Average over 10 runs is 166.708 ms (percentile time is 167.732).
```

### 4.Export part of maskrcnn & convert to onnx format

```
cd ~/caffe2_TRT/maskrcnn_convert
python2 python/infer_simple_onnx.py --cfg ../model/e2e_mask_rcnn_R-50-FPN_2x.yaml --output-dir /tmp/detectron --image-ext jpg --wts ../model/model_final.pkl ../test_pic/test --export
python2 python/infer_simple_onnx.py --cfg ../model/e2e_mask_rcnn_R-50-FPN_2x.yaml --output-dir /tmp/detectron --image-ext jpg --wts ../model/model_final.pkl ../test_pic/test --convert
```

### 5. Add Plugins 
* Use the onnx operator "ATen" for operator RoIAlign, BatchPermutation, GenerateProposalsOp, CollectAndDistributeFpnRpnProposalsOp
* Add Plugins code for GenerateProposalsOp/CollectAndDistributeFpnRpnProposalsOp/RoIAlign/BatchPermutation 
* Add contract layers to reduce the number of input/output to avoid error:

```
../builder/costTensor.h:271: nvinfer1::builder::ChoiceTupleMap<T>::ChoiceTupleMap(unsigned int, T) [with T = float]: Assertion `rank <= ChoiceTuple::kMAX_RANK' failed
```

* The dimensions of inputs/output can not changed when calling TensorRT runtime. Use the max dimensions when building the TensorRT CudaEngine. 

### 6.Load onnx with tensorrt and save CudaEngine as .trt

```
cd ~/caffe2_TRT/maskrcnn_c
./bin/onnx2trt ~/caffe2_TRT/detectron/model.onnx  -o model_net_fp16.trt -b 1 -d 16  -v
./bin/onnx2trt -o ~/model_net_fp16.trt -b 1 -d 16 -p
```

|   Layers     | time    |  Note  |
| --------   | -----:  | :----:  |
|(Layer* 0) [Convolution]   |	6.876ms | |
|... | ... | |
|(Layer* 155) [Convolution]   |	6.359ms | |
|... | ... | |
|(Layer* 179) [Convolution]   |	8.285ms |  ResNet50 - Conv5: tot 	179ms |
|(Layer* 184) [Convolution]	      |  11.545ms| |	  
|(Layer* 185) [Convolution]	      | 44.411ms | |	
|(Layer* 187) [Convolution]   |	44.999ms | |
|(Layer* 194) [Convolution] 	|11.571ms | |
|... | ... | |
|(Layer* 219) [Activation] output  |0.005ms |Rpn_fpn + Conv: tot  143ms |
|(Layer* 225) [Plugin]             |27.781ms | |
|(Layer* 230) [Plugin]              |13.923ms | |
|(Layer* 232) [Fully Connected] 	 |22.924ms | |
|(Layer* 242) [Fully Connected] 	 |0.012ms   |RoiAlign + FC: tot  81ms |
|Time over all layers: |408.251ms |		 fp16 totTime |
|... | ... | |
|Time over all layers: |630.544ms |		 fp32 totTime |

### 7.Run test

```
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:/home/nvidia/caffe2_TRT/maskrcnn_c/bin"

result on TX2 :
original one with caffe2:
INFO infer_tensorrt_c.py: 253:  | im_detect_bbox_part1_res_net: 1.004s
INFO infer_tensorrt_c.py: 253:  | im_detect_bbox_part2_roi_net: 0.180s
INFO infer_tensorrt_c.py: 253:  | im_detect_bbox: 1.259s
INFO infer_tensorrt_c.py: 253:  | im_detect_mask: 0.008s
INFO infer_tensorrt_c.py: 253:  | misc_mask: 0.020s
INFO infer_tensorrt_c.py: 253:  | misc_bbox: 0.002s
INFO infer_tensorrt_c.py: 254: Inference time: 1.291s

 ./bin/maskrcnn_c infer_tensorrt_c 1
c-python-caffe2: 
Inference time: 1.246s
 | im_detect_bbox: 1.244s
 | im_detect_bbox_part1_res_net: 0.991s
 | misc_bbox: 0.001s
 | im_detect_bbox_part2_roi_net: 0.182s
C host Call time = 1.246822 s

 ./bin/maskrcnn_c infer_tensorrt_c 2
c-python-im-tensorrt
Inference time: 0.768s
 | im_detect_bbox: 0.766s
 | im_detect_bbox_part1_res_net: 0.485s
 | misc_bbox: 0.001s
 | im_detect_bbox_part2_roi_net: 0.200s
C host Call time = 0.768531 s

./bin/maskrcnn_c infer_tensorrt_c 3
c-python-data-tensorrt:
Inference time: 0.684s
 | im_detect_bbox: 0.680s
 | im_detect_bbox_part1_res_net: 0.457s
 | misc_bbox: 0.003s
 | im_detect_bbox_part2_roi_net: 0.218s
C host Call time = 0.684489 s
```

