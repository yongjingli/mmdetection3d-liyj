#ifndef __INCLUDE_CONV_RELU_H__
#define __INCLUDE_CONV_RELU_H__
/* For mask-net of maskrcnn
 * Copyright (c) 2018, Xiaopeng. All rights reserved.
 * Create by caizw @ 2018.9.20
 * details of RunMaskNet:
 * "rois"/"bbox_pred"/"cls_prob" ->	bbox_transform -> nms -> FPN level ->
 * "mask_rois_fpnX" "mask_rois_fpnX" ->	RoiAlign -> Concat -> BatchPermutation
 * -> "_[mask]_rois_feat"
 * "_[mask]_rois_feat" -> Conv/Relu -> Conv/Relu -> Conv/Relu ->
 * ConvTranspose/Relu -> Conv/Sigmoid -> "mask_fcn_probs"
 */

extern "C" {

/*
pWeight[in]: pointer to the buffer or file name of weight
InputSize[in]: size of input data ( in bytes )
maxbatch[in]: max batch size ( as the maximum number of rois to detect )
nOutLayer[in]: set the output layer ( fixed to 6 in this case and 0~5 for debug
only )
*/
int InitMaskNet(void *pWeight, int InputSize, int maxbatch, int nOutLayer);

int ForwardMaskNet(void *pInput, int InputSize, int nbatch, void *pOutput,
                   int OutputSize);

int DestroyMaskNet();

/*
inputsGPU[in]: inputs from tensorrt (binding buffers of engine)
bufferDims[in]: dimensions of buffers
output[out]: outbuffer(CPU)
outputGPU[out]:outbuffer(GPU)
stream[in]: cuda stream
*/
void RunMaskNet(std::vector<void *> &inputsGPU, nvinfer1::Dims *bufferDims,
                float *output, void *outputGPU, cudaStream_t stream);

/*
imageHeight: height of image
imageWidth: width of image
height64, width64, im_scale: input data from im_info
*/
void SetImInfo(int imageHeight, int imageWidth, int height64, int width64,
               float im_scale);
}

void findContours(unsigned char *InputImage, int Width_i, int Height_i,
                  int pad_height, int pad_width, void *contours);

#endif  //__INCLUDE_CONV_RELU_H__
