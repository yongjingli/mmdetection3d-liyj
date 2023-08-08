/*
 *  * Copyright (c) 2018, Xpeng Motor. All rights reserved.
 *  * Create @ 2018.11.16 by Zhiwen Cai
 *  * V0.2.0 @ 2019.9.21:
 *  * V0.3.0 @ 2019.10.11: Added Function: GetBufferOfEngine
 *  * V0.3.5 @ 2019.11.12: Combined Roi layers to one Plugin
 *  * V0.4.0 @ 2020.01.03: Get Stream from GetBufferOfEngine
 *  * V0.4.1 @ 2020.01.21: Optimized for MOD of xpmodel
 *  * V0.4.2 @ 2020.02.15: Added plugin BatchConcatPad
 *  * V0.4.3 @ 2020.03.30: Supported to save input data for int8-calibration
 *  * V0.5.0 @ 2020.04.30: Supported convert NvMedia Safe DLA
 *  * V0.6.0 @ 2020.06.28: Added ConvLSTM Plugin, optimized int8+fp16 configure. Set layer name.
 *  * V0.6.1 @ 2020.08.05: Used one buffer for GPU, and copy output only once. Adjust scale for Upsample.
 *  * V0.6.2 @ 2020.08.11: CreateEngine supported buffer input.
 *  * V0.7.0 @ 2020.08.19: Added ConfidenceFilter plugin to reduce output size of MOD/SOD.
 *  * V0.7.1 @ 2020.08.21: Upgraded buffer order for AP+FSD model, added BufferType, buffer name. Fixed some small bugs.
 *  * v0.7.2 @ 2020.08.29: Optimized warming up. Merged with Lamei's nms optimization. Fixed bug of buffer.
 *  * v0.7.3 @ 2020.09.04: Reduced memory copy and alloc of GPU/CPU buffer for AP+FSD. Reorder output buffer.
 *  * v0.7.4 @ 2020.09.18: Combined Channel2Spatial  to CondifienceFilter plugin.
 *  * v0.7.5 @ 2020.09.28: Supported MatMul + BatchNormal in new AP+FSD model. Add model description.
 *  * v0.7.6 @ 2020.10.15: Supported Char/Text output. Add model description.
 *  * V0.8.0 @ 2020.12.17: Change -std=c++11 to -std=c++14. Updated face sample. Supported int8 calibration for ConvLSTM
 * Plugin.
 *  * V0.8.1 @ 2020.12.29: Updata dla lib and face sample to avoid memcpy from gpu to cpu. Update ONNXTRT_MAX_BUFFERTYPE
 * and nBufferType in EngineBuffer struct . Add sync before memcpy in mode inference. Update test_multi_models
 *  * V0.8.2 @ 2021.01.15: Added "nTensorFormat" in EngineBuffer, for int8 input and DLA.
 *  * V0.8.3 @ 2021.01.21: Supported Math op like 'Sin','Cos','ArgMax'
 *  * V0.8.4 @ 2021.01.27: Confidence_filter miss sync of output when batch=2(>= 0.7.6). Revert retinanet_nms.cu to OTA2
 * version(0.7.4)
 */
#ifndef _ONNXTRT_RCNN_ONNXTRT_H_
#define _ONNXTRT_RCNN_ONNXTRT_H_

#ifdef __cplusplus
extern "C" {
#endif

#define ONNXTRT_VERSION_STRING "V0.8.4"
#define ONNXTRT_OK 0
#define ONNXTRT_GENEREL_ERR -1    // General error, no detail yet.
#define ONNXTRT_INVALID_ERR -2    // invalid resource/ID, not free or idle.
#define ONNXTRT_PARAMETER_ERR -3  // Error paramenter/argument.
#define ONNXTRT_IO_ERR -4         // Error when read/write buffer or file.

#define ONNXTRT_MAX_ENGINENUM 32   // Max number of engine ID
#define ONNXTRT_MAX_BUFFERNUM 32   // Max number of EngineBuffer
#define ONNXTRT_MAX_BUFFERDIMS 6   // Max number of dimensions for EngineBuffer
#define ONNXTRT_MAX_BUFFERTYPE 16  // Max number of buffer type

typedef struct {        // size: 64byte
  int nBufferType;      // 0: gpu_input, 1: gpu_output, 2: gpu_variable ,10:cpu_input ,11:cpu_output
  short nDataType;      // 0: FP32/kFLOAT, 1: FP16/kHALF, 2: INT8, 3: kINT32
  short nTensorFormat;  // 0: kLINEAR/kNCHW, 1: kCHW2, 2: kHWC8, 3: kCHW4 4: kCHW16 5: kCHW32
  int nBufferSize;
  int resurved;
  void *p;  // inner buffer pointer
  int nDims;
  int nMaxBatch;
  int d[ONNXTRT_MAX_BUFFERDIMS];
  const char *name;
} EngineBuffer;

/*
 * ch: channel ID for multiple models:
    >=0:Fixed ID, must < ONNXTRT_MAX_ENGINENUM
    -1: Use available ID
 * engine_filename: tensorrt engine file or buffer
 * pMaskWeight: mask weight for maskrcnn/retinamask
 * return: >=0: the channel ID, successfully created . <0 : Error
 */
int CreateEngine(int ch, const char *engine_filename, const char *pMaskWeight);

/*
 * ch: channel ID for multiple models:
 * batchsize: batch size of inference. Must <= max_batch_size of model.
 * inputData: buffer for input, on CPU or GPU
 * inputType: 0:cpu float32, 1:gpu float32, 2: xxx
 * outData: buffer for output, on CPU only
 * outType: 0, 1, 2 , ...
   0: copy of all buffers to cpu: buf0.bs0, buf0.bs1, buf0.bs0, buf1.bs1, ...
   1: packed roi & mask, for maskrcnn/retinamask
   2: packed buffers by batchsize: buf0.bs0, buf1.bs0,..., buf0.bs1 ,buf1.bs1,
   3: optimizied for MOD of xpmodel
 */
int RunEngine(int ch, int batchsize, char *inputData, int inputType, char *outData, int outType);

/*
 * ch: channel ID for multiple models
 */
int DestoryEngine(int ch);

/*
 * ch: channel ID for multiple models:
 * bufferInfo: point to the bufferInfo
 * pNum: the number of dims returned.
 * pStream: the pointer to cudaStream_t
 */
int GetBufferOfEngine(int ch, EngineBuffer **bufferInfo, int *pNum, void **pStream);

int ParseEngineData(void *input, void *output, int batchSize, int outType);

// function prototypes for dlopen & dlsym

typedef int (*PTCreateEngine)(int ch, const char *pModelName, const char *pMaskWeight);

typedef int (*PTRunEngine)(int ch, int batch, char *inputData, int inputType, char *outData, int outType);

typedef int (*PTDestoryEngine)(int ch);

typedef int (*PTGetBufferOfEngine)(int ch, EngineBuffer **bufferInfo, int *pNum, void **pStream);

#ifdef __cplusplus
}
#endif

#endif  //_ONNXTRT_RCNN_ONNXTRT_H_
