/*
 *  * Copyright (c) 2021, Xpeng Motor. All rights reserved.
 *  * Create @ 2018.11.16  by Zhiwen Cai
 *  * V1.0.0 @ 2021.7.1   : Copy from PDK5112, for PDK526 standard
 *  * V1.0.1 @ 2021.7.23  : Upgraded plugin code, confidenceFilter.
 *  * V1.0.2 @ 2021.8.24  : Upgraded BEV filter for mod3d(cngpV7.x).
 *  * V1.0.3 @ 2021.9.3   : Add Res2Channel Pad
 *  * V1.0.4 @ 2021.9.18  : Upgraded BEV filter with 1 input. Fixed bugs of Res2channel for input type 3 (RGB)
 *  * V1.0.5 @ 2021.11.9  : Add more Cuda Check log. Tools support to modifiy CudaGraph output.
 *  * V1.0.6 @ 2022.3.4   : Upgraded to adapt TensorRT 8.3
 *  * V1.0.7 @ 2022.3.18  : Upgraded ConvLSTM, fix Upsample, BatchConcatPad, and add FusedMultionHeadAttention plugin
 *  * V1.0.8 @ 2022.4.27  : Support adaptive explicit/implicit batch, no change of onnx;
 *                          Add GridSample, YoloLayer, MSDeformableAttention plugin
 *  * V1.0.9 @ 2022.5.26  : Fix DlaTool for GPU buffer input and CPU buffer output. adapt TensorRT 8.4.10
 */

#ifndef _ONNXTRT_RCNN_ONNXTRT_H_
#define _ONNXTRT_RCNN_ONNXTRT_H_

#ifdef __cplusplus
extern "C" {
#endif

#define ONNXTRT_VERSION_STRING "V1.0.9.orin"
#define ONNXTRT_OK 0
#define ONNXTRT_GENEREL_ERR -1    // General error, no detail yet.
#define ONNXTRT_INVALID_ERR -2    // invalid resource/ID, not free or idle.
#define ONNXTRT_PARAMETER_ERR -3  // Error paramenter/argument.
#define ONNXTRT_IO_ERR -4         // Error when read/write buffer or file.

#define ONNXTRT_MAX_ENGINENUM 16   // Max number of engine ID
#define ONNXTRT_MAX_BATCHNUM  32   // Max number of batch for an engine
#define ONNXTRT_MAX_BUFFERNUM 32   // Max number of EngineBuffer
#define ONNXTRT_MAX_BUFFERDIMS 6   // Max number of dimensions for EngineBuffer
#define ONNXTRT_MAX_BUFFERTYPE 16  // Max number of buffer type

#define ONNXTRT_CONVLSTM_MEMORY (1<<16) // ConvLSTM flag in inputType. 1: Use memory of prev-frame data.   Else: Use zeros as prev-frame data (for first frame). 

typedef struct {        // size: 64byte
  int nBufferType;      // 0: gpu_input, 1: gpu_output, 2: gpu_variable ,10:cpu_input ,11:cpu_output
  short nDataType;      // 0: FP32/kFLOAT, 1: FP16/kHALF, 2: INT8, 3: kINT32
  short nTensorFormat;  // 0: kLINEAR/kNCHW, 1: kCHW2, 2: kHWC8, 3: kCHW4 4: kCHW16 5: kCHW32
  int nBufferSize;
  int resurved;
  void *p;  // inner buffer pointer
  int nDims;
  int nMaxBatch;
  // char nMaxBatch;
  // unsigned char nBatchGroupMask; // 0: not used; every bit means whether the group is used
  // short resurved_2;
  int d[ONNXTRT_MAX_BUFFERDIMS];
  const char *name;
} EngineBuffer;

typedef struct {
  void*  pointCloud;
  int    pointNum;
  int    pointType; // 3:XYZ, 4:XYZI 
} LidarInput;

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
 * inputType: 0:cpu float32, 1:gpu float32, 2: xxx, -1: skip launch with input
 * outData: buffer for output, on CPU only
 * outType: 0, 1, 2 , ...
   0: copy of all buffers to cpu: buf0.bs0, buf0.bs1, buf0.bs0, buf1.bs1, ...
   1: packed roi & mask, for maskrcnn/retinamask
   2: packed buffers by batchsize: buf0.bs0, buf1.bs0,..., buf0.bs1 ,buf1.bs1,
   3: optimizied for Filter in DLA model
   -1: skip stream sync and copy output.
   -2: copy output but skip stream sync ( work with cuLaunchHostFunc )
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

//These two functions are just for test only
/*
 * allocate GPU space
 */
void* AllocateSpaceGPU(size_t bytes, int num);

/*
 * Copy host to device
 */
void MemcpyHost2DeviceGPU(void* dst, void* src, size_t bytes, int num);

void FreeSpaceGPU(void* gpuPtr);

typedef void* (*PTAllocateSpaceGPU)(size_t bytes, int num);

typedef void  (*PTMemcpyHost2DeviceGPU)(void* dst, void* src, size_t bytes, int num);

typedef void  (*PTFreeSpaceGPU)(void* gpuPtr);

#ifdef __cplusplus
}
#endif

#endif  //_ONNXTRT_RCNN_ONNXTRT_H_
