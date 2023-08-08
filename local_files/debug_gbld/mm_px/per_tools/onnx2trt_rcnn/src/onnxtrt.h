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
 */
#ifndef _ONNXTRT_RCNN_ONNXTRT_H_
#define _ONNXTRT_RCNN_ONNXTRT_H_

#ifdef __cplusplus
extern "C" {
#endif

#define ONNXTRT_VERSION_STRING "V0.5.0"
#define ONNXTRT_OK 0
#define ONNXTRT_GENEREL_ERR -1    // General error, no detail yet.
#define ONNXTRT_INVALID_ERR -2    // invalid resource/ID, not free or idle.
#define ONNXTRT_PARAMETER_ERR -3  // Error paramenter/argument.
#define ONNXTRT_IO_ERR -4         // Error when read/write buffer or file.

#define ONNXTRT_MAX_ENGINENUM 32  // Max number of engine ID
#define ONNXTRT_MAX_BUFFERNUM 32  // Max number of EngineBuffer
#define ONNXTRT_MAX_BUFFERDIMS 8  // Max number of dimensions for EngineBuffer
typedef struct {                  // size: 64byte
  int nBufferType;                // 0: input, 1: output
  int nDataType;  // 0: FP32/kFLOAT, 1: FP16/kHALF, 2: INT8, 3: kINT32
  int nBufferSize;
  int resurved;
  void *p;  // inner buffer pointer
  int nDims;
  int nMaxBatch;
  int d[ONNXTRT_MAX_BUFFERDIMS];
} EngineBuffer;

/*
 * ch: channel ID for multiple models:
    >=0:Fixed ID, must < ONNXTRT_MAX_ENGINENUM
    -1: Use available ID
 * engine_filename: tensorrt engine file
 * pMaskWeight: mask weight for maskrcnn/retinamask
 * return: >=0: the channel ID successfully created . <0 : Error
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
int RunEngine(int ch, int batchsize, char *inputData, int inputType,
              char *outData, int outType);

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
int GetBufferOfEngine(int ch, EngineBuffer **bufferInfo, int *pNum,
                      void **pStream);

int ParseEngineData(void *input, void *output, int batchSize, int outType);

// function prototypes for dlopen & dlsym

typedef int (*PTCreateEngine)(int ch, const char *pModelName,
                              const char *pMaskWeight);

typedef int (*PTRunEngine)(int ch, int batch, char *inputData, int inputType,
                           char *outData, int outType);

typedef int (*PTDestoryEngine)(int ch);

typedef int (*PTGetBufferOfEngine)(int ch, EngineBuffer **bufferInfo, int *pNum,
                                   void **pStream);

#ifdef __cplusplus
}
#endif

#endif  //_ONNXTRT_RCNN_ONNXTRT_H_
