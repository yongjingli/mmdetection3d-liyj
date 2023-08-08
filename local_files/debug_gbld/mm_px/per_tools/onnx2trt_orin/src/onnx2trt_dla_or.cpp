/*
 * Copyright (c) 2020, Xpeng Motor. All rights reserved.
 * libonnxtrt_dla.so : support NvMeadia DLA runtime.
 * 2020-5-8 : Create.
 */

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fstream>
#include <iostream>

#include "DlaTool.h"
#include "onnxtrt.h"
#include "utils.h"

class IDLAWorkspace {
 private:
  int status = 0;  //-1, 0: not inited, >=1: inited,
  int ch = 0;
  DlaTool *dla_ptr = nullptr;

  // index for buffer, usefull when different device has different buffer order
  std::vector<int> bufferIndex;
  int nbBindings = 0;
  int nMaxBatchsize = 0;
  EngineBuffer bufferInfo[ONNXTRT_MAX_BUFFERNUM];
  cudaStream_t stream = nullptr;
  int frameID = 0;
  std::string savePath = "";  // save input data

 public:
  IDLAWorkspace() { memset(bufferInfo, 0, sizeof(bufferInfo)); };
  ~IDLAWorkspace() { release(); };

  void setStatus(int s) { status = s; }
  bool isInited() { return (nullptr != dla_ptr && status > 0); }
  int init(int setch, uint32_t numTasks, std::string loadableName);

  void doInference(void *inputData, void *outData, int batchSize = 1, int inputType = 0, int outType = 0);

  void release();

  // get input/output buffer infomation ; get cudaStream ;
  int getBufferInfo(EngineBuffer **pBufferInfo, int *pNum, void **pStream) {
    if (NULL == pNum || NULL == pBufferInfo) return ONNXTRT_PARAMETER_ERR;
    if (NULL != pStream) *pStream = stream;
    *pNum = nbBindings;
    *pBufferInfo = bufferInfo;
    return 0;
  }

  int saveCPUBuf(char *cpuBuf, int64_t size) {
    if (cpuBuf == nullptr || size <= 0) return -1;

    char outfile[260];
    snprintf(outfile, 256, "%s/DLA%d_%d.bin", savePath.c_str(), ch, frameID);
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file.write(cpuBuf, size);
    file.close();

    LOG_INFO("Save DLA input %s size=%ld\n", outfile, size);
    return size;
  }
};

int IDLAWorkspace::init(int setch, uint32_t numTasks, std::string loadableName) {
  bool testPing = false;
  int dlaId = setch % 2;
  cudaStreamCreate(&stream);
  dla_ptr = new DlaTool(dlaId, numTasks, loadableName, testPing, stream);
  NvMediaStatus nvmstatus = dla_ptr->SetUp();
  if (nvmstatus != NVMEDIA_STATUS_OK) {
    LOG_ERR("DLA setup fails \n");
    return ONNXTRT_PARAMETER_ERR;
  }

  nbBindings = dla_ptr->GetBufferInfo(bufferInfo);
  nMaxBatchsize = bufferInfo[0].nMaxBatch;

  return 0;
}

void IDLAWorkspace::doInference(void *inputData, void *outData, int batchSize, int inputType, int outType) {
  if (nbBindings < 2) return;
  int inSize = (inputType >= 0) ? bufferInfo[0].nBufferSize : 0;
  // Support multi output
  int outSize = 0;
  if (outType >= 0) {
    for (auto bufferInfo_ : bufferInfo) {
      if (11 == bufferInfo_.nBufferType || 1 == bufferInfo->nBufferType) {  // cpu output or gpu output
        outSize += bufferInfo_.nBufferSize;
      }
    }
  }
  // int outSize = (outType >= 0) ? bufferInfo[1].nBufferSize : 0;
  NvMediaStatus nvmstatus = dla_ptr->Run(inputData, inSize, inputType, outData, outSize, outType);
  if (nvmstatus != NVMEDIA_STATUS_OK) {
    LOG_ERR("DLA Run fails \n");
    // status = 2;
  }
}

void IDLAWorkspace::release() {
  status = -2;
  nbBindings = 0;
  if (nullptr != dla_ptr) {
    delete dla_ptr;
    dla_ptr = nullptr;
  }
  status = 0;
}

const int MAX_BATCH = ONNXTRT_MAX_ENGINENUM;
const int MAX_CH = ONNXTRT_MAX_ENGINENUM;
static IDLAWorkspace gworkspace[MAX_CH];

// interface for so/python
/*
 * ch: channel ID for multiple models:  0:maskrcnn, 1:resnet ...
 * engine_filename: tensorrt engine file
 * pMaskWeight: mask weight for maskrcnn/retinamask
 */
extern "C" int CreateEngine(int ch, const char *engine_filename, const char *pMaskWeight) {
  if (ch >= 0 && ch < ONNXTRT_MAX_ENGINENUM) {
    if (gworkspace[ch].isInited()) return -2;
  } else if (-1 == ch) {  //-1: Find available ID
    for (int i = ONNXTRT_MAX_ENGINENUM - 1; i >= 0; i--) {
      if (!gworkspace[i].isInited()) {
        ch = i;
        break;
      }
    }
    if (-1 == ch) return -2;
  } else
    return ONNXTRT_PARAMETER_ERR;

  gworkspace[ch].init(ch, ONNXTRT_MAX_ENGINENUM - 1, engine_filename);
  gworkspace[ch].setStatus(1);

  return ch;
}

// Supported multiply input and output Tensor on DLA.
// But only copy one buffer throw interface "RunEngine".
// Please Get multi-IO pointers throw GetBufferOfEngine to avoid memcpy.
// inputType: 0:cpu buffer, 1:gpu buffer,<0: avoid memcpy and inference
// outType: 0:cpu buffer, 1:gpu buffer,<0: avoid memcpy and sync
extern "C" int RunEngine(int ch, int batch, char *inputData, int inputType, char *outData, int outType) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }

  gworkspace[ch].doInference(inputData, outData, batch, inputType, outType);
  return ONNXTRT_OK;
}

extern "C" int DestoryEngine(int ch) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }
  gworkspace[ch].release();

  return ONNXTRT_OK;
}

extern "C" int GetBufferOfEngine(int ch, EngineBuffer **pBufferInfo, int *pNum, void **pStream) {
  return gworkspace[ch].getBufferInfo(pBufferInfo, pNum, pStream);
}

// static funtion called when using dlopen & dlclose
static __attribute__((constructor)) void lib_init(void) {
  int TRT_DEBUGLEVEL = 1;
  {
    char *val = getenv("TRT_DEBUGLEVEL");
    if (NULL != val) {
      TRT_DEBUGLEVEL = atoi(val);
      SET_LOG_LEVEL((CLogger::LogLevel)TRT_DEBUGLEVEL);
    }
  }
  printf("Load onnx2trt DLA lib %s built@%s %s DebugLevel=%d\n", ONNXTRT_VERSION_STRING, __DATE__, __TIME__,
         TRT_DEBUGLEVEL);
}

static __attribute__((destructor)) void lib_deinit(void) {
  printf("Unload onnx2trt DLA lib %s built@%s %s\n", ONNXTRT_VERSION_STRING, __DATE__, __TIME__);
}
