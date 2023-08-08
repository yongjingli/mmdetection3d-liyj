/*
 * Copyright (c) 2018, Xpeng Motor. All rights reserved.
 * libonnxtrt_lite.so : simple version without any plugin\profile addin.
 * 2019-5-16 : remove any c++ library like vecotr/fstream/cout.
 */
// Refer from tensorrt4.0 trtexec.cpp
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "onnxtrt.h"

int TRT_DEBUGLEVEL;  // 4:VERBOSE, 3:DEBUG, 2:INFO, 1:WARN, 0:ERROR
#define DPRINTF(level, x...)         \
  do {                               \
    if ((level) <= TRT_DEBUGLEVEL) { \
      printf(x);                     \
    }                                \
  } while (0)

#define CHECK(status)                     \
  {                                       \
    if (status != 0) {                    \
      printf("Cuda failure: %d", status); \
      abort();                            \
    }                                     \
  }

const int MAX_CH = 16;
const int MAX_BIND = 32;
typedef struct {
  int nbBindings;  // must < MAX_BIND
  void *buffers[MAX_BIND];
  int64_t buffersSizes[MAX_BIND];
  int bufferIndex[MAX_BIND];  // index for buffer,
} IWorkspace;

IWorkspace gworkspace[MAX_CH];

// interface for so/python
extern "C" int CreateEngine(int ch, const char *engine_filename,
                            const char *pMaskWeight) {
  DPRINTF(3, "Enter CreateEngine \n");
  {
    char *val = getenv("TRT_DEBUGLEVEL");
    if (NULL != val) {
      TRT_DEBUGLEVEL = atoi(val);
    }
  }

  return 0;
}

// inputType: 0:cpu float32, 1:gpu float32, 2: xxx
// outType: 0:cpu roi & feat map, 1: cpu roi & mask
extern "C" int RunEngine(int ch, int batch, char *inputData, int inputType,
                         char *outData, int outType) {
  DPRINTF(3, "Enter RunEngine \n");

  return 0;
}

extern "C" int DestoryEngine(int ch) {
  DPRINTF(3, "Enter DestoryEngine \n");

  return 0;
}

extern "C" int GetBufferOfEngine(int ch, EngineBuffer **pBufferInfo, int *pNum, void **pStream) {
  return 0;
}

extern "C" int ParseEngineData(void *input, void *output, int batchSize, int outType) { return -1; }

//These two functions are for test multi model
extern "C" void* AllocateSpaceGPU(size_t bytes, int num) {
  void* gpuTmpPtr = nullptr;

  return gpuTmpPtr;
}

extern "C" void MemcpyHost2DeviceGPU(void* dst, void* src, size_t bytes, int num) {
  
}