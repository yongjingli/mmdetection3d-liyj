/*
 * Copyright (c) 2018, Xpeng Motors. All rights reserved.
 *
 */

#pragma once

#include <assert.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <vector>

//#include "common.h"
#include "NvInfer.h"
#include "NvUffParser.h"
using namespace nvinfer1;
using namespace nvuffparser;

extern int TRT_DEBUGLEVEL;  // 4:VERBOSE, 3:DEBUG, 2:INFO, 1:WARN, 0:ERROR
#define DPRINTF(level, x...)         \
  do {                               \
    if ((level) <= TRT_DEBUGLEVEL) { \
      printf(x);                     \
    }                                \
  } while (0)

#if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH <= 1
#define FIX_BUG_TENOSRRT5 \
  1  // just for tensorrt5.0.0.x, must be 0 for tensorrt4.x or tensorrt5.0.2.x
#endif

// Refer from sampleUffSSD.cpp
inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
#if NV_TENSORRT_MAJOR >= 4
    case nvinfer1::DataType::kINT32:
      return 4;
#endif
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

inline int64_t volume(const nvinfer1::Dims &d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

extern "C" int filteMOD(int batch, const void *inputbuf, void *outputbuf,
                        const nvinfer1::Dims dim, float score_thresh,
                        void *workspace, size_t ws_size, cudaStream_t stream);
